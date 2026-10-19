"""
UniBox Hybrid Ingestion Pipeline.

Orchestrates the full ingestion workflow from a single JSON seed registry
(``data/seed_sources.json``) that specifies three source types:

``url``
    Web pages crawled via ``crawl4ai`` (HTML → markdown → chunks).
``pdf_url``
    Remote PDFs downloaded and extracted via :mod:`ingestion.pdf_loader`.
``pdf_file``
    Local PDFs (relative to the project root) extracted via the same loader.

All extracted text is routed through :func:`ingestion.chunker.chunk_text`
with an explicit semantic ``title`` so embeddings are context-aware, then
upserted into MongoDB Atlas in configurable batch sizes.

Usage::

    # Run from the repository root
    python scripts/run_ingestion.py

    # Dry-run (crawl + chunk + embed, but skip MongoDB writes)
    python scripts/run_ingestion.py --dry-run

    # Filter to a subset of tags defined in seed_sources.json
    python scripts/run_ingestion.py --tags kjsce fees

Exit codes
----------
0  — pipeline completed (even if some individual sources failed; see summary).
1  — fatal error (registry missing, env vars absent, etc.).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — ensure the repo root is on sys.path when the script is
# invoked directly (python scripts/run_ingestion.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingestion.chunker import chunk_text
from ingestion.crawler import crawl_urls
from ingestion.embedder import generate_embeddings
from ingestion.pdf_loader import load_pdf_text
from database.mongo_client import insert_documents

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("unibox.ingestion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEED_REGISTRY_PATH: Path = _PROJECT_ROOT / "data" / "seed_sources.json"
_MONGO_BATCH_SIZE: int = 100


# ---------------------------------------------------------------------------
# Registry loading & filtering
# ---------------------------------------------------------------------------

def load_seed_registry(
    registry_path: Path,
    filter_tags: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse ``data/seed_sources.json`` and return filtered HTML and PDF sources.

    Args:
        registry_path:
            Absolute path to the JSON seed registry.
        filter_tags:
            When provided, only entries whose ``tags`` list contains *at least
            one* of the supplied tags are included.  ``None`` means no filter.

    Returns:
        A tuple ``(html_sources, pdf_sources)`` — each is a list of entry dicts
        whose ``active`` flag is ``true`` (and whose tags match the filter).

    Raises:
        FileNotFoundError: If the registry file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Seed registry not found at: {registry_path}\n"
            "Expected: data/seed_sources.json relative to the project root."
        )

    with registry_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    def _keep(entry: dict[str, Any]) -> bool:
        if not entry.get("active", True):
            return False
        if filter_tags:
            return bool(set(entry.get("tags", [])) & set(filter_tags))
        return True

    html_sources: list[dict[str, Any]] = [
        e for e in data.get("html_sources", []) if _keep(e)
    ]
    pdf_sources: list[dict[str, Any]] = [
        e for e in data.get("pdf_sources", []) if _keep(e)
    ]

    logger.info(
        "Registry loaded — %d HTML source(s), %d PDF source(s) will be processed.",
        len(html_sources),
        len(pdf_sources),
    )
    return html_sources, pdf_sources


# ---------------------------------------------------------------------------
# Document representation
# ---------------------------------------------------------------------------

class IngestableDocument:
    """
    Lightweight value object representing a single extracted document
    before chunking.

    Attributes:
        url:     Canonical identifier (file path or page URL) used as the
                 deduplication key in MongoDB alongside ``chunk_index``.
        title:   Semantic title passed to ``chunk_text(title=...)`` so every
                 chunk embedding is context-anchored.
        content: Raw extracted text.
        source_type: ``"html"`` or ``"pdf"`` — recorded for diagnostics.
    """

    __slots__ = ("url", "title", "content", "source_type")

    def __init__(
        self,
        url: str,
        title: str,
        content: str,
        source_type: str,
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.source_type = source_type

    def __repr__(self) -> str:
        return (
            f"IngestableDocument(source_type={self.source_type!r}, "
            f"url={self.url!r}, content_len={len(self.content)})"
        )


# ---------------------------------------------------------------------------
# HTML ingestion (crawl4ai)
# ---------------------------------------------------------------------------

async def collect_html_documents(
    html_sources: list[dict[str, Any]],
) -> list[IngestableDocument]:
    """
    Crawl all active HTML seed URLs and return :class:`IngestableDocument` objects.

    The seed titles from the registry are preferred over page ``<title>`` tags
    because registry titles are curated for semantic clarity.  If the registry
    title is empty, the crawled page metadata title is used instead.

    Args:
        html_sources: Active ``url`` entries from the seed registry.

    Returns:
        Flat list of :class:`IngestableDocument` for every successfully
        crawled page (including pages discovered via BFS depth crawl).
    """
    if not html_sources:
        return []

    seed_urls: list[str] = [entry["source"] for entry in html_sources]

    # Build a lookup: seed_url → registry title for title enrichment.
    seed_title_map: dict[str, str] = {
        entry["source"]: entry.get("title", "") for entry in html_sources
    }

    logger.info("[HTML] Starting crawl of %d seed URL(s).", len(seed_urls))
    crawled_pages = await crawl_urls(seed_urls)
    logger.info("[HTML] Crawl complete — %d page(s) retrieved.", len(crawled_pages))

    documents: list[IngestableDocument] = []
    for page in crawled_pages:
        # Prefer the crawled page's own title (more specific for deep pages),
        # but fall back to the registry title for the seed URL itself.
        page_title: str = page.get("title", "").strip()
        if not page_title:
            page_title = seed_title_map.get(page["url"], page["url"])

        content: str = page.get("content", "").strip()
        if not content:
            logger.debug("[HTML] Skipping empty page: %s", page["url"])
            continue

        documents.append(
            IngestableDocument(
                url=page["url"],
                title=page_title,
                content=content,
                source_type="html",
            )
        )

    return documents


# ---------------------------------------------------------------------------
# PDF ingestion (pdf_loader)
# ---------------------------------------------------------------------------

def collect_pdf_documents(
    pdf_sources: list[dict[str, Any]],
) -> list[IngestableDocument]:
    """
    Extract text from all active PDF sources (local files and remote URLs).

    Each source is processed independently; failures are logged and skipped
    rather than aborting the entire pipeline.

    Args:
        pdf_sources: Active ``pdf_url`` and ``pdf_file`` entries from the
                     seed registry.

    Returns:
        List of :class:`IngestableDocument` for each successfully extracted PDF.
    """
    if not pdf_sources:
        return []

    documents: list[IngestableDocument] = []

    for entry in pdf_sources:
        source: str = entry["source"]
        title: str = entry.get("title", source)
        source_type: str = entry.get("type", "pdf_url")

        # Resolve local paths relative to the project root.
        resolved_source: str = source
        if source_type == "pdf_file":
            local_path = _PROJECT_ROOT / source
            if not local_path.exists():
                logger.warning(
                    "[PDF] Local file not found, skipping: %s",
                    local_path,
                )
                continue
            resolved_source = str(local_path)

        logger.info("[PDF] Loading: %s", resolved_source)
        try:
            text = load_pdf_text(resolved_source)
        except FileNotFoundError:
            logger.warning("[PDF] File not found — skipping: %s", resolved_source)
            continue
        except Exception as exc:
            logger.error(
                "[PDF] Failed to load %r: %s — skipping.", resolved_source, exc
            )
            continue

        if not text.strip():
            logger.warning("[PDF] No text extracted from %r — skipping.", resolved_source)
            continue

        logger.info(
            "[PDF] Extracted %d characters from %r.",
            len(text),
            resolved_source,
        )
        documents.append(
            IngestableDocument(
                url=resolved_source,
                title=title,
                content=text,
                source_type="pdf",
            )
        )

    return documents


# ---------------------------------------------------------------------------
# Chunking → embedding → MongoDB upsert
# ---------------------------------------------------------------------------

def chunk_and_embed(
    documents: list[IngestableDocument],
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Convert raw documents into embedded MongoDB chunks and upsert them.

    For every document:
    1. ``chunk_text(text, title)`` splits the content into overlapping
       token-window chunks prefixed with ``"Title: {title} | Content: "``.
    2. ``generate_embeddings(chunks)`` produces 384-dim all-MiniLM vectors.
    3. Chunks are collected into batches of :data:`_MONGO_BATCH_SIZE` and
       upserted via ``insert_documents``.

    When *dry_run* is ``True``, steps 1–2 run normally but step 3 is skipped,
    allowing pipeline validation without touching the database.

    Args:
        documents: Flat list of :class:`IngestableDocument` to process.
        dry_run:   Skip MongoDB writes when ``True``.

    Returns:
        Summary dict with keys ``"total_chunks"``, ``"inserted"``,
        ``"updated"``, ``"failed_docs"``.
    """
    mongo_batch: list[dict[str, Any]] = []
    total_chunks = 0
    total_inserted = 0
    total_updated = 0
    failed_docs = 0

    for doc in documents:
        logger.info(
            "[CHUNK] Processing %s doc: %s (%d chars)",
            doc.source_type.upper(),
            doc.url,
            len(doc.content),
        )

        try:
            chunks: list[str] = chunk_text(text=doc.content, title=doc.title)
        except Exception as exc:
            logger.error("[CHUNK] Failed to chunk %r: %s — skipping.", doc.url, exc)
            failed_docs += 1
            continue

        if not chunks:
            logger.warning("[CHUNK] No chunks produced for %r — skipping.", doc.url)
            continue

        logger.info("[CHUNK] %d chunk(s) from %r.", len(chunks), doc.url)

        try:
            embeddings: list[list[float]] = generate_embeddings(chunks)
        except Exception as exc:
            logger.error(
                "[EMBED] Failed to embed chunks for %r: %s — skipping.", doc.url, exc
            )
            failed_docs += 1
            continue

        if len(embeddings) != len(chunks):
            logger.warning(
                "[EMBED] Mismatch: %d chunks vs %d embeddings for %r — skipping.",
                len(chunks),
                len(embeddings),
                doc.url,
            )
            failed_docs += 1
            continue

        for chunk_index, (chunk_text_val, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            mongo_batch.append(
                {
                    "url": doc.url,
                    "title": doc.title,
                    "text": chunk_text_val,
                    "chunk_index": chunk_index,
                    "source_type": doc.source_type,
                    "embedding": embedding,
                }
            )
            total_chunks += 1

            # Flush batch when it reaches the configured size
            if len(mongo_batch) >= _MONGO_BATCH_SIZE:
                inserted, updated = _flush_batch(mongo_batch, dry_run)
                total_inserted += inserted
                total_updated += updated
                mongo_batch.clear()

    # Flush any remaining partial batch
    if mongo_batch:
        inserted, updated = _flush_batch(mongo_batch, dry_run)
        total_inserted += inserted
        total_updated += updated

    return {
        "total_chunks": total_chunks,
        "inserted": total_inserted,
        "updated": total_updated,
        "failed_docs": failed_docs,
    }


def _flush_batch(
    batch: list[dict[str, Any]],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Upsert a single batch of chunk documents to MongoDB Atlas.

    Args:
        batch:   List of ready-to-upsert chunk dicts.
        dry_run: When ``True`` log intent but skip the actual write.

    Returns:
        ``(inserted_count, updated_count)`` from the bulk-write result.
    """
    if dry_run:
        logger.info("[MONGO] DRY-RUN — would upsert %d chunk(s).", len(batch))
        return 0, 0

    try:
        result = insert_documents(batch)
        logger.info(
            "[MONGO] Upserted %d chunk(s): %d new, %d updated.",
            len(batch),
            result.get("inserted", 0),
            result.get("updated", 0),
        )
        return result.get("inserted", 0), result.get("updated", 0)
    except Exception as exc:
        logger.error("[MONGO] Batch upsert failed: %s", exc)
        return 0, 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UniBox Hybrid Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Crawl, chunk, and embed — but skip MongoDB writes.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        metavar="TAG",
        default=None,
        help=(
            "Only process seed sources whose 'tags' list contains at least "
            "one of these values.  Omit to process all active sources."
        ),
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        default=False,
        help="Skip crawl4ai HTML sources (useful when testing PDF-only ingestion).",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        default=False,
        help="Skip PDF sources (useful when testing HTML-only ingestion).",
    )
    return parser.parse_args()


async def run_pipeline(
    dry_run: bool = False,
    filter_tags: list[str] | None = None,
    skip_html: bool = False,
    skip_pdf: bool = False,
) -> int:
    """
    Orchestrate the full hybrid ingestion pipeline.

    Args:
        dry_run:      Skip MongoDB writes.
        filter_tags:  Restrict processing to sources matching these tags.
        skip_html:    Skip crawl4ai HTML sources.
        skip_pdf:     Skip PDF sources.

    Returns:
        Exit code — ``0`` on success, ``1`` on fatal error.
    """
    start_time = time.monotonic()
    banner = "=" * 64

    print(banner)
    print("  UNIBOX HYBRID INGESTION PIPELINE")
    if dry_run:
        print("  *** DRY-RUN MODE — MongoDB writes disabled ***")
    if filter_tags:
        print(f"  Filtering by tags: {filter_tags}")
    print(banner)

    # ------------------------------------------------------------------
    # STEP 1 — Load seed registry
    # ------------------------------------------------------------------
    print("\n[1/4] Loading seed registry…")
    try:
        html_sources, pdf_sources = load_seed_registry(
            _SEED_REGISTRY_PATH, filter_tags=filter_tags
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.critical("Cannot load seed registry: %s", exc)
        return 1

    if skip_html:
        logger.info("--skip-html set: skipping %d HTML source(s).", len(html_sources))
        html_sources = []
    if skip_pdf:
        logger.info("--skip-pdf set: skipping %d PDF source(s).", len(pdf_sources))
        pdf_sources = []

    if not html_sources and not pdf_sources:
        logger.warning("No active sources to process. Check seed_sources.json and tag filters.")
        return 0

    # ------------------------------------------------------------------
    # STEP 2 — Collect documents (HTML crawl + PDF extraction)
    # ------------------------------------------------------------------
    print("\n[2/4] Collecting documents…")

    all_documents: list[IngestableDocument] = []

    # HTML crawl (async)
    if html_sources:
        html_docs = await collect_html_documents(html_sources)
        print(f"       [HTML] {len(html_docs)} page(s) crawled.")
        all_documents.extend(html_docs)

    # PDF extraction (synchronous — I/O bound but not worth async complexity)
    if pdf_sources:
        pdf_docs = collect_pdf_documents(pdf_sources)
        print(f"       [PDF]  {len(pdf_docs)} PDF(s) extracted.")
        all_documents.extend(pdf_docs)

    if not all_documents:
        logger.warning("No documents collected — nothing to ingest.")
        return 0

    print(f"\n       Total documents: {len(all_documents)}")

    # ------------------------------------------------------------------
    # STEP 3 — Chunk + embed + upsert
    # ------------------------------------------------------------------
    print("\n[3/4] Chunking, embedding, and upserting…")
    summary = chunk_and_embed(all_documents, dry_run=dry_run)

    # ------------------------------------------------------------------
    # STEP 4 — Summary
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - start_time
    print(f"\n[4/4] INGESTION COMPLETE  ({elapsed:.1f}s)")
    print(banner)
    print(f"  Documents processed : {len(all_documents)}")
    print(f"  Total chunks        : {summary['total_chunks']}")
    print(f"  New (inserted)      : {summary['inserted']}")
    print(f"  Updated             : {summary['updated']}")
    print(f"  Failed documents    : {summary['failed_docs']}")
    if dry_run:
        print("  *** DRY-RUN — no data was written to MongoDB ***")
    print(banner)

    return 0


if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(
        run_pipeline(
            dry_run=args.dry_run,
            filter_tags=args.tags,
            skip_html=args.skip_html,
            skip_pdf=args.skip_pdf,
        )
    )
    sys.exit(exit_code)