"""
Crawler module for UniBox using crawl4ai.

Provides async web crawling with depth-limited BFS traversal, noise
reduction via content pruning, and rate-limited concurrency. Returns
isolated per-page document dicts ready for downstream token-based chunking.
"""

import asyncio
import logging
from typing import TypedDict
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class CrawledDocument(TypedDict):
    """A single crawled webpage document."""
    url: str
    title: str
    content: str


# ---------------------------------------------------------------------------
# Link normalization helper
# ---------------------------------------------------------------------------

def normalize_url(raw_url: str, base_url: str | None = None) -> str:
    """
    Clean and standardize a URL using stdlib urllib.parse.

    This avoids fragile regex patterns that break on complex paths
    (encoded characters, query strings, ports, etc.).

    Steps:
        1. Strip surrounding whitespace.
        2. Resolve relative URLs against *base_url* when provided.
        3. Drop the fragment component (``#section``).
        4. Normalize the scheme to lowercase.
        5. Remove default ports (80 for http, 443 for https).
        6. Collapse empty paths to ``/``.
        7. Strip trailing slashes on the path (except bare root ``/``).

    Args:
        raw_url:  The URL string to normalize (may be relative).
        base_url: Optional base URL for resolving relative references.

    Returns:
        A cleaned, absolute URL string.
    """
    url = raw_url.strip()
    if not url:
        return ""

    # Resolve relative URLs
    if base_url:
        url = urljoin(base_url, url)

    # Remove fragment
    url, _ = urldefrag(url)

    parsed = urlparse(url)

    # Lowercase scheme
    scheme = parsed.scheme.lower()

    # Strip default ports
    netloc = parsed.hostname or ""
    port = parsed.port
    if port and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"

    # Normalise path: ensure at least "/" and strip redundant trailing slash
    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def clean_markdown_links(text: str) -> str:
    """
    Sanitize markdown link targets without regex.

    Walks the text looking for ``](`` link patterns, extracts each href,
    and normalizes it via :func:`normalize_url`.  Non-link text passes
    through unchanged.

    Args:
        text: Markdown-formatted content string.

    Returns:
        The same markdown with all link URLs normalized.
    """
    if not text:
        return text

    result_parts: list[str] = []
    search_start = 0

    while True:
        # Find the next markdown link opening: ](
        link_marker = text.find("](", search_start)
        if link_marker == -1:
            # No more links — append the rest and stop
            result_parts.append(text[search_start:])
            break

        # Copy everything up to and including "]("
        result_parts.append(text[search_start : link_marker + 2])

        # Find the closing parenthesis (handle nested parens)
        url_start = link_marker + 2
        depth = 1
        pos = url_start
        while pos < len(text) and depth > 0:
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1

        raw_href = text[url_start : pos - 1]  # exclude closing ")"
        normalized = normalize_url(raw_href)
        result_parts.append(normalized)
        result_parts.append(")")

        search_start = pos

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Core crawl function
# ---------------------------------------------------------------------------

_MAX_DEPTH = 2
_MAX_PAGES_PER_URL = 50
_SEMAPHORE_COUNT = 3
_DELAY_BETWEEN_REQUESTS = 1.5  # seconds
_PAGE_TIMEOUT_MS = 30_000

# HTML tags commonly carrying navigation / boilerplate noise
_EXCLUDED_TAGS = [
    "nav", "footer", "header", "aside",
    "form", "script", "style", "noscript",
    "iframe"
]


async def crawl_urls(urls: list[str]) -> list[CrawledDocument]:
    """
    Crawl one or more seed URLs asynchronously with depth-limited BFS.

    For every seed URL the crawler will follow same-domain links up to
    ``depth=2``, strip web noise (menus, footers, sidebars), and return
    a flat list of per-page document dicts.

    Concurrency is bounded by an asyncio semaphore and an inter-request
    delay to avoid overwhelming target servers.

    Args:
        urls: Seed URLs to crawl.

    Returns:
        A list of :class:`CrawledDocument` dicts, one per successfully
        crawled page.  The ``content`` field contains clean body text
        ready for token-based chunking.
    """
    if not urls:
        return []

    # --- Browser configuration (shared across all crawls) ----------------
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    # --- Content-filter pipeline -----------------------------------------
    pruning_filter = PruningContentFilter(
        threshold=0.45,
        threshold_type="fixed",
        min_word_threshold=15,
    )

    md_generator = DefaultMarkdownGenerator(
        content_filter=pruning_filter,
    )

    documents: list[CrawledDocument] = []
    seen_urls: set[str] = set()
    semaphore = asyncio.Semaphore(_SEMAPHORE_COUNT)

    async def _crawl_seed(crawler: AsyncWebCrawler, seed_url: str) -> None:
        """Crawl a single seed URL (with depth) under a semaphore."""
        async with semaphore:
            deep_strategy = BFSDeepCrawlStrategy(
                max_depth=_MAX_DEPTH,
                include_external=False,  # same-domain only
                max_pages=_MAX_PAGES_PER_URL,
            )

            config = CrawlerRunConfig(
                deep_crawl_strategy=deep_strategy,
                markdown_generator=md_generator,
                excluded_tags=_EXCLUDED_TAGS,
                page_timeout=_PAGE_TIMEOUT_MS,
                verbose=False,
            )

            try:
                results = await crawler.arun(seed_url, config=config)

                # arun returns a list when a deep_crawl_strategy is set
                if not isinstance(results, list):
                    results = [results]

                for result in results:
                    if not result.success:
                        logger.warning(
                            "Failed to crawl %s (status=%s)",
                            result.url,
                            result.status_code,
                        )
                        continue

                    normalized = normalize_url(result.url)
                    if normalized in seen_urls:
                        continue
                    seen_urls.add(normalized)

                    # Prefer fit_markdown (noise-filtered) over raw
                    content = ""
                    if result.markdown:
                        content = (
                            result.markdown.fit_markdown
                            or result.markdown.raw_markdown
                            or ""
                        )
                    content = clean_markdown_links(content).strip()
                    # --- Hardcode noise removal ---
                    noise_strings = [
                        "We are happy to answer all your admission related enquiries. Fill out the form and we will be in touch with you shortly.",
                        "We acknowledge the receipt of your enquiry. Our team will get back to you shortly.",
                        "We acknowledge the receipt of your enquiry. Our team will get back to you short..."
                    ]
                    for noise in noise_strings:
                        content = content.replace(noise, "")
                    
                    content = content.strip()
                    # ------------------------------

                    if not content:
                        logger.debug("Empty content for %s — skipping", result.url)
                        continue

                    # Extract title from page metadata or fall back to URL
                    title = ""
                    if result.metadata and isinstance(result.metadata, dict):
                        title = result.metadata.get("title", "")
                    title = title or normalized

                    documents.append(
                        CrawledDocument(
                            url=normalized,
                            title=title,
                            content=content,
                        )
                    )

            except asyncio.TimeoutError:
                logger.error("Timeout while crawling seed URL: %s", seed_url)
            except Exception:
                logger.exception("Unexpected error crawling %s", seed_url)

            # Rate-limit delay between seeds
            await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

    # --- Run all seeds concurrently (bounded by semaphore) ---------------
    async with AsyncWebCrawler(config=browser_config) as crawler:
        tasks = [_crawl_seed(crawler, url) for url in urls]
        await asyncio.gather(*tasks)

    logger.info(
        "Crawling complete — %d documents from %d seed URL(s)",
        len(documents),
        len(urls),
    )
    return documents


# ---------------------------------------------------------------------------
# Quick verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    sample_urls = [
        "https://www.somaiya.edu",
        "https://kjsce.somaiya.edu",
    ]

    async def _main() -> None:
        docs = await crawl_urls(sample_urls)
        print(f"\n{'='*60}")
        print(f"  Crawled {len(docs)} document(s)")
        print(f"{'='*60}\n")
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc['title']}")
            print(f"    URL:     {doc['url']}")
            print(f"    Content: {doc['content'][:200]}...")
            print()

    asyncio.run(_main())
