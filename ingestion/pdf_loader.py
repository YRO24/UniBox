"""
PDF text extraction utility for UniBox hybrid ingestion.

Supports both local file paths and remote HTTPS URLs.  Remote PDFs are
downloaded to a temporary file (auto-deleted on close) so no intermediate
state persists on disk.

Exported API
------------
load_pdf_text(file_path_or_url: str) -> str
    Extract and normalize all text from a PDF source.

Design decisions
----------------
* Uses ``pypdf`` (pure-Python, no native deps) as the primary extractor.
  If a page yields no text (e.g. scanned image), it is silently skipped
  rather than raising, so partially-digital PDFs still produce usable output.
* Whitespace normalization collapses the excessive blank lines that PDF
  layout engines produce, making downstream token-based chunking more
  efficient.
* All ``requests`` timeouts are explicit (connect=10 s, read=60 s) to avoid
  hanging the ingestion pipeline on slow government/university file servers.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Maximum PDF download size (bytes).  Prevents accidentally streaming a
# multi-GB file into a temporary file during ingestion.
_MAX_DOWNLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Requests session reused across calls for connection pooling.
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a lazily-initialised, shared :class:`requests.Session`."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = (
            "UniBox-PDF-Loader/1.0 (educational research crawler)"
        )
    return _session


def _is_url(source: str) -> bool:
    """Return ``True`` when *source* looks like an HTTP(S) URL."""
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _download_pdf(url: str) -> IO[bytes]:
    """
    Stream a remote PDF into an auto-deleted :class:`tempfile.NamedTemporaryFile`.

    Args:
        url: HTTPS URL pointing to a PDF file.

    Returns:
        An open binary file-like object positioned at byte 0.

    Raises:
        requests.HTTPError: On non-2xx responses.
        ValueError: If the response body exceeds :data:`_MAX_DOWNLOAD_BYTES`.
        requests.RequestException: On network / timeout failures.
    """
    logger.info("Downloading PDF: %s", url)
    session = _get_session()

    response = session.get(url, stream=True, timeout=(10, 60))
    response.raise_for_status()

    # Pre-check Content-Type header if present
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            f"URL returned HTML (Content-Type: {content_type!r}), not a binary PDF: {url}"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=True)
    downloaded = 0

    for chunk in response.iter_content(chunk_size=65_536):
        if chunk:
            downloaded += len(chunk)
            if downloaded > _MAX_DOWNLOAD_BYTES:
                tmp.close()
                raise ValueError(
                    f"PDF at {url!r} exceeds the {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB "
                    "safety limit.  Increase _MAX_DOWNLOAD_BYTES if intentional."
                )
            tmp.write(chunk)

    tmp.flush()
    tmp.seek(0)

    # Validate PDF magic header
    header = tmp.read(5)
    tmp.seek(0)
    if not header.startswith(b"%PDF"):
        tmp.close()
        raise ValueError(
            f"Downloaded file from {url} does not start with PDF magic bytes (%PDF); likely an error page."
        )

    logger.info("Downloaded %.1f KB from %s", downloaded / 1024, url)
    return tmp


def _extract_text_from_reader(reader: PdfReader, source_label: str) -> str:
    """
    Iterate over all pages of *reader* and concatenate their text.

    Pages that produce no text (e.g. scanned images without an OCR layer) are
    silently skipped with a debug log entry.

    Args:
        reader:       An initialised :class:`pypdf.PdfReader` instance.
        source_label: Human-readable identifier for log messages (path or URL).

    Returns:
        Concatenated raw text from all readable pages, separated by newlines.
    """
    page_texts: list[str] = []
    n_pages = len(reader.pages)
    skipped = 0

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract text from page %d/%d of %r: %s",
                page_num,
                n_pages,
                source_label,
                exc,
            )
            skipped += 1
            continue

        if page_text.strip():
            page_texts.append(page_text)
        else:
            logger.debug(
                "Page %d/%d of %r yielded no text (possibly scanned).",
                page_num,
                n_pages,
                source_label,
            )
            skipped += 1

    if skipped:
        logger.info(
            "%d/%d page(s) skipped in %r (no extractable text).",
            skipped,
            n_pages,
            source_label,
        )

    return "\n".join(page_texts)


def _normalize_whitespace(text: str) -> str:
    """
    Collapse pathological whitespace patterns produced by PDF layout engines.

    Steps applied (in order):
    1. Strip leading/trailing whitespace from every line.
    2. Collapse runs of more than two consecutive blank lines into a single
       blank line, reducing excessive vertical spacing.
    3. Strip leading/trailing whitespace from the entire string.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text suitable for token-based chunking.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line individually
    lines = [line.strip() for line in text.split("\n")]

    # Collapse runs of >2 blank lines into exactly one blank line
    normalized_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                normalized_lines.append(line)
        else:
            blank_run = 0
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pdf_text(file_path_or_url: str) -> str:
    """
    Extract and normalize all text from a local PDF or a remote PDF URL.

    For **local paths**, the file is opened directly.  The caller is
    responsible for ensuring the path is accessible.

    For **remote URLs** (``http://`` / ``https://``), the PDF is streamed
    into an auto-deleted temporary file before extraction.  The temporary
    file is closed (and deleted) automatically when this function returns,
    regardless of whether extraction succeeds or raises.

    Whitespace normalization is applied unconditionally so that the returned
    string is ready for :func:`ingestion.chunker.chunk_text`.

    Args:
        file_path_or_url:
            Absolute or relative local file path  **or** an HTTPS URL
            pointing to a publicly accessible PDF.

    Returns:
        Normalized text extracted from all readable pages.
        Returns an empty string if the PDF contains no extractable text.

    Raises:
        FileNotFoundError:       If a local path does not exist.
        IsADirectoryError:       If a local path points to a directory.
        requests.HTTPError:      If a remote URL returns a non-2xx status.
        ValueError:              If a remote PDF exceeds the size safety limit.
        requests.RequestException: On network / timeout failures.

    Example::

        # Local PDF
        text = load_pdf_text("data/raw_docs/kjsce_fee_structure_2026.pdf")

        # Remote PDF
        text = load_pdf_text(
            "https://kjsce.somaiya.edu/assets/docs/mandatory_disclosure.pdf"
        )
    """
    source_label = file_path_or_url  # used in log messages

    tmp_file: IO[bytes] | None = None  # only set for remote downloads

    try:
        if _is_url(file_path_or_url):
            # --- Remote PDF ---
            tmp_file = _download_pdf(file_path_or_url)
            reader = PdfReader(tmp_file)
        else:
            # --- Local PDF ---
            local_path = Path(file_path_or_url)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PDF not found at local path: {local_path.resolve()}"
                )
            if local_path.is_dir():
                raise IsADirectoryError(
                    f"Expected a PDF file but got a directory: {local_path.resolve()}"
                )
            logger.info("Reading local PDF: %s", local_path.resolve())
            reader = PdfReader(str(local_path))

        raw_text = _extract_text_from_reader(reader, source_label)
        normalized = _normalize_whitespace(raw_text)

        logger.info(
            "Extracted %d characters from %r (%d pages).",
            len(normalized),
            source_label,
            len(reader.pages),
        )
        return normalized

    except (FileNotFoundError, IsADirectoryError, ValueError):
        # Re-raise domain errors with full context already logged by helpers.
        raise
    except requests.RequestException as exc:
        logger.error("Network error loading PDF %r: %s", source_label, exc)
        raise
    except Exception as exc:
        logger.exception("Unexpected error loading PDF %r: %s", source_label, exc)
        raise
    finally:
        if tmp_file is not None:
            try:
                tmp_file.close()
            except Exception:
                pass  # Best-effort cleanup; ignore close errors.
