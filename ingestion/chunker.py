"""
Chunker module for UniBox.
"""

from typing import List

from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 30


print("[CHUNKER] Loading tokenizer...")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("[CHUNKER] Tokenizer loaded successfully.")


def chunk_text(
    text: str,
    title: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:
    """
    Split text into overlapping token-based chunks.

    When *title* is provided, each chunk is prefixed with
    ``"Title: {title} | Content: "`` so the embedding model can
    anchor the chunk to its source document topic.

    Args:
        text:       Raw document text to chunk.
        title:      Document title to prepend for context-aware embeddings.
        chunk_size: Maximum tokens per chunk (excluding prefix).
        overlap:    Number of tokens shared between consecutive chunks.

    Returns:
        List of context-enriched text chunks.
    """

    print("[CHUNKER] chunk_text() started.")

    if not text or not text.strip():
        print("[CHUNKER] Empty text received.")
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    print(f"[CHUNKER] Input characters: {len(text)}")
    print(f"[CHUNKER] Chunk size: {chunk_size}")
    print(f"[CHUNKER] Overlap: {overlap}")
    if title:
        print(f"[CHUNKER] Title prefix: {title}")

    print("[CHUNKER] Tokenizing input text...")

    tokens = _tokenizer.encode(
        text.strip(),
        add_special_tokens=False,
        truncation=False
    )

    print(f"[CHUNKER] Tokenization complete.")
    print(f"[CHUNKER] Total tokens: {len(tokens)}")

    # Build the context prefix that will be prepended to every chunk.
    context_prefix = ""
    if title and title.strip():
        context_prefix = f"Title: {title.strip()} | Content: "

    chunks = []

    start = 0
    chunk_number = 1

    while start < len(tokens):

        print(
            f"[CHUNKER] Creating chunk {chunk_number} "
            f"(start={start})..."
        )

        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk = _tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        ).strip()

        if chunk:
            # Prepend the title context so the embedding captures
            # what this chunk is actually about.
            enriched_chunk = f"{context_prefix}{chunk}" if context_prefix else chunk
            chunks.append(enriched_chunk)

            print(
                f"[CHUNKER] Chunk {chunk_number} created "
                f"with {len(chunk_tokens)} tokens."
            )

        # We reached the end of the text.
        if end >= len(tokens):
            break

        # Move forward while keeping the overlap.
        start = end - overlap

        chunk_number += 1

    print(
        f"[CHUNKER] Finished. Generated {len(chunks)} chunks."
    )

    return chunks