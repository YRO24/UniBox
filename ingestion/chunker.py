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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:

    print("[CHUNKER] chunk_text() started.")

    if not text or not text.strip():
        print("[CHUNKER] Empty text received.")
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    print(f"[CHUNKER] Input characters: {len(text)}")
    print(f"[CHUNKER] Chunk size: {chunk_size}")
    print(f"[CHUNKER] Overlap: {overlap}")

    print("[CHUNKER] Tokenizing input text...")

    tokens = _tokenizer.encode(
        text.strip(),
        add_special_tokens=False,
        truncation=False
    )

    print(f"[CHUNKER] Tokenization complete.")
    print(f"[CHUNKER] Total tokens: {len(tokens)}")

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
            chunks.append(chunk)

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