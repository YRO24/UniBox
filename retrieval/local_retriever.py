
"""
Local PDF retrieval for UniBox.
Builds an in-memory semantic index without MongoDB.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from ingestion.embedder import generate_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw_docs"

_index: list[dict[str, Any]] = []
_index_ready = False


def build_local_index(force: bool = False) -> int:
    global _index, _index_ready

    if _index_ready and not force:
        return len(_index)

    documents: list[dict[str, str]] = []
    pdf_paths = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {PDF_DIRECTORY}")

    # STEP 1: Extract PDF text
    for pdf_path in pdf_paths:
        try:
            print(f"[PDF] Reading: {pdf_path.name}", flush=True)

            text = load_pdf_text(str(pdf_path))

            print(
                f"[PDF] Extracted {len(text)} characters",
                flush=True
            )

            if text.strip():
                documents.append({
                    "title": pdf_path.stem,
                    "path": str(pdf_path),
                    "text": text
                })
            else:
                print(f"[PDF] No readable text: {pdf_path.name}", flush=True)

        except Exception as exc:
            print(f"[PDF ERROR] {pdf_path.name}: {exc}", flush=True)

    if not documents:
        raise RuntimeError(
            "Could not extract readable text from any local PDF."
        )

    # STEP 2: Chunk and embed documents
    new_index: list[dict[str, Any]] = []

    for document in documents:

        print(
            f"[CHUNK] Processing: {document['title']}",
            flush=True
        )

        chunks = chunk_text(
            document["text"],
            title=document["title"]
        )

        print(
            f"[CHUNK] Generated {len(chunks)} chunks",
            flush=True
        )

        if not chunks:
            continue

        print("[EMBED] Generating embeddings...", flush=True)

        vectors = generate_embeddings(chunks)

        print(
            f"[EMBED] Generated {len(vectors)} embeddings",
            flush=True
        )

        for chunk_number, (chunk, vector) in enumerate(
            zip(chunks, vectors), start=1
        ):
            new_index.append({
                "title": document["title"],
                "source": Path(document["path"]).name,
                "text": chunk,
                "chunk_index": chunk_number,
                "embedding": vector
            })

    if not new_index:
        raise RuntimeError("No chunks were successfully indexed.")

    _index = new_index
    _index_ready = True

    print(
        f"[LOCAL INDEX] Ready: {len(_index)} chunks "
        f"from {len(documents)} PDFs.",
        flush=True
    )

    return len(_index)


def search_local(query: str, limit: int = 5) -> list[dict[str, Any]]:

    if not _index_ready:
        build_local_index()

    print("[SEARCH] Generating query embedding...", flush=True)

    query_vectors = generate_embeddings([query])

    if not query_vectors:
        return []

    matrix = np.asarray(
        [item["embedding"] for item in _index],
        dtype=np.float32
    )

    query_vector = np.asarray(
        query_vectors[0],
        dtype=np.float32
    )

    scores = matrix @ query_vector

    best_indices = np.argsort(scores)[::-1][:limit]

    results = []

    for idx in best_indices:
        item = _index[int(idx)]

        results.append({
            "title": item["title"],
            "source": item["source"],
            "text": item["text"],
            "chunk_index": item["chunk_index"],
            "score": float(scores[int(idx)])
        })

    print(f"[SEARCH] Retrieved {len(results)} chunks.", flush=True)

    return results
