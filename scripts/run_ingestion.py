import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ingestion.crawler import crawl_urls
from ingestion.chunker import chunk_text
from ingestion.embedder import generate_embeddings
from database.mongo_client import insert_documents


SOURCE_URLS = [
    "https://www.somaiya.edu",
    "https://kjsce.somaiya.edu",
]


async def run_pipeline():

    print("=" * 60)
    print("UNIBOX INGESTION PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: CRAWLING
    # ---------------------------------------------------------
    print("\n[1/4] Crawling websites...")

    documents = await crawl_urls(SOURCE_URLS)

    print(f"[CRAWLER] Retrieved {len(documents)} pages.")

    if not documents:
        print("[ERROR] No documents were crawled.")
        return

    # ---------------------------------------------------------
    # STEP 2: CHUNKING + EMBEDDING
    # ---------------------------------------------------------
    print("\n[2/4] Chunking + embedding documents...")

    mongo_documents = []
    total_chunks = 0

    for document in documents:

        url = document["url"]
        title = document["title"]
        content = document["content"]

        chunks = chunk_text(content)

        if not chunks:
            continue

        print(
            f"[CHUNKER] {title}: "
            f"{len(chunks)} chunks"
        )

        embeddings = generate_embeddings(chunks)

        if len(embeddings) != len(chunks):
            print(
                f"[WARNING] Embedding count mismatch "
                f"for {url}"
            )
            continue

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            mongo_documents.append({
                "url": url,
                "title": title,
                "text": chunk,
                "chunk_index": index,
                "embedding": embedding
            })

            total_chunks += 1

    print(
        f"\n[EMBEDDER] Generated "
        f"{total_chunks} embeddings."
    )

    # ---------------------------------------------------------
    # STEP 3: STORE IN MONGODB
    # ---------------------------------------------------------
    print("\n[3/4] Storing vectors in MongoDB...")

    batch_size = 100

    total_inserted = 0
    total_updated = 0

    for start in range(
        0,
        len(mongo_documents),
        batch_size
    ):

        batch = mongo_documents[
            start:start + batch_size
        ]

        result = insert_documents(batch)

        total_inserted += result["inserted"]
        total_updated += result["updated"]

        print(
            f"[MONGODB] Batch "
            f"{start // batch_size + 1}: "
            f"{result}"
        )

    # ---------------------------------------------------------
    # STEP 4: COMPLETE
    # ---------------------------------------------------------
    print("\n[4/4] INGESTION COMPLETE")

    print(f"New documents: {total_inserted}")
    print(f"Updated documents: {total_updated}")
    print(f"Total chunks processed: {total_chunks}")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())