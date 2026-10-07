import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "unibox_rag")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "documents")
VECTOR_INDEX_NAME = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")


# ---------------------------------------------------------
# MONGODB CONNECTION
# ---------------------------------------------------------

if not MONGODB_URI:
    print("Warning: MONGODB_URI is not set in the .env file.")

client: Optional[MongoClient] = None
db: Optional[Database] = None
collection: Optional[Collection] = None


def get_db_client() -> MongoClient:
    """Return the MongoDB client instance."""

    global client

    if client is None:
        uri = os.getenv("MONGODB_URI")

        if not uri:
            raise ValueError(
                "MONGODB_URI is not set in environment or .env file."
            )

        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=10000
        )

        # Test the connection
        client.admin.command("ping")

    return client


def get_collection(
    collection_name: str = COLLECTION_NAME
) -> Collection:
    """Return a collection from the database."""

    global db, collection

    if db is None:
        db_client = get_db_client()
        db = db_client[DB_NAME]

    if collection is None or collection.name != collection_name:
        collection = db[collection_name]

    return collection


# ---------------------------------------------------------
# INSERT / UPDATE DOCUMENTS
# ---------------------------------------------------------

def insert_documents(
    docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Insert or update documents in MongoDB.

    Each chunk is uniquely identified using:
        url + chunk_index
    """

    if not docs:
        return {
            "inserted": 0,
            "updated": 0
        }

    col = get_collection()

    operations = []

    for doc in docs:

        if not doc.get("url"):
            continue

        if "chunk_index" not in doc:
            continue

        operations.append(
            UpdateOne(
                {
                    "url": doc["url"],
                    "chunk_index": doc["chunk_index"]
                },
                {
                    "$set": {
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "embedding": doc.get("embedding", []),
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
        )

    if not operations:
        return {
            "inserted": 0,
            "updated": 0
        }

    result = col.bulk_write(
        operations,
        ordered=False
    )

    return {
        "inserted": result.upserted_count,
        "updated": result.modified_count
    }


# ---------------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------------

def vector_search(
    query_embedding: List[float],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform vector search using MongoDB Atlas Vector Search.
    """

    if not query_embedding:
        return []

    col = get_collection()

    # Keep result limit reasonable
    limit = max(1, min(limit, 20))

    # Number of candidates considered during vector search
    num_candidates = max(limit * 10, 50)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 0,
                "url": 1,
                "title": 1,
                "text": 1,
                "chunk_index": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = list(
        col.aggregate(pipeline)
    )

    return results


# ---------------------------------------------------------
# FEEDBACK
# ---------------------------------------------------------

def store_feedback(
    query: str,
    helpful: bool,
    result_urls: Optional[List[str]] = None
) -> str:
    """
    Store user feedback for a search result.
    """

    feedback_collection = get_collection("feedback")

    document = {
        "query": query,
        "helpful": helpful,
        "result_urls": result_urls or [],
        "created_at": datetime.now(timezone.utc)
    }

    result = feedback_collection.insert_one(document)

    return str(result.inserted_id)


# ---------------------------------------------------------
# DATABASE HEALTH CHECK
# ---------------------------------------------------------

def check_database_connection() -> bool:
    """Check whether MongoDB is reachable."""

    try:
        db_client = get_db_client()
        db_client.admin.command("ping")
        return True

    except Exception:
        return False