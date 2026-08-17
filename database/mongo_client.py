import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "unibox_rag")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "documents")
VECTOR_INDEX_NAME = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")

# Initialize MongoDB Client
if not MONGODB_URI:
    print("Warning: MONGODB_URI is not set in the environment or .env file.")

client: Optional[MongoClient] = MongoClient(MONGODB_URI) if MONGODB_URI else None
db: Optional[Database] = client[DB_NAME] if client is not None else None
collection: Optional[Collection] = db[COLLECTION_NAME] if db is not None else None


def get_db_client() -> MongoClient:
    """Return the MongoDB client instance."""
    global client
    if client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is not set in environment or .env file.")
        client = MongoClient(uri)
    return client


def get_collection(collection_name: str = COLLECTION_NAME) -> Collection:
    """Return a collection from the database."""
    global db
    if db is None:
        db_client = get_db_client()
        db = db_client[DB_NAME]
    return db[collection_name]


def insert_documents(docs: List[Dict[str, Any]]) -> Any:
    """
    Insert documents into the MongoDB collection.
    
    Args:
        docs: List of document dictionaries to insert.
    """
    # TODO: Implement document insertion logic
    pass


def vector_search(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a vector search using MongoDB Atlas Vector Search.
    
    Args:
        query_embedding: Dense vector representation of the search query.
        limit: Maximum number of search results to return.
        
    Returns:
        List of matching document dictionaries.
    """
    # TODO: Implement vector search aggregation pipeline
    pass
