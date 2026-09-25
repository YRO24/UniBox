from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from database.mongo_client import (
    vector_search,
    store_feedback,
    check_database_connection
)
from ingestion.embedder import generate_embeddings


app = FastAPI(
    title="UniBox RAG API",
    description="Backend API for UniBox semantic search",
    version="1.0.0"
)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User search query"
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=20
    )


class FeedbackRequest(BaseModel):
    query: str
    helpful: bool
    result_urls: Optional[List[str]] = None


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "UniBox RAG API is running",
        "status": "ok"
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    mongo_ok = check_database_connection()

    return {
        "api": "ok",
        "mongodb": "connected" if mongo_ok else "disconnected"
    }


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------

@app.post("/search")
def search(request: SearchRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    try:

        # Generate embedding for user's query
        embeddings = generate_embeddings([query])

        if not embeddings:
            raise HTTPException(
                status_code=500,
                detail="Could not generate query embedding."
            )

        query_embedding = embeddings[0]

        # Search MongoDB Atlas Vector Search
        results = vector_search(
            query_embedding=query_embedding,
            limit=request.limit
        )

        return {
            "query": query,
            "count": len(results),
            "results": results
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ---------------------------------------------------------
# FEEDBACK
# ---------------------------------------------------------

@app.post("/feedback")
def feedback(request: FeedbackRequest):

    try:

        feedback_id = store_feedback(
            query=request.query,
            helpful=request.helpful,
            result_urls=request.result_urls
        )

        return {
            "success": True,
            "message": "Feedback recorded successfully.",
            "feedback_id": feedback_id
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not store feedback: {str(e)}"
        )