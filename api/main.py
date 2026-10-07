from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

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
# CROSS-ENCODER RE-RANKER  (loaded once at startup)
# ---------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"[API] Loading cross-encoder model: {CROSS_ENCODER_MODEL} ...")
_reranker = CrossEncoder(CROSS_ENCODER_MODEL)
print("[API] Cross-encoder loaded successfully.")

# Number of candidates to retrieve from vector search before re-ranking.
RERANK_CANDIDATE_COUNT = 30


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
# SEMANTIC SEARCH  (two-stage: vector recall → cross-encoder re-rank)
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

        # --------------------------------------------------
        # Stage 1: Broad vector recall (top 30 candidates)
        # --------------------------------------------------
        candidates = vector_search(
            query_embedding=query_embedding,
            limit=RERANK_CANDIDATE_COUNT
        )

        if not candidates:
            return {
                "query": query,
                "count": 0,
                "results": []
            }

        # --------------------------------------------------
        # Stage 2: Cross-encoder re-ranking
        # --------------------------------------------------
        # Build (query, document_text) pairs for the cross-encoder.
        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        # The cross-encoder returns a relevance score for each pair.
        ce_scores = _reranker.predict(pairs).tolist()

        # Attach the cross-encoder score to each candidate.
        for candidate, ce_score in zip(candidates, ce_scores):
            candidate["vector_score"] = candidate.pop("score", 0.0)
            candidate["rerank_score"] = float(ce_score)

        # Sort by cross-encoder score (higher = more relevant).
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        # Return only the top `limit` results.
        results = candidates[: request.limit]

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