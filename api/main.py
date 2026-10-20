import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from database.mongo_client import (
    vector_search,
    store_feedback,
    check_database_connection
)
from ingestion.embedder import generate_embeddings
from llm import gemini_service
from llm.gemini_service import LLMError

logger = logging.getLogger("unibox.api")


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

class ChatTurn(BaseModel):
    """One earlier message of the conversation (kept by the frontend session)."""
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User search query"
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=20
    )

    # --- Optional LLM fields (backwards compatible: omit them and /search
    # --- behaves exactly as before and returns retrieval results only). ---
    generate_answer: bool = Field(
        default=False,
        description="When true, also generate a natural-language answer with Gemini."
    )

    history: List[ChatTurn] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier turns of THIS conversation, oldest first."
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
        "mongodb": "connected" if mongo_ok else "disconnected",
        # Only reports whether a key is set; makes no call to Google.
        "gemini": (
            "configured" if gemini_service.is_configured() else "not_configured"
        ),
    }


# ---------------------------------------------------------
# RETRIEVAL  (two-stage: vector recall -> cross-encoder re-rank)
# ---------------------------------------------------------

def retrieve_chunks(query: str, limit: int) -> list:
    """
    Existing retrieval pipeline, unchanged in behaviour:
    embed the query, pull RERANK_CANDIDATE_COUNT candidates from MongoDB Atlas
    Vector Search, re-rank them with the cross-encoder and return the top
    ``limit`` chunks.
    """

    # Generate embedding for the query
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
        return []

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
    return candidates[:limit]


# ---------------------------------------------------------
# SEMANTIC SEARCH + (optional) GEMINI ANSWER
# ---------------------------------------------------------

@app.post("/search")
def search(request: SearchRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    history = [turn.model_dump() for turn in request.history]

    # For follow-ups ("explain that more simply") the raw message is a useless
    # search query, so turn it into a standalone one first. This is a no-op
    # without history, without a key, or if the rewrite call fails.
    retrieval_query = query
    if request.generate_answer and history:
        retrieval_query = gemini_service.rewrite_followup_query(query, history)

    try:
        results = retrieve_chunks(retrieval_query, request.limit)

    except HTTPException:
        raise

    except Exception:
        # Details go to the server log, not to the client.
        logger.exception("Search failed")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Check the backend logs for details."
        )

    response = {
        "query": query,
        "count": len(results),
        "results": results
    }

    if retrieval_query != query:
        response["retrieval_query"] = retrieval_query

    if not request.generate_answer:
        return response

    # --------------------------------------------------
    # Generation: Gemini only ever sees the retrieved chunks.
    # Failures degrade gracefully: retrieval results are still returned.
    # --------------------------------------------------
    try:
        generated = gemini_service.generate_answer(
            question=query,
            chunks=results,
            history=history,
        )
        response["answer"] = generated["answer"]
        response["sources"] = generated["sources"]
        response["model"] = generated["model"]

    except LLMError as exc:
        logger.warning("Answer generation failed: %s", exc.code)
        response["answer"] = None
        response["answer_error"] = {
            "code": exc.code,
            "message": exc.message
        }

    except Exception:
        logger.exception("Unexpected error during answer generation")
        response["answer"] = None
        response["answer_error"] = {
            "code": "unexpected_error",
            "message": "Something went wrong while generating the answer. Please try again."
        }

    return response


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