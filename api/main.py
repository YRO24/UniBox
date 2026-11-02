
"""
UniBox API: Local PDF Retrieval + Gemini.
MongoDB is disabled in this phase.
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.local_retriever import (
    build_local_index,
    search_local
)

from llm.gemini_service import (
    generate_answer,
    MODEL_NAME
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[STARTUP] Building local PDF index...", flush=True)

    try:
        count = build_local_index()

        app.state.index_status = f"ready ({count} chunks)"

        print("[STARTUP] PDF index ready!", flush=True)

    except Exception as exc:

        app.state.index_status = f"error: {exc}"

        print(f"[STARTUP ERROR] {exc}", flush=True)

    yield


app = FastAPI(
    title="UniBox RAG API",
    description="Local PDF retrieval with Gemini",
    version="2.0.0",
    lifespan=lifespan
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)


@app.get("/")
def root():

    return {
        "message": "UniBox RAG API is running",
        "status": "ok",
        "mode": "local-pdf + Gemini",
        "mongodb": "disabled"
    }


@app.get("/health")
def health():

    return {
        "api": "ok",
        "mongodb": "disabled",
        "local_pdf_index": getattr(
            app.state,
            "index_status",
            "not initialized"
        ),
        "gemini_model": MODEL_NAME
    }


@app.post("/search")
def search(request: SearchRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    try:

        results = search_local(query, request.limit)

        return {
            "query": query,
            "count": len(results),
            "results": results
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Local document search failed: {exc}"
        )


@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        print("[CHAT] Received question:", question, flush=True)

        print("[CHAT] Searching local PDFs...", flush=True)

        retrieved = search_local(question, limit=5)

        print(
            f"[CHAT] Retrieved {len(retrieved)} chunks",
            flush=True
        )

        print("[CHAT] Sending context to Gemini...", flush=True)

        answer = generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            history=[
                item.model_dump()
                for item in request.history
            ]
        )

        print("[CHAT] Gemini response received!", flush=True)

        sources = []
        seen = set()

        for item in retrieved:

            if item["source"] not in seen:

                sources.append({
                    "title": item["title"],
                    "source": item["source"],
                    "score": item["score"]
                })

                seen.add(item["source"])

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved)
        }

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not generate answer: {exc}"
        )
