
"""Gemini answer generation for UniBox."""

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.5-flash-lite"
)
MODEL_NAME = PRIMARY_MODEL


def _get_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    return genai.Client(api_key=api_key)


def _is_temporary_service_error(exc: Exception) -> bool:

    status = (
        getattr(exc, "code", None)
        or getattr(exc, "status_code", None)
        or getattr(
            getattr(exc, "response", None),
            "status_code",
            None
        )
    )

    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None

    message = str(exc).lower()

    return status in {
        408, 429, 500, 502, 503, 504
    } or any(
        marker in message
        for marker in (
            "503 unavailable",
            "service unavailable",
            "temporarily overloaded",
            "high demand"
        )
    )


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None
) -> str:

    if not retrieved_chunks:

        return (
            "I couldn't find relevant information about that "
            "in the university documents currently available to me."
        )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"[Source {number}: {chunk['source']}, "
            f"chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    conversation = ""

    if history:

        recent_history = history[-6:]

        conversation = "\n".join(
            f"{'User' if item.get('role') == 'user' else 'Assistant'}: "
            f"{item.get('content', '')}"
            for item in recent_history
            if item.get("role") in {"user", "assistant"}
        )

    prompt = f"""
Conversation so far:
{conversation if conversation else "(No previous conversation)"}

University document context:
{context}

Current user question:
{question}

Write a direct, helpful answer in simple, natural language.

Use only facts supported by the university documents.

Do not invent fees, dates, eligibility rules, names or policies.

If the context does not contain the answer, clearly say so.

Explain technical or administrative terms briefly.

Use short paragraphs or bullets when useful.

Mention source document filenames when useful.
"""

    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=0.25,
        max_output_tokens=900,
        system_instruction=(
            "You are UniBox, a helpful university information assistant. "
            "Answer students politely in clear, human-understandable "
            "language. Be accurate and grounded in the supplied documents."
        )
    )

    models_to_try = [PRIMARY_MODEL]

    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None

    for model_name in models_to_try:

        try:

            print(
                f"[GEMINI] Calling model: {model_name}",
                flush=True
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )

            print(
                "[GEMINI] Generation completed!",
                flush=True
            )

            answer = response.text

            if not answer:
                raise RuntimeError(
                    f"Gemini model {model_name} returned an empty response."
                )

            print(
                f"[GEMINI] Answer generated using {model_name}",
                flush=True
            )

            return answer.strip()

        except Exception as exc:

            last_error = exc

            if (
                model_name != models_to_try[-1]
                and _is_temporary_service_error(exc)
            ):

                print(
                    f"[GEMINI] {model_name} unavailable. "
                    f"Trying fallback: {FALLBACK_MODEL}",
                    flush=True
                )

                continue

            raise RuntimeError(
                f"Gemini request failed using {model_name}: {exc}"
            ) from exc

    raise RuntimeError(
        f"Gemini request failed: {last_error}"
    )
