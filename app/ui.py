import os
from typing import Any, Optional

import requests
import streamlit as st

# --- Configuration ---
# Pointing to the FastAPI layer. Only the backend ever talks to Gemini;
# no API key is needed (or used) in the frontend.
API_BASE_URL = os.getenv("UNIBOX_API_BASE_URL", "http://localhost:8000")

# Number of document chunks retrieved per question and handed to Gemini.
TOP_K = 6

# Number of most recent messages (user + assistant) sent as conversation
# context for follow-up questions. The backend applies its own limits too.
HISTORY_MESSAGES = 6

# Generating an answer = retrieval + (optional query rewrite) + Gemini call.
REQUEST_TIMEOUT_SECONDS = 120

st.set_page_config(
    page_title="UniBox RAG Search",
    page_icon="🎓",
    layout="centered"
)

# --- Session State Initialization ---
# Streamlit session state is per browser session, so conversations are
# isolated between users and are cleared when the page is refreshed.
# Each message is a dict:
#   user:      {"role": "user", "content": str}
#   assistant: {"role": "assistant", "content": str | None, "query": str,
#               "results": list, "sources": list, "error": str | None,
#               "feedback_given": bool}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --- API Interaction Functions ---
def build_history() -> list[dict[str, str]]:
    """
    Earlier turns of the current conversation, oldest first, for follow-ups.

    Only successfully answered turns are included; failed attempts and the
    message currently being answered are left out.
    """
    history: list[dict[str, str]] = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg.get("content"):
            history.append({"role": "assistant", "content": msg["content"]})
        else:
            # Failed/unanswered turn: drop the user message that preceded it.
            if history and history[-1]["role"] == "user":
                history.pop()
    # The pending user message is the last one; it is sent as `query`.
    if history and history[-1]["role"] == "user":
        history.pop()
    return history[-HISTORY_MESSAGES:]


def perform_search(query: str) -> dict[str, Any]:
    """
    Call POST /search with ``generate_answer`` enabled and return an assistant
    message dict for the chat transcript.

    The backend returns::

        {
            "query": str,
            "count": int,
            "results": [
                {
                    "url": str, "title": str, "text": str, "chunk_index": int,
                    "rerank_score": float,   # cross-encoder logit (may be negative)
                    "vector_score": float,   # cosine similarity in [0, 1]
                }, ...
            ],
            "answer": str | None,            # Gemini answer (None if generation failed)
            "sources": [{"ids": [int], "title": str, "url": str, "cited": bool}],
            "answer_error": {"code": str, "message": str}   # only when answer is None
        }
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "query": query,
        "results": [],
        "sources": [],
        "error": None,
        "feedback_given": False,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "limit": TOP_K,
                "generate_answer": True,
                "history": build_history(),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        message["results"] = data.get("results", [])
        message["sources"] = data.get("sources", [])
        message["content"] = data.get("answer")
        if not message["content"]:
            err = data.get("answer_error") or {}
            message["error"] = err.get(
                "message", "The answer could not be generated."
            )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        message["error"] = (
            f"API returned an error ({e.response.status_code})"
            + (f": {detail}" if detail else ".")
        )
    except requests.exceptions.ConnectionError:
        message["error"] = (
            "Cannot reach the backend. "
            "Ensure FastAPI is running on port 8000 (`uvicorn api.main:app --reload`)."
        )
    except requests.exceptions.Timeout:
        message["error"] = (
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS} s. "
            "The backend may be overloaded."
        )
    except requests.exceptions.RequestException as e:
        message["error"] = f"Unexpected network error: {e}"
    except ValueError:
        message["error"] = "The backend returned an invalid response."

    return message


def submit_feedback(msg_index: int, is_helpful: bool) -> None:
    """
    Post binary feedback to POST /feedback (used as a button callback).

    Ships the original query, the helpfulness verdict and the URLs of the
    retrieved documents for the chosen answer.

    Args:
        msg_index: Index of the assistant message in st.session_state.messages.
        is_helpful: True when the user found the answer useful.
    """
    msg = st.session_state.messages[msg_index]
    result_urls: list[str] = [r.get("url", "") for r in msg.get("results", [])]

    payload = {
        "query": msg.get("query", ""),
        "helpful": is_helpful,          # matches FeedbackRequest.helpful on the API
        "result_urls": result_urls,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        response.raise_for_status()
        msg["feedback_given"] = True
        st.toast("Feedback recorded — thank you for helping us improve!")
    except requests.exceptions.RequestException as e:
        st.toast(f"Failed to submit feedback: {e}", icon="⚠️")


def start_new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None


# --- Rendering helpers ---
def render_sources(sources: list[dict[str, Any]]) -> None:
    """Show the documents the answer was based on (cited ones first)."""
    if not sources:
        return
    cited = [s for s in sources if s.get("cited")]
    shown = cited if cited else sources
    label = "📚 Sources" if cited else "📚 Related documents"
    with st.expander(label, expanded=bool(cited)):
        for s in shown:
            title = s.get("title") or "Untitled Document"
            url = s.get("url") or ""
            ids = ", ".join(f"[{i}]" for i in s.get("ids", []))
            link = f"[{title}]({url})" if url else title
            st.markdown(f"- {ids} {link}")


def render_result_cards(chunks: list[dict[str, Any]]) -> None:
    """The original retrieved-passage cards (title, text, source, scores)."""
    for idx, res in enumerate(chunks, 1):
        title: str = res.get("title", "Untitled Document")
        url: str = res.get("url", "#")
        text: str = res.get("text", "No content available.")
        chunk_idx: int = res.get("chunk_index", 0)

        # --- Score extraction (dual scores with fallback) ---
        rerank_score = res.get("rerank_score", res.get("score", 0.0))
        vector_score = res.get("vector_score", 0.0)

        # Handle negative logits and NoneType gracefully
        try:
            rerank_score = float(rerank_score) if rerank_score is not None else 0.0
        except (ValueError, TypeError):
            rerank_score = 0.0

        try:
            vector_score = float(vector_score) if vector_score is not None else 0.0
        except (ValueError, TypeError):
            vector_score = 0.0

        with st.container():
            st.markdown(f"### [{idx}. {title}]({url})")
            st.markdown(f"> {text}")

            # Metadata footer — three equal columns
            col_src, col_rr, col_vs = st.columns([4, 2, 2])
            with col_src:
                st.caption(f"🔗 Source: {url} | Chunk #{chunk_idx}")
            with col_rr:
                # Use signed format so negative logits render correctly
                st.caption(f"🤖 Re-rank: **{rerank_score:+.4f}**")
            with col_vs:
                st.caption(f"🔢 Vector: **{vector_score:.4f}**")
            st.divider()


def render_assistant_message(msg: dict[str, Any], idx: int, is_latest: bool) -> None:
    if msg.get("content"):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))
    elif msg.get("error"):
        st.error(msg["error"])

    chunks = msg.get("results", [])
    if chunks:
        # When Gemini could not answer, show the raw passages right away so the
        # user still gets the original search experience.
        with st.expander(
            f"🔎 Retrieved passages ({len(chunks)})",
            expanded=not msg.get("content") and is_latest,
        ):
            render_result_cards(chunks)
    elif not msg.get("error"):
        st.info("No relevant documents found. Try rephrasing your query.")

    # --- Feedback Section (latest answer only) ---
    if is_latest and chunks:
        if not msg.get("feedback_given"):
            st.markdown("#### Was this response helpful?")
            fb_col1, fb_col2, _ = st.columns([1, 1, 4])
            with fb_col1:
                st.button("👍 Yes", key=f"fb_yes_{idx}",
                          on_click=submit_feedback, args=(idx, True))
            with fb_col2:
                st.button("👎 No", key=f"fb_no_{idx}",
                          on_click=submit_feedback, args=(idx, False))
        else:
            st.success(
                "Feedback submitted successfully. "
                "This data will be used to train our re-ranking models."
            )


# --- User Interface ---
st.title("🎓 UniBox Search")
st.markdown(
    "Ask questions about the university. Answers are written by Gemini 2.5 Flash "
    "from documents found with MongoDB Atlas Vector Search and all-MiniLM-L6-v2."
)

# The transcript container is created first so that new messages and the
# loading spinner appear above the input form.
chat_area = st.container()

is_busy = st.session_state.pending_query is not None

# Search Bar. A form means Enter/Search submit exactly once (no re-search on
# every rerun), and the button is disabled while an answer is being generated
# to prevent duplicate submissions.
with st.form("search_form", clear_on_submit=True):
    user_query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What is the annual fee for first year engineering?",
        disabled=is_busy,
    )
    submitted = st.form_submit_button("Search", type="primary", disabled=is_busy)

if st.session_state.messages and not is_busy:
    st.button("🗑️ New conversation", on_click=start_new_conversation)

if submitted and user_query.strip() and not is_busy:
    st.session_state.messages.append({"role": "user", "content": user_query.strip()})
    st.session_state.pending_query = user_query.strip()
    st.rerun()

# --- Transcript ---
last_assistant_idx: Optional[int] = None
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "assistant":
        last_assistant_idx = i

with chat_area:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant_message(m, i, is_latest=(i == last_assistant_idx))

    # Answer the pending question (loading indicator shown inside the chat).
    if st.session_state.pending_query is not None:
        with st.chat_message("assistant"):
            with st.spinner("Searching university records and writing an answer..."):
                reply = perform_search(st.session_state.pending_query)
        st.session_state.messages.append(reply)
        st.session_state.pending_query = None
        st.rerun()
