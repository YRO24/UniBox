import streamlit as st
import requests
from datetime import datetime
from typing import Any

# --- Configuration ---
# Pointing to Saniya's FastAPI layer
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="UniBox RAG Search",
    page_icon="🎓",
    layout="centered"
)

# --- Session State Initialization ---
# This prevents Streamlit from wiping results when feedback buttons are clicked
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# --- API Interaction Functions ---
def perform_search(query: str) -> None:
    """
    Call the POST /search endpoint and cache the full response envelope.

    The backend returns::

        {
            "query": str,
            "count": int,
            "results": [
                {
                    "url": str,
                    "title": str,
                    "text": str,
                    "chunk_index": int,
                    "rerank_score": float,   # cross-encoder logit (may be negative)
                    "vector_score": float,   # cosine similarity in [0, 1]
                },
                ...
            ]
        }

    Args:
        query: The user's raw search string.
    """
    with st.spinner("Searching university records..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/search",
                json={"query": query},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.search_results = response.json()
            st.session_state.current_query = query
            st.session_state.feedback_given = False
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error ({e.response.status_code}): {e}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. "
                "Ensure FastAPI is running on port 8000 (`uvicorn api.main:app --reload`)."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out after 30 s. The backend may be overloaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"Unexpected network error: {e}")

def submit_feedback(is_helpful: bool) -> None:
    """
    Post binary feedback to POST /feedback.

    Extracts the result URLs from the cached search response and ships them
    alongside the original query and the helpfulness verdict.

    Args:
        is_helpful: True when the user found the results useful.
    """
    data: dict[str, Any] | None = st.session_state.search_results
    if not data:
        return

    actual_chunks: list[dict[str, Any]] = data.get("results", [])
    result_urls: list[str] = [res.get("url", "") for res in actual_chunks]

    payload = {
        "query": st.session_state.current_query,
        "helpful": is_helpful,          # matches FeedbackRequest.helpful on the API
        "result_urls": result_urls,
    }

    try:
        requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        st.session_state.feedback_given = True
        st.toast("Feedback recorded — thank you for helping us improve!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to submit feedback: {e}")

# --- User Interface ---
st.title("🎓 UniBox Search")
st.markdown("Semantic search powered by MongoDB Atlas Vector Search and all-MiniLM-L6-v2.")

# Search Bar
user_query = st.text_input("What would you like to know?", placeholder="e.g., What is the annual fee for first year engineering?")

if st.button("Search", type="primary") or user_query:
    if user_query.strip():
        perform_search(user_query.strip())

# --- Results Display ---
if st.session_state.search_results is not None:
    # 1. Get the top-level dictionary
    data = st.session_state.search_results
    
    # 2. Extract the actual list of chunks using the "results" key
    actual_chunks = data.get("results", [])
    
    if not actual_chunks:
        st.info("No relevant documents found. Try rephrasing your query.")
    else:
        st.subheader("Results")
        
        for idx, res in enumerate(actual_chunks, 1):
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
                
        # --- Feedback Section ---
        if not st.session_state.feedback_given:
            st.markdown("#### Was this response helpful?")
            fb_col1, fb_col2, _ = st.columns([1, 1, 4])
            
            with fb_col1:
                if st.button("👍 Yes"):
                    submit_feedback(is_helpful=True)
            with fb_col2:
                if st.button("👎 No"):
                    submit_feedback(is_helpful=False)
        else:
            st.success("Feedback submitted successfully. This data will be used to train our re-ranking models.")