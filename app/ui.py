import streamlit as st
import requests
from datetime import datetime

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
def perform_search(query: str):
    """Hits Saniya's POST /search endpoint and caches the response."""
    with st.spinner("Searching university records..."):
        try:
            response = requests.post(f"{API_BASE_URL}/search", json={"query": query})
            response.raise_for_status()
            
            # Assuming the backend returns a JSON list of chunk dictionaries
            st.session_state.search_results = response.json()
            st.session_state.current_query = query
            st.session_state.feedback_given = False
            
        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection failed. Ensure FastAPI is running on port 8000. Error: {e}")

def submit_feedback(is_helpful: bool):
    """Hits Saniya's POST /feedback endpoint with user analytics."""
    if not st.session_state.search_results:
        return
        
    try:
        # Extract the URLs from the cached results
        result_urls = [res.get("url", "") for res in st.session_state.search_results]
        
        payload = {
            "query": st.session_state.current_query,
            "is_helpful": is_helpful,
            "result_urls": result_urls,
            "timestamp": datetime.now().isoformat()
        }
        
        requests.post(f"{API_BASE_URL}/feedback", json=payload)
        st.session_state.feedback_given = True
        st.toast("Feedback recorded. Thanks for helping us improve!")
        
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
            # 3. Use the exact keys from Saniya's JSON
            title = res.get("title", "Untitled Document")
            url = res.get("url", "#")
            text = res.get("text", "No content available.")
            score = res.get("score", 0.0)          # Fixed key name
            chunk_idx = res.get("chunk_index", 0)
            
            with st.container():
                st.markdown(f"### [{idx}. {title}]({url})")
                st.markdown(f"> {text}")
                
                # Metadata footer
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"🔗 Source: {url} | Chunk: {chunk_idx}")
                with col2:
                    st.caption(f"🎯 Relevance: **{score:.4f}**")
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