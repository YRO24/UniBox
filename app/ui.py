import streamlit as st
import requests
from datetime import datetime
from typing import Any

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="UniBox - Somaiya Vidyavihar University",
    page_icon="🏛️",
    layout="wide"
)

# --- Somaiya Custom Styling (Somaiya Red & Charcoal Branding) ---
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

/* Eliminate empty top and side whitespace from Streamlit & force clean light background */
header[data-testid="stHeader"] {
    display: none !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #23282D !important;
}
.block-container {
    padding-top: 0.25rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 96% !important;
    background-color: transparent !important;
}

:root {
    --somaiya-red: #A6192E;
    --somaiya-red-dark: #821021;
    --somaiya-red-light: #FDF2F4;
    --somaiya-gold: #c29953;
    --charcoal: #23282D;
    --dark-slate: #343b42;
    --light-bg: #F8F9FA;
    --border-color: #E2E8F0;
    --text-muted: #5F6B7A;
}

/* Compact, clean top header */
.top-header-strip {
    background-color: #ffffff;
    border-bottom: 2px solid #F0F0F0;
    padding: 6px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    border-radius: 8px;
}

.somaiya-branding {
    display: flex;
    align-items: center;
    gap: 16px;
}

.somaiya-crest-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--somaiya-red), var(--somaiya-red-dark));
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 20px;
    box-shadow: 0 4px 10px rgba(166, 25, 46, 0.3);
}

.somaiya-titles {
    display: flex;
    flex-direction: column;
}

.somaiya-main-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--somaiya-red);
    letter-spacing: 0.5px;
    line-height: 1.2;
}

.somaiya-sub-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #555555;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.top-actions {
    display: flex;
    align-items: center;
    gap: 14px;
}

.sign-in-btn {
    border: 1px solid #c0c0c0;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #333;
    text-decoration: none;
    transition: all 0.2s ease;
}
.sign-in-btn:hover {
    border-color: var(--somaiya-red);
    color: var(--somaiya-red);
}

.somaiya-trust-badge {
    background-color: #0b2545;
    color: white;
    padding: 6px 14px;
    border-radius: 4px;
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-block;
}

.unibox-hero {
    background: linear-gradient(135deg, #ffffff 0%, #FAF0F2 100%);
    border-left: 6px solid var(--somaiya-red);
    padding: 10px 24px;
    border-radius: 8px;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 4px 18px rgba(166, 25, 46, 0.06);
}

.unibox-hero h1 {
    font-family: 'Playfair Display', serif;
    color: var(--somaiya-red);
    margin-bottom: 2px;
    font-size: 1.85rem;
    font-weight: 700;
}

.unibox-hero p {
    color: var(--dark-slate);
    font-size: 0.95rem;
    margin: 0;
}

.somaiya-feature-strip {
    background: #2b3036;
    color: white;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.75rem !important;
    display: flex;
    align-items: stretch;
}

.feature-strip-label {
    background: #394047;
    padding: 10px 18px;
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    display: flex;
    align-items: center;
    border-right: 4px solid var(--somaiya-red);
    min-width: 180px;
}

.feature-strip-items {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    padding: 8px 18px;
    background: #ffffff;
    color: #222;
    flex-grow: 1;
}

.feature-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    background: #F8F9FA;
    border: 1px solid #ECECEC;
    font-size: 0.82rem;
    font-weight: 500;
}
.feature-chip strong {
    color: var(--somaiya-red);
}

.side-ribbon-container {
    position: fixed;
    right: 0;
    top: 35%;
    display: flex;
    flex-direction: column;
    z-index: 9999;
    box-shadow: -2px 0 10px rgba(0,0,0,0.15);
}

.side-ribbon {
    writing-mode: vertical-rl;
    text-orientation: mixed;
    padding: 14px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 0.2s ease, filter 0.2s ease;
}
.side-ribbon:hover {
    filter: brightness(1.15);
    transform: translateX(-4px);
}

.ribbon-apply { background-color: #A6192E; }
.ribbon-enquire { background-color: #8C5835; }
.ribbon-chat { background-color: #1765A4; }

.result-card {
    background: #ffffff;
    border: 1px solid #E8ECEF;
    border-left: 4px solid var(--somaiya-red);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 1.25rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}

.result-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--somaiya-red) !important;
    text-decoration: none;
}
.result-title:hover {
    text-decoration: underline;
    color: var(--somaiya-red-dark) !important;
}

.result-quote {
    border-left: 3px solid #E2E8F0;
    padding-left: 14px;
    margin: 10px 0;
    color: #4A5568;
    font-size: 0.95rem;
    line-height: 1.5;
}

.somaiya-footer {
    margin-top: 4rem;
    padding-top: 2.5rem;
    border-top: 4px solid var(--somaiya-red);
    background: #ffffff;
}

.explore-somaiya-header {
    font-family: 'Playfair Display', serif;
    color: var(--somaiya-red);
    font-size: 1.7rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
}

.explore-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 3rem;
}

.explore-tile {
    background: #40454B;
    color: #ffffff !important;
    padding: 16px 20px;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 0.95rem;
    text-decoration: none !important;
    transition: background-color 0.2s ease;
}
.explore-tile:hover {
    background: var(--somaiya-red) !important;
    color: #ffffff !important;
}

.footer-columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) 160px;
    gap: 30px;
    padding-bottom: 2.5rem;
    border-bottom: 1px solid #ECECEC;
}

.footer-col h4 {
    font-size: 0.82rem;
    font-weight: 800;
    color: #212529;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
}

.footer-col ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.footer-col ul li {
    margin-bottom: 8px;
}

.footer-col ul li a {
    color: #6C757D;
    text-decoration: none;
    font-size: 0.85rem;
    transition: color 0.15s ease;
}
.footer-col ul li a:hover {
    color: var(--somaiya-red);
}

.action-btn-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.footer-action-button {
    display: block;
    border: 1px solid #c9c9c9;
    padding: 8px 12px;
    border-radius: 4px;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    color: #333 !important;
    text-decoration: none !important;
    transition: all 0.2s;
}
.footer-action-button:hover {
    border-color: var(--somaiya-red);
    background-color: var(--somaiya-red);
    color: #ffffff !important;
}

.footer-bottom-bar {
    padding: 24px 0 10px 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #718096;
    font-size: 0.8rem;
}

.social-links {
    display: flex;
    gap: 18px;
    font-size: 1.1rem;
    color: #4A5568;
}

.bottom-disclaimers {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    justify-content: center;
}
.bottom-disclaimers a {
    color: #718096;
    text-decoration: none;
}
.bottom-disclaimers a:hover {
    color: var(--somaiya-red);
}

div.stButton > button:first-child {
    background: linear-gradient(135deg, #A6192E 0%, #821021 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    height: 3.8rem !important;
    border-radius: 14px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 6px 20px rgba(166, 25, 46, 0.35) !important;
    transition: all 0.25s ease !important;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #be1b34 0%, #991227 100%) !important;
    box-shadow: 0 8px 25px rgba(166, 25, 46, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* Centered High-Impact Search Arena with Gray/Maroon Fade to White */
.search-arena-backdrop {
    background: radial-gradient(circle at 50% 30%, rgba(166, 25, 46, 0.08) 0%, rgba(52, 59, 66, 0.05) 50%, rgba(255, 255, 255, 0) 80%);
    padding: 8px 0 4px 0;
    margin: 0.5rem auto 0.25rem auto;
    width: 100%;
}

.search-spotlight-wrapper {
    background: linear-gradient(180deg, #ffffff 0%, #FFFDFE 100%);
    border: 2px solid #ECCCD3;
    box-shadow: 0 16px 36px rgba(166, 25, 46, 0.10), 0 4px 14px rgba(43, 48, 54, 0.06);
    border-radius: 16px;
    padding: 16px 24px 14px 24px;
    margin: 0 auto;
    width: 100%;
    position: relative;
    overflow: hidden;
}

.search-spotlight-wrapper::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #343b42 0%, #A6192E 30%, #D8314A 70%, #343b42 100%);
}

.search-spotlight-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.search-spotlight-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--charcoal);
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: 0.2px;
}

.search-spotlight-tag {
    background: #FDF2F4;
    color: var(--somaiya-red);
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid #F5CCD3;
}

/* Extra large, high-impact input centered in the screen */
div[data-testid="stTextInputRootElement"] input,
div[data-baseweb="input"] input,
.stTextInput input {
    height: 3.8rem !important;
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    border: 2.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding-left: 20px !important;
    background-color: #ffffff !important;
    color: #1A202C !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInputRootElement"] input:focus,
div[data-baseweb="input"] input:focus,
.stTextInput input:focus {
    border-color: var(--somaiya-red) !important;
    box-shadow: 0 0 0 5px rgba(166, 25, 46, 0.2) !important;
}

div[data-testid="stTextInputRootElement"] input::placeholder,
div[data-baseweb="input"] input::placeholder,
.stTextInput input::placeholder {
    color: #8C98A4 !important;
    font-size: 1.02rem !important;
}

.popular-query-pill {
    background: #ffffff;
    border: 1px solid #E2E8F0;
    color: #4A5568;
    padding: 6px 14px;
    border-radius: 18px;
    font-size: 0.84rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 5px rgba(0,0,0,0.03);
}
.popular-query-pill:hover {
    border-color: var(--somaiya-red);
    color: var(--somaiya-red);
    background: #FDF2F4;
    transform: translateY(-1px);
}
</style>""", unsafe_allow_html=True)

# --- Somaiya Side Ribbons (Pic 1 & 2) ---
st.markdown("""<div class="side-ribbon-container"><div class="side-ribbon ribbon-apply">Apply Now</div><div class="side-ribbon ribbon-enquire">Enquire Now</div><div class="side-ribbon ribbon-chat">Chat with a Student</div></div>""", unsafe_allow_html=True)

# --- Top Navigation Header (Pic 3) ---
st.markdown("""<div class="top-header-strip"><div class="somaiya-branding"><div class="somaiya-crest-icon">🪔</div><div class="somaiya-titles"><span class="somaiya-main-title">SOMAIYA</span><span class="somaiya-sub-title">VIDYAVIHAR UNIVERSITY</span></div></div><div class="top-actions"><a href="#signin" class="sign-in-btn">SIGN IN</a><div class="somaiya-trust-badge">Somaiya TRUST</div></div></div>""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# --- Mock Search Fallback for Instant Usability ---
def get_sample_somaiya_results(query: str):
    """Provides instant sample Somaiya data if backend FastAPI is connecting or cold."""
    return {
        "results": [
            {
                "title": "K. J. Somaiya College of Engineering - First Year Admissions & Fee Structure",
                "url": "https://kjsce.somaiya.edu/en/admission/btech",
                "text": "The annual tuition fee structure for First Year B.Tech Engineering at K. J. Somaiya College of Engineering for 2024-25 is approximately INR 4,75,000 to INR 5,20,000 depending on specializations. Scholarships are provided through Somaiya Vidyavihar Trust based on merit and financial need.",
                "score": 0.9412,
                "chunk_index": 2
            },
            {
                "title": "Somaiya Vidyavihar University - Academic Programs & Course Roster",
                "url": "https://somaiya.edu/en/academics/programmes",
                "text": "Somaiya Vidyavihar University offers over 130+ undergraduate, postgraduate, and doctoral degree programs across Engineering, Management, Arts, Science, and Humanities with interdisciplinary credit transfer options.",
                "score": 0.8931,
                "chunk_index": 5
            },
            {
                "title": "riidl (Research Innovation Incubation Design Laboratory) Somaiya Incubator",
                "url": "https://riidl.org",
                "text": "Supported by the Department of Science and Technology (DST), Govt. of India, riidl incubation centre at Somaiya provides seed funding, patent mentoring, prototyping lab access, and co-working space to student startups.",
                "score": 0.8420,
                "chunk_index": 1
            }
        ]
    }

# --- API Interaction Functions ---
<<<<<<< HEAD
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
=======
def perform_search(query: str):
    """Hits FastAPI /search endpoint with graceful fallback."""
    with st.spinner("Searching Somaiya university knowledge base..."):
        try:
            response = requests.post(f"{API_BASE_URL}/search", json={"query": query}, timeout=3)
            response.raise_for_status()
            st.session_state.search_results = response.json()
        except Exception:
            # Fallback gracefully to demo Somaiya results so the app is immediately testable
            st.session_state.search_results = get_sample_somaiya_results(query)
            st.toast("Showing Somaiya Knowledge Base results (FastAPI offline/mock mode)")
            
        st.session_state.current_query = query
        st.session_state.feedback_given = False

def submit_feedback(is_helpful: bool):
    """Submits user analytics feedback."""
    if not st.session_state.search_results:
>>>>>>> 01828db (feat(ui): implement Somaiya University themed frontend styling and layout)
        return

    actual_chunks: list[dict[str, Any]] = data.get("results", [])
    result_urls: list[str] = [res.get("url", "") for res in actual_chunks]

    payload = {
        "query": st.session_state.current_query,
        "helpful": is_helpful,          # matches FeedbackRequest.helpful on the API
        "result_urls": result_urls,
    }

    try:
<<<<<<< HEAD
        requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        st.session_state.feedback_given = True
        st.toast("Feedback recorded — thank you for helping us improve!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to submit feedback: {e}")
=======
        result_urls = [res.get("url", "") for res in st.session_state.search_results.get("results", [])]
        payload = {
            "query": st.session_state.current_query,
            "is_helpful": is_helpful,
            "result_urls": result_urls,
            "timestamp": datetime.now().isoformat()
        }
        requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=2)
    except Exception:
        pass
    st.session_state.feedback_given = True
    st.toast("Feedback recorded. Thank you!")
>>>>>>> 01828db (feat(ui): implement Somaiya University themed frontend styling and layout)

# --- Hero Section ---
st.markdown("""<div class="unibox-hero"><h1>UniBox Intelligent Search</h1><p>Discover courses, admission guidelines, campus services, faculty directories, and student resources across Somaiya Vidyavihar University.</p></div>""", unsafe_allow_html=True)

# --- Somaiya Accents Strip (Pic 2: Affiliation & Accreditations) ---
st.markdown("""<div class="somaiya-feature-strip"><div class="feature-strip-label">Affiliation &<br/>Accreditation</div><div class="feature-strip-items"><div class="feature-chip"><strong>NIRF</strong> Rank within top 200 in Engineering (2018-2024 band)</div><div class="feature-chip"><strong>UGC Approved</strong> Self-Financed Private University from Aug 2019</div><div class="feature-chip"><strong>NAAC 'A' Grade</strong> Accredited Campus</div></div></div>""", unsafe_allow_html=True)

# --- Centered Accentuated Search Arena with Gray/Maroon Fade ---
st.markdown("""<div class="search-arena-backdrop"><div class="search-spotlight-wrapper"><div class="search-spotlight-header"><div class="search-spotlight-title"><span style="font-size: 1.35rem;">🔍</span> Search Somaiya University Knowledge Hub</div><span class="search-spotlight-tag">⚡ AI SEMANTIC SEARCH</span></div></div></div>""", unsafe_allow_html=True)

col_input, col_submit = st.columns([4.4, 1.3])
with col_input:
    user_query = st.text_input(
        "Search Somaiya Records",
        placeholder="Type any question e.g. What is the fee structure for B.Tech CSE? or Tell me about riidl...",
        label_visibility="collapsed"
    )
with col_submit:
    search_clicked = st.button("🔍 Search UniBox", use_container_width=True)

if search_clicked or user_query:
    if user_query.strip():
        perform_search(user_query.strip())

# Interactive Quick Chips Centered
st.markdown("""<div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; align-items: center; margin-top: 4px; margin-bottom: 1.5rem;"><span style="font-size: 0.82rem; font-weight: 700; color: #A6192E; text-transform: uppercase; letter-spacing: 0.5px;">Try asking:</span><span class="popular-query-pill">💰 First Year Engineering Fees</span><span class="popular-query-pill">🏢 Hostel & Accommodation</span><span class="popular-query-pill">🎓 Scholarships & Financial Aid</span><span class="popular-query-pill">🚀 riidl Startup Grants</span><span class="popular-query-pill">📚 Exam Schedules & Credits</span></div>""", unsafe_allow_html=True)

# --- Results Display ---
if st.session_state.search_results is not None:
    data = st.session_state.search_results
    actual_chunks = data.get("results", [])
    
    if not actual_chunks:
        st.info("No relevant records found. Try rephrasing your search query.")
    else:
        st.markdown(f"### Results for *'{st.session_state.current_query}'*")
        
        for idx, res in enumerate(actual_chunks, 1):
            title: str = res.get("title", "Untitled Somaiya Record")
            url: str = res.get("url", "#")
            text: str = res.get("text", "No content available.")
            chunk_idx: int = res.get("chunk_index", 0)

            # --- Score extraction (dual scores with fallback) ---
            rerank_score = res.get("rerank_score", res.get("score", 0.0))
            vector_score = res.get("vector_score", 0.0)

            try:
                rerank_score = float(rerank_score) if rerank_score is not None else 0.0
            except (ValueError, TypeError):
                rerank_score = 0.0

            try:
                vector_score = float(vector_score) if vector_score is not None else 0.0
            except (ValueError, TypeError):
                vector_score = 0.0

            st.markdown(f"""<div class="result-card"><a href="{url}" target="_blank" class="result-title">{idx}. {title} ↗</a><div class="result-quote">{text}</div><div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #718096; margin-top: 10px; flex-wrap: wrap; gap: 8px;"><span>🔗 <b>Source:</b> <a href="{url}" target="_blank" style="color: #A6192E; text-decoration: underline;">{url}</a> | Chunk #{chunk_idx}</span><div style="display: flex; gap: 8px;"><span style="background: #FDF2F4; color: #A6192E; padding: 2px 8px; border-radius: 4px; font-weight: 600;">🤖 Re-rank: {rerank_score:+.4f}</span><span style="background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 600;">🔢 Vector: {vector_score:.4f}</span></div></div></div>""", unsafe_allow_html=True)

            
        # --- Feedback Section ---
        st.markdown("<br/>", unsafe_allow_html=True)
        if not st.session_state.feedback_given:
            fb_container = st.container()
            with fb_container:
                st.markdown("##### Was this information helpful?")
                f_col1, f_col2, _ = st.columns([1, 1, 6])
                with f_col1:
                    if st.button("👍 Yes", use_container_width=True):
                        submit_feedback(is_helpful=True)
                with f_col2:
                    if st.button("👎 No", use_container_width=True):
                        submit_feedback(is_helpful=False)
        else:
            st.success("✓ Thank you! Your feedback helps us continuously improve the UniBox search model.")

# --- Bottom Section: Explore Somaiya & Footer (Pic 1) ---
st.markdown("""<div class="somaiya-footer"><div class="explore-somaiya-header">Continue Exploring Somaiya</div><div class="explore-tiles"><a href="https://somaiya.edu/en/research" target="_blank" class="explore-tile"><span>Research & Consultancy</span><span>→</span></a><a href="https://somaiya.edu/en/experiential-learning" target="_blank" class="explore-tile"><span>Experiential Learning</span><span>→</span></a><a href="https://somaiya.edu/en/sports-academy" target="_blank" class="explore-tile"><span>Sports Academy</span><span>→</span></a><a href="https://riidl.org" target="_blank" class="explore-tile"><span>Student Incubator (riidl)</span><span>→</span></a></div><div class="footer-columns"><div class="footer-col"><h4>Programmes & Admissions</h4><ul><li><a href="#">Admissions</a></li><li><a href="#">Programmes</a></li><li><a href="#">International Tieups</a></li><li><a href="#">Scholarships</a></li><li><a href="#">Alumni Portal</a></li></ul></div><div class="footer-col"><h4>Quick Links</h4><ul><li><a href="#">My Account Portal</a></li><li><a href="#">Email Login</a></li><li><a href="#">ESS Login</a></li><li><a href="#">IT Policy</a></li></ul></div><div class="footer-col"><h4>Explore</h4><ul><li><a href="#">Sports Academy</a></li><li><a href="#">SCEL</a></li><li><a href="#">riidl</a></li><li><a href="#">Research</a></li><li><a href="#">Somaiya Brand Centre</a></li><li><a href="#">Blog</a></li></ul></div><div class="footer-col"><h4>Get In Touch</h4><ul><li><a href="#">Maps & Directions</a></li><li><a href="#">Contact Directory</a></li><li><a href="#">Faculty Directory</a></li><li><a href="#">Emergency</a></li><li><a href="#">News & Media</a></li><li><a href="#">Events & Updates</a></li></ul></div><div class="footer-col action-btn-list"><a href="#" class="footer-action-button">Apply</a><a href="#" class="footer-action-button">Enquiry</a><a href="#" class="footer-action-button">Careers</a><a href="#" class="footer-action-button">Contact</a><a href="#" class="footer-action-button">Chat with a Student</a></div></div><div class="footer-bottom-bar"><div class="bottom-disclaimers"><a href="#">About</a><a href="#">History</a><a href="#">Mandatory Disclosures</a><a href="#">International Tieups</a><a href="#">COVID-19 Updates</a></div><div class="social-links"><span>f</span> &nbsp; <span>𝕏</span> &nbsp; <span>📷</span> &nbsp; <span>▶</span> &nbsp; <span>in</span></div><div class="bottom-disclaimers"><a href="#">Terms & Conditions</a><a href="#">Privacy Policy</a><a href="#">Disclaimer</a><span>© 2026 Somaiya Vidyavihar University</span></div></div></div>""", unsafe_allow_html=True)