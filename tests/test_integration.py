import json
import time

import pytest
from fastapi.testclient import TestClient

from tests import fake_gemini_server as fg

# ---------- fixtures ----------

CHUNKS = [
    {"url": "https://x.edu/fees", "title": "Fees", "text": "Annual tuition fee for first year B.Tech is Rs 2,00,000.", "chunk_index": 0, "score": 0.9},
    {"url": "https://x.edu/fees", "title": "Fees", "text": "Hostel fee is Rs 90,000 per year.", "chunk_index": 1, "score": 0.8},
    {"url": "https://x.edu/grievance", "title": "Grievance", "text": "Complaints go to the committee.", "chunk_index": 0, "score": 0.5},
]


@pytest.fixture(scope="module")
def server():
    srv, url = fg.start()
    yield url
    srv.shutdown()


@pytest.fixture()
def api(server, monkeypatch):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server)
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-not-real")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    fg.State.mode = "ok"
    fg.State.requests = []
    fg.State.reply = "FAKE ANSWER [1]"
    fg.State.fail_first_n = 0
    fg.State.calls = 0
    import llm.gemini_service as gs
    gs._client = None
    monkeypatch.setattr(time, "sleep", lambda s: None)
    import api.main as m
    monkeypatch.setattr(m, "vector_search", lambda **kw: [dict(c) for c in CHUNKS])
    stored = []
    monkeypatch.setattr(m, "store_feedback", lambda **kw: stored.append(kw) or "abc123")
    monkeypatch.setattr(m, "check_database_connection", lambda: True)
    client = TestClient(m.app)
    client.stored = stored
    return client


def prompt_text(req):
    body = req["body"]
    return " ".join(p.get("text", "") for c in body["contents"] for p in c["parts"])


# ---------- retrieval unchanged ----------

def test_search_without_generation_is_backward_compatible(api):
    r = api.post("/search", json={"query": "tuition fee", "limit": 2})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"query", "count", "results"}
    assert d["count"] == 2
    assert "rerank_score" in d["results"][0] and "vector_score" in d["results"][0]
    assert fg.State.calls == 0          # Gemini not called


def test_empty_query_rejected(api):
    assert api.post("/search", json={"query": "   "}).status_code == 400
    assert api.post("/search", json={"query": ""}).status_code == 422


# ---------- generation ----------

def test_generates_grounded_answer_with_sources(api):
    r = api.post("/search", json={"query": "tuition fee", "limit": 3, "generate_answer": True})
    d = r.json()
    assert r.status_code == 200
    assert d["answer"] == "FAKE ANSWER [1]"
    assert d["model"] == "gemini-2.5-flash"
    assert d["count"] == 3
    urls = [s["url"] for s in d["sources"]]
    assert urls.count("https://x.edu/fees") == 1           # de-duplicated
    assert [s for s in d["sources"] if s["cited"]][0]["url"] == "https://x.edu/fees"

    req = fg.State.requests[-1]
    assert req["path"].endswith("models/gemini-2.5-flash:generateContent")
    assert req["key"] == "test-dummy-key-not-real"   # key goes server -> Google only
    text = prompt_text(req)
    assert "Annual tuition fee" in text and "tuition fee" in text   # chunks + question sent
    sysi = json.dumps(req["body"]["systemInstruction"])
    assert "ONLY on the" in sysi                                  # grounding rule present
    tc = req["body"]["generationConfig"]["thinkingConfig"]      # SDK may emit snake_case or camelCase
    assert tc.get("thinkingBudget", tc.get("thinking_budget")) == 0


def test_no_chunks_still_tells_model_nothing_was_found(api, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "vector_search", lambda **kw: [])
    d = api.post("/search", json={"query": "moon base admissions", "generate_answer": True}).json()
    assert d["count"] == 0 and d["answer"] == "FAKE ANSWER [1]"
    assert "No relevant excerpts" in prompt_text(fg.State.requests[-1])


def test_response_never_contains_api_key(api):
    r = api.post("/search", json={"query": "fee", "generate_answer": True})
    assert "test-dummy-key-not-real" not in r.text
    assert "test-dummy-key-not-real" not in api.get("/health").text


# ---------- follow-ups / history ----------

def test_followup_rewrites_query_and_sends_history(api):
    import api.main as m
    seen = {}
    orig = m.retrieve_chunks
    m.retrieve_chunks = lambda q, l: seen.setdefault("q", q) and orig(q, l)
    try:
        fg.State.reply = "first fee tuition"       # used for the rewrite call AND the answer
        hist = [{"role": "user", "content": "What is the tuition fee?"},
                {"role": "assistant", "content": "It is Rs 2,00,000 per year."}]
        d = api.post("/search", json={"query": "Can you explain that more simply?",
                                      "generate_answer": True, "history": hist}).json()
    finally:
        m.retrieve_chunks = orig
    assert d["retrieval_query"] == "first fee tuition"        # rewritten query used for retrieval
    assert seen["q"] == "first fee tuition"
    assert d["query"] == "Can you explain that more simply?"  # original preserved
    assert len(fg.State.requests) == 2                        # rewrite + answer
    contents = fg.State.requests[-1]["body"]["contents"]
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model", "user"]
    assert "It is Rs 2,00,000 per year." in contents[1]["parts"][0]["text"]
    assert "explain that more simply" in contents[2]["parts"][0]["text"]


def test_history_is_limited(api, monkeypatch):
    monkeypatch.setenv("LLM_MAX_HISTORY_TURNS", "2")
    monkeypatch.setenv("LLM_REWRITE_FOLLOWUPS", "false")
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    api.post("/search", json={"query": "next", "generate_answer": True, "history": hist})
    contents = fg.State.requests[-1]["body"]["contents"]
    assert len(contents) == 2 * 2 + 1
    assert contents[0]["parts"][0]["text"] == "q8"            # oldest turns dropped


def test_history_validation(api):
    bad = {"query": "x", "generate_answer": True, "history": [{"role": "system", "content": "hi"}]}
    assert api.post("/search", json=bad).status_code == 422
    too_many = {"query": "x", "history": [{"role": "user", "content": "a"}] * 21}
    assert api.post("/search", json=too_many).status_code == 422


def test_no_history_means_no_rewrite_call(api):
    api.post("/search", json={"query": "fee", "generate_answer": True})
    assert len(fg.State.requests) == 1


def test_sessions_are_isolated_server_is_stateless(api):
    api.post("/search", json={"query": "secret question A", "generate_answer": True,
                              "history": [{"role": "user", "content": "SESSION_A_MARKER"}]})
    api.post("/search", json={"query": "question B", "generate_answer": True})
    assert "SESSION_A_MARKER" not in prompt_text(fg.State.requests[-1])


# ---------- failure handling ----------

@pytest.mark.parametrize("mode,code", [
    ("403_key", "invalid_api_key"),
    ("429", "rate_limited"),
    ("503", "service_unavailable"),
    ("404", "model_not_found"),
    ("empty", "empty_response"),
    ("blocked", "blocked"),
])
def test_api_failures_degrade_gracefully(api, mode, code):
    fg.State.mode = mode
    r = api.post("/search", json={"query": "tuition fee", "generate_answer": True})
    d = r.json()
    assert r.status_code == 200                      # retrieval results still returned
    assert d["count"] == 3 and d["answer"] is None
    assert d["answer_error"]["code"] == code
    blob = r.text
    assert "test-dummy-key-not-real" not in blob and "Traceback" not in blob


def test_transient_error_is_retried(api):
    fg.State.fail_first_n = 1
    d = api.post("/search", json={"query": "fee", "generate_answer": True}).json()
    assert d["answer"] == "FAKE ANSWER [1]"
    assert fg.State.calls == 2


def test_missing_key(api, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")   # .env.example placeholder
    d = api.post("/search", json={"query": "fee", "generate_answer": True}).json()
    assert d["answer_error"]["code"] == "missing_api_key"
    assert d["count"] == 3
    assert api.get("/health").json()["gemini"] == "not_configured"


def test_network_failure(api, monkeypatch):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:9")   # nothing listens
    import llm.gemini_service as gs
    gs._client = None
    d = api.post("/search", json={"query": "fee", "generate_answer": True}).json()
    assert d["answer_error"]["code"] == "network_error"


def test_rewrite_failure_falls_back_to_original_query(api):
    fg.State.mode = "429"
    d = api.post("/search", json={"query": "explain simpler", "generate_answer": True,
                                  "history": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}).json()
    assert "retrieval_query" not in d and d["count"] == 3


def test_retrieval_error_is_sanitized(api, monkeypatch):
    import api.main as m
    def boom(**kw): raise RuntimeError("connection failed for host with password SECRETPW")
    monkeypatch.setattr(m, "vector_search", boom)
    r = api.post("/search", json={"query": "fee"})
    assert r.status_code == 500 and "SECRETPW" not in r.text


# ---------- feedback / health ----------

def test_feedback_roundtrip_matches_frontend_payload(api):
    payload = {"query": "fee", "helpful": True, "result_urls": ["https://x.edu/fees"]}
    r = api.post("/feedback", json=payload)
    assert r.status_code == 200 and r.json()["success"] is True
    assert api.stored[0] == payload


def test_health(api):
    d = api.get("/health").json()
    assert d == {"api": "ok", "mongodb": "connected", "gemini": "configured"}
