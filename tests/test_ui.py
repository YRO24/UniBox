"""Headless Streamlit tests (streamlit.testing AppTest) with the HTTP layer mocked."""
from unittest import mock

import requests
from streamlit.testing.v1 import AppTest

from pathlib import Path
UI = str(Path(__file__).resolve().parents[1] / "app" / "ui.py")

RESULTS = [{"url": "https://x.edu/fees", "title": "Fees", "text": "Fee is 2 lakh.",
            "chunk_index": 0, "rerank_score": 3.2, "vector_score": 0.8}]


class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status
    def raise_for_status(self):
        if self.status_code >= 400:
            e = requests.exceptions.HTTPError("err"); e.response = self; raise e
    def json(self):
        return self._d


def ok(answer="The fee is about 2 lakh. [1]"):
    return FakeResp({"query": "q", "count": 1, "results": RESULTS, "answer": answer,
                     "sources": [{"ids": [1], "title": "Fees", "url": "https://x.edu/fees", "cited": True}]})


def ask(at, text):
    at.text_input[0].set_value(text)
    [b for b in at.button if b.label == "Search"][0].click()   # form submit button
    return at.run()


def test_full_flow_and_followup_history():
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json)); return ok() if url.endswith("/search") else FakeResp({"success": True})
    with mock.patch("requests.post", fake_post):
        at = AppTest.from_file(UI, default_timeout=30).run()
        assert not at.exception
        at = ask(at, "What is the fee?")
        assert not at.exception, at.exception
        assert len(calls) == 1                                    # exactly one backend call
        body = calls[0][1]
        assert body["generate_answer"] is True and body["history"] == [] and body["limit"] == 6
        assert any("2 lakh" in m.value for m in at.markdown)
        assert [m.name for m in at.chat_message] == ["user", "assistant"]

        at = ask(at, "Explain that more simply")
        assert len(calls) == 2
        assert calls[1][1]["history"] == [
            {"role": "user", "content": "What is the fee?"},
            {"role": "assistant", "content": "The fee is about 2 lakh. [1]"}]
        assert len(at.chat_message) == 4


def test_rerun_does_not_resubmit_and_feedback_works():
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json)); return ok() if url.endswith("/search") else FakeResp({"success": True})
    with mock.patch("requests.post", fake_post):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "What is the fee?")
        searches = lambda: [c for c in calls if c[0].endswith("/search")]
        assert len(searches()) == 1
        yes = [b for b in at.button if b.label == "👍 Yes"][0]
        yes.click(); at.run()
        fb = [c for c in calls if c[0].endswith("/feedback")]
        assert len(fb) == 1 and fb[0][1] == {"query": "What is the fee?", "helpful": True,
                                             "result_urls": ["https://x.edu/fees"]}
        assert len(searches()) == 1                               # feedback click did NOT re-search
        assert any("Feedback submitted" in s.value for s in at.success)
        at.run()
        assert len(searches()) == 1


def test_generation_failure_shows_error_and_passages():
    def fake_post(url, json=None, timeout=None):
        return FakeResp({"query": "q", "count": 1, "results": RESULTS, "answer": None,
                         "answer_error": {"code": "rate_limited", "message": "Too many requests, wait."}})
    with mock.patch("requests.post", fake_post):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "fee?")
        assert any("Too many requests" in e.value for e in at.error)
        assert any("Retrieved passages" in e.label for e in at.expander)


def test_backend_down_and_http_error():
    with mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "fee?")
        assert any("Cannot reach the backend" in e.value for e in at.error)
    with mock.patch("requests.post", lambda *a, **k: FakeResp({"detail": "Search failed."}, 500)):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "fee?")
        assert any("500" in e.value and "Search failed" in e.value for e in at.error)


def test_failed_turn_not_sent_as_history_and_new_conversation_clears():
    seq = [requests.exceptions.ConnectionError(), ok()]
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        r = seq.pop(0)
        if isinstance(r, Exception): raise r
        return r
    with mock.patch("requests.post", fake_post):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "first (fails)")
        at = ask(at, "second")
        assert calls[1]["history"] == []                          # failed turn excluded
        new = [b for b in at.button if "New conversation" in b.label][0]
        new.click(); at.run()
        assert len(at.chat_message) == 0


def test_empty_query_does_nothing():
    with mock.patch("requests.post") as p:
        at = AppTest.from_file(UI, default_timeout=30).run()
        at = ask(at, "   ")
        assert p.call_count == 0
