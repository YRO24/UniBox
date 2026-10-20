"""
Test setup. The real embedding / cross-encoder models need to be downloaded
from Hugging Face and MongoDB needs Atlas, so they are replaced with light
fakes. Gemini is exercised through the REAL google-genai SDK against a local
fake Gemini HTTP server (see test_gemini_http.py).
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeScores(list):
    def tolist(self):
        return list(self)


class FakeCrossEncoder:
    def __init__(self, *a, **k):
        pass

    def predict(self, pairs):
        # Higher score when the query's words appear in the passage.
        out = []
        for q, t in pairs:
            qw = set(q.lower().split())
            out.append(float(sum(1 for w in qw if w in t.lower())))
        return _FakeScores(out)


_st = types.ModuleType("sentence_transformers")
_st.CrossEncoder = FakeCrossEncoder
_st.SentenceTransformer = lambda *a, **k: None
sys.modules["sentence_transformers"] = _st

_emb = types.ModuleType("ingestion.embedder")
_emb.generate_embeddings = lambda texts, batch_size=32: [[0.0] * 384 for _ in texts]
sys.modules["ingestion.embedder"] = _emb
