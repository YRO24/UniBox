# UniBox

Streamlit + FastAPI + MongoDB Atlas Vector Search RAG assistant. Answers are written by
**Google Gemini 2.5 Flash** from document chunks retrieved by the existing pipeline
(MiniLM embeddings -> Atlas `$vectorSearch` -> cross-encoder re-rank).

## Request flow

1. Streamlit (`app/ui.py`) sends `POST /search` with `{query, limit: 6, generate_answer: true, history}`;
   `history` is the last few turns of that browser session only.
2. FastAPI (`api/main.py`): for follow-ups, Gemini first rewrites the message into a standalone
   search query (`llm.gemini_service.rewrite_followup_query`; falls back to the original text).
3. `retrieve_chunks()` embeds the query, pulls 30 candidates from Atlas Vector Search and re-ranks
   them with the cross-encoder; the top chunks are kept.
4. `llm.gemini_service.generate_answer()` sends the question, the numbered chunks and the limited
   history to Gemini with a grounding system prompt.
5. The response contains the unchanged `results` plus `answer`, `sources`, `model`
   (or `answer: null` + `answer_error` if Gemini failed; retrieval results are still returned).
6. Streamlit renders the answer, a Sources list, the retrieved passages (expander) and feedback buttons.

## Setup (Windows PowerShell)

```powershell
cd UniBox-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env                           # fill in MONGODB_URI and GEMINI_API_KEY

# Optional: verify MongoDB connectivity
python scripts\test_mongodb.py

# Only if the collection is empty / documents changed
python scripts\run_ingestion.py

# Terminal 1 - backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 - frontend (activate the venv again in this terminal)
streamlit run app\ui.py
```

Open http://localhost:8501. Check http://localhost:8000/health: it should report
`"mongodb": "connected"` and `"gemini": "configured"`.

## Getting a Gemini API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** (choose or create a Google Cloud project when asked).
3. Copy the key into `.env` as `GEMINI_API_KEY=...`. Never commit `.env` (it is in `.gitignore`).
4. Restart the backend after changing `.env`.

The key is read only from the `GEMINI_API_KEY` environment variable (loaded from `.env` by
`python-dotenv`). It is not written anywhere in the Python code, is used only by the FastAPI
backend, and is never sent to Streamlit or the browser. `.env.example` ships with the
placeholder `your_gemini_api_key_here`; the backend treats that placeholder as "not configured"
and the UI then shows a clear message while document retrieval keeps working.

## MongoDB Atlas: manual setup

`scripts/create_vector_index.py` is an empty file in the original project, so the vector index
must already exist or be created by hand: Atlas -> your cluster -> **Atlas Search** -> **Create
Search Index** -> **Vector Search** -> JSON editor, database/collection = `MONGODB_DB_NAME` /
`MONGODB_COLLECTION_NAME`, index name = `MONGODB_VECTOR_INDEX` (default `vector_index`):

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" }
  ]
}
```

Also make sure your IP is allowed under **Network Access**. No index change is needed for Gemini.

## Tests

```powershell
pip install pytest httpx
python -m pytest tests -q
```

The tests replace the Hugging Face models and MongoDB with fakes and run the real `google-genai`
SDK against a local fake Gemini HTTP server. They do not call Google or Atlas.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "The AI answer service is not configured" | `GEMINI_API_KEY` missing or still the placeholder; edit `.env`, restart uvicorn |
| "The Gemini API key was rejected" | Wrong/revoked key; create a new one in AI Studio |
| "too many requests (rate limit or quota)" | Quota hit; wait a minute or enable billing |
| "configured Gemini model was not found" | Check `GEMINI_MODEL` (default `gemini-2.5-flash`) |
| "Could not reach the AI service" | No internet, or a proxy/firewall blocks `generativelanguage.googleapis.com` |
| "Cannot reach the backend" in the UI | uvicorn not running on port 8000; or set `UNIBOX_API_BASE_URL` |
| `/health` shows `mongodb: disconnected` | Bad `MONGODB_URI`, Atlas IP allow-list, or paused cluster |
| Search returns 0 results | Collection empty (run ingestion) or vector index name/dimensions wrong |
| First start is slow | MiniLM and cross-encoder models download on first run |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass` |
