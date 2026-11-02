# UniBox: Gemini + local PDFs (MongoDB-free phase)

## What this version does

- Reads the PDFs in `data/raw_docs/`.
- Splits the extracted text into chunks and creates embeddings locally.
- Retrieves the most relevant chunks for each question using cosine similarity.
- Sends the question and retrieved context to Google Gemini.
- Displays a natural-language answer and source PDF filenames in Streamlit.
- Does not connect to MongoDB or require Atlas credentials.

## 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download the sentence-transformer and tokenizer models.

## 3. Add your Gemini API key

1. Open Google AI Studio: https://aistudio.google.com/app/apikey
2. Create an API key.
3. Copy `.env.example` to a file named `.env`.
4. Replace `your_gemini_api_key_here` with your key.

Never share the key or commit `.env` to GitHub.

## 4. Start the FastAPI backend

Run these commands from the UniBox project root:

```powershell
uvicorn api.main:app --reload
```

On startup, UniBox extracts the local PDFs and builds an in-memory semantic index. This can take a few minutes the first time.

Check the backend at:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs

The health endpoint should show MongoDB as disabled and the local PDF index as ready. Gemini uses `gemini-3.8-flash` first and tries `gemini-3.5-flash-lite` if the primary model is temporarily overloaded.

## 5. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
streamlit run app/ui.py
```

## 6. Try questions

Examples:

- What is the role of the Admission Committee?
- What does the Finance Committee handle?
- How can a student raise a grievance?
- Summarize the university brochure.

Answers are grounded in the PDFs under `data/raw_docs/`. If the documents do not contain the answer, UniBox should say so rather than inventing information.

## Troubleshooting

- **GEMINI_API_KEY missing:** Check that `.env` is in the project root and that the key is set correctly. Restart FastAPI after editing `.env`.
- **Gemini API error:** Verify the key in AI Studio, internet access, model availability, and API quota.
- **No readable PDF text:** The current PDF loader extracts embedded text. Scanned/image-only PDFs need OCR.
- **Slow first startup:** SentenceTransformer and tokenizer model files may need to download once.
- **Port already in use:** Stop the existing process or use a different port and update `API_BASE_URL` in `app/ui.py`.

## Later MongoDB integration

The Gemini generation service is deliberately separate from retrieval. In the next phase, replace `retrieval/local_retriever.py` with a MongoDB-backed retriever that returns the same fields (`title`, `source`, `text`, `chunk_index`, and `score`). The `/chat` endpoint and Gemini prompt can then stay almost unchanged.
