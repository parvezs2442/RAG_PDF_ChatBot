# AI Research Assistant 🤖

An enterprise-grade **Retrieval-Augmented Generation (RAG)** platform designed to ingest private documents, perform high-precision hybrid semantic retrieval, rerank results with cross-encoders, and generate verifiable, grounded answers using Google Gemini.

---

## 🧠 System Architecture & Workflow

The platform operates across two distinct pipelines:

```
=== 1. DOCUMENT INGESTION PIPELINE (Offline / Batch) ===

 [ Source Document (PDF) ]
            │
            ▼
    [ PyPDFLoader ] ─────────────────> Extracts text pages & metadata
            │
            ▼
 [ RecursiveCharacterTextSplitter ]
  (chunk_size=500, overlap=100) ─────> Coherent chunks preserving boundaries
            │
            ▼
 [ all-MiniLM-L6-v2 Embeddings ] ───> Generates 384-dimensional dense vectors
            │
            ▼
   [ Qdrant Vector Store ] ──────────> Persisted locally in 'qdrant_db/'


=== 2. ADVANCED RETRIEVAL & GENERATION PIPELINE (Online / Real-time) ===

                         [ User Research Query ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      [ Dense Vector Search ]             [ Sparse BM25 Search ]
     (Semantic & Conceptual Match)        (Exact Keyword/Acronym Match)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                     [ Reciprocal Rank Fusion (RRF) ]
                       Candidate Pool (Top 8 Chunks)
                                    │
                                    ▼
                     [ Cross-Encoder RERANKER ] 🌟
                  (cross-encoder/ms-marco-MiniLM-L-6-v2)
                 Deep [Query + Chunk] mutual cross-attention
                                    │
                                    ▼
                     [ Quality Threshold & Filter ]
                        Drops irrelevant candidates
                                    │
                                    ▼
                         [ Top-3 Pure Evidence ]
                                    │
                                    ▼
                    [ Google Gemini (gemini-3.5-flash) ]
                   "Answer ONLY using provided context"
                                    │
                                    ▼
                    [ Grounded, Hallucination-Free Answer ]
```

---

## 🔍 Why Advanced RAG? (Dense + Sparse + Reranker)

| Stage | What it does | Why it is necessary |
| :--- | :--- | :--- |
| **Dense Vector (Qdrant)** | Semantic similarity via Cosine distance | Understands concepts, synonyms, and context even with different phrasing. |
| **Sparse BM25 (rank-bm25)** | Exact keyword matching & token rarity (TF-IDF) | Catches acronyms, specific IDs, model names, and exact terms that vector embeddings compress or miss. |
| **Reciprocal Rank Fusion (RRF)** | Rank-based candidate fusion: $\sum \frac{1}{60 + \text{rank}}$ | Combines vector and BM25 candidate lists into a single ranked pool without score normalization bias. |
| **Cross-Encoder Reranker** | Joint transformer cross-attention on `[Query, Chunk]` | Bi-encoders encode queries and chunks independently; the Cross-Encoder deeply evaluates true question-answering relevance. |
| **Quality Thresholding** | Drops chunks with relevance scores below cutoff | Prevents LLM hallucinations when questions are out-of-domain. |

---

## 🛠️ Tech Stack

- **Backend Framework:** Python 3.11+, FastAPI, Uvicorn, Pydantic
- **LLM Engine:** Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **Dense Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **Sparse Search Algorithm:** BM25 (`rank-bm25`, `BM25Okapi`)
- **Reranking Engine:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector Database:** Qdrant (Local disk persistence with Singleton client locking)
- **Document Ingestion:** LangChain, `pypdf`
- **Environment Management:** `python-dotenv`

---

## 📁 Repository Structure

```text
├── app/
│   ├── __init__.py
│   ├── ingestion.py      # PDF parsing, text chunking, and Qdrant vector indexing
│   ├── retrieval.py      # Hybrid retrieval (Vector + BM25), RRF, and Cross-Encoder reranking
│   ├── llm.py            # Gemini client & grounded prompt synthesis
│   └── main.py           # FastAPI REST API endpoints
├── data/                 # Source PDFs and documents
├── qdrant_db/            # Local Qdrant vector database storage
├── requirements.txt      # Project dependencies
├── .env                  # Environment secrets (API keys)
├── .gitignore            # Git ignore configurations
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/parvezs2442/AI_Research_Assistant.git
cd AI_Research_Assistant
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## 🏃 Execution Guide

### Step 1: Ingest Documents into Vector Database
Place your PDF in the `data/` folder and run the offline ingestion pipeline:
```bash
python -m app.ingestion
```

### Step 2: Test Advanced Retrieval in Terminal
Execute the interactive terminal pipeline with real-time diagnostic logging:
```bash
python -m app.retrieval
```

*Sample Diagnostic Terminal Output:*
```text
=======================================================
[Day 3 Pipeline Diagnostics - Reranker Scores]
=======================================================
Rank 1 | Score:   0.94 | Snippet: PARVEZ SAIFI  Full Stack AI Engineer  New Delhi, India...
Rank 2 | Score: -10.08 | Snippet: PROJECTS  AI Video Assistant  Multilingual Meeting...
=======================================================
```

### Step 3: Start the FastAPI Server
Launch the production API server:
```bash
uvicorn app.main:app --reload
```
The REST API will be available at `http://127.0.0.1:8000`.

---

## 📡 API Endpoints

### 1. Health Check
* **Endpoint:** `GET /`
* **Response:**
  ```json
  {
    "message": "AI Research Platform RAG API is running"
  }
  ```

### 2. Research Q&A
* **Endpoint:** `POST /ask`
* **Request Body:**
  ```json
  {
    "query": "What projects has Parvez built?"
  }
  ```
* **Response:**
  ```json
  {
    "question": "What projects has Parvez built?",
    "answer": "Based on the provided document, Parvez has built the following project:\n\n* **AI Video Assistant — Multilingual Meeting Intelligence & RAG**: A multilingual meeting-intelligence pipeline that processes YouTube URLs and audio/video uploads, combining local Whisper for English transcription and Sarvam AI for Hindi/Hinglish transcription."
  }
  ```

Interactive Swagger documentation is live at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 👤 Author

**Parvez Saifi**  
- GitHub: [@parvezs2442](https://github.com/parvezs2442)
