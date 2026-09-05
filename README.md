# AI Research Assistant 🤖

An AI-powered Research Platform and Retrieval-Augmented Generation (RAG) system that allows users to ask questions and extract insights from private documents (PDFs) with high accuracy and factual grounding.

---

## 🧠 How It Works (Architecture)

The system currently operates across two pipelines:

### 1. Document Ingestion (Offline)
1. **Load Document:** Extracts text from uploaded PDF files using `PyPDFLoader`.
2. **Chunking:** Splits the text into 500-character chunks with a 100-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embeddings:** Converts each chunk into a 384-dimensional vector using Hugging Face's `all-MiniLM-L6-v2`.
4. **Vector Storage:** Stores chunk embeddings and metadata locally in a `Qdrant` vector database.

### 2. Advanced Retrieval & Generation Pipeline (Online / Query Time)
1. **User Query:** User submits a research question via the FastAPI `/ask` endpoint or CLI.
2. **Hybrid Candidate Search:**
   - **Dense Search (Semantic):** Qdrant vector similarity search via `all-MiniLM-L6-v2`.
   - **Sparse Search (Exact Keywords):** BM25 index search via `rank-bm25`.
   - **Reciprocal Rank Fusion (RRF):** Combines dense and sparse results ($1 / (60 + \text{rank})$) to assemble a wide candidate pool.
3. **Cross-Encoder Reranking:** 
   - A dedicated Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) evaluates deep `[Query + Chunk]` interactions.
   - Assigns true relevance scores to each candidate chunk.
4. **Relevance Thresholding & Deduplication:** Filters out poor-scoring or duplicate chunks.
5. **Prompt Grounding & Synthesis:** Best chunks are formatted as context for Google Gemini (`gemini-3.5-flash`), producing a verified, grounded answer.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic
- **LLM:** Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **Dense Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Sparse Keyword Search:** BM25 (`rank-bm25`)
- **Reranker Model:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector Database:** Qdrant (Local disk-based storage)
- **Document Processing:** LangChain, `pypdf`
- **Configuration:** `python-dotenv`

---

## 📁 Project Structure

```text
├── app/
│   ├── __init__.py
│   ├── ingestion.py      # PDF parsing, text chunking, and Qdrant indexing
│   ├── retrieval.py      # Similarity search to fetch relevant document chunks
│   ├── llm.py            # Gemini client & grounded answer prompt logic
│   └── main.py           # FastAPI REST application endpoints
├── data/                 # Folder for reference PDFs / documents
├── qdrant_db/            # Local Qdrant vector database storage
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore            # Files ignored by git
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/parvezs2442/AI_Research_Assistant.git
cd AI_Research_Assistant
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root folder:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## 🏃 How to Run

### Step 1: Ingest Your Document
Place your PDF in the `data/` folder and run the ingestion script to create vector embeddings:
```bash
python -m app.ingestion
```

### Step 2: Test via Terminal (Optional)
Test the retrieval and generation pipeline interactively in your terminal:
```bash
python -m app.retrieval
```

### Step 3: Run the FastAPI Web Server
Start the backend server:
```bash
uvicorn app.main:app --reload
```
The server will start at: `http://127.0.0.1:8000`

---

## 📡 API Endpoints

### 1. Health Check
- **Endpoint:** `GET /`
- **Response:**
  ```json
  {
    "message": "RAG API is running"
  }
  ```

### 2. Ask Question
- **Endpoint:** `POST /ask`
- **Request Body:**
  ```json
  {
    "query": "What are the candidate's core technical skills?"
  }
  ```
- **Response:**
  ```json
  {
    "question": "What are the candidate's core technical skills?",
    "answer": "According to the document, the candidate has expertise in Python, FastAPI, LangChain, Qdrant, and Generative AI application development."
  }
  ```

Interactive Swagger documentation is available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 👤 Author

**Parvez Saifi**  
- GitHub: [@parvezs2442](https://github.com/parvezs2442)
