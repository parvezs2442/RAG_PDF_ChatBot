# AI Research Assistant 🤖

An AI-powered Research Platform that allows users to ask questions and conduct research using private documents (PDFs) and advanced AI. 

Built as part of a **10-day hands-on GenAI engineering roadmap**, the project evolves from a foundational Document Knowledge Base (RAG) into an autonomous Multi-Agent Research Assistant capable of web browsing, cross-source analysis, and citation-backed reporting.

---

## 📌 Project Status

- **Current Progress:** Day 2 of 10 Completed
- **Current Capability:** Basic RAG pipeline with PDF ingestion, semantic vector search, grounded LLM response generation, and a FastAPI backend.

---

## 🚀 10-Day Project Roadmap

- [x] **Day 1: Foundation & Project Skeleton** — Backend API setup, environment variables, and LLM integration.
- [x] **Day 2: Basic RAG: Document Knowledge Base** — PDF text extraction, chunking, embeddings, vector database, and grounded Q&A.
- [ ] **Day 3: Advanced RAG** — Hybrid retrieval (vector + keyword search), reranking, and metadata filtering.
- [ ] **Day 4: RAG Intelligence** — Conversation-aware query rewriting, multi-query expansion, and parent-child retrieval.
- [ ] **Day 5: Web Research & Ingestion** — Live web search tool, URL reading, HTML content cleaning, and source citations.
- [ ] **Day 6: Agentic RAG & Tool Calling** — Planner agent, autonomous tool selection, state management, and bounded retries.
- [ ] **Day 7: Multi-Agent Research Pipeline** — Coordinated agent team: *Researcher → Reader → Analyst → Writer → Critic*.
- [ ] **Day 8: Evaluation & Reliability** — RAG quality metrics (faithfulness, context relevance, citation accuracy).
- [ ] **Day 9: Security & Guardrails** — Prompt injection defense, input validation, and access control.
- [ ] **Day 10: Production & Observability** — UI dashboard, response streaming, latency/token tracing, and deployment.

---

## 🧠 How It Works (Current Architecture)

The system currently operates across two pipelines:

### 1. Document Ingestion (Offline)
1. **Load Document:** Extracts text from uploaded PDF files using `PyPDFLoader`.
2. **Chunking:** Splits the text into 500-character chunks with a 100-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embeddings:** Converts each chunk into a 384-dimensional vector using Hugging Face's `all-MiniLM-L6-v2`.
4. **Vector Storage:** Stores chunk embeddings and metadata locally in a `Qdrant` vector database.

### 2. Retrieval & Generation (Online / Query Time)
1. **Query:** User submits a question via the FastAPI `/ask` endpoint or CLI.
2. **Similarity Search:** The query is embedded and compared against Qdrant vectors to find the Top-3 closest chunks.
3. **Prompt Grounding:** Retrieved chunks are injected into a strict system prompt.
4. **LLM Synthesis:** Google Gemini (`gemini-3.5-flash`) generates an answer strictly grounded in the document context.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic
- **LLM:** Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
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
