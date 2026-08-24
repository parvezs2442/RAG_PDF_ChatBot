from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval import retrieve_documents
from app.llm import generate_answer


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    # Retrieve relevant chunks from Qdrant
    documents = retrieve_documents(request.query)

    # Generate answer using LLM
    answer = generate_answer(
        request.query,
        documents
    )

    return {
        "question": request.query,
        "answer": answer
    }