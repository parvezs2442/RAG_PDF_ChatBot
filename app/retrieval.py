
from typing import List, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.llm import generate_answer

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "resume_data"

_qdrant_client = None
_vector_store = None
_bm25_instance = None
_corpus_docs = None


def get_qdrant_client():
    """Singleton Qdrant client to prevent multi-instance lock issues."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def get_vector_store():
    """Singleton Qdrant vector store."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = get_qdrant_client()
    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return _vector_store


def get_all_documents() -> List[Document]:
    """Fetch all document chunks and metadata stored in Qdrant."""
    global _corpus_docs
    if _corpus_docs is not None:
        return _corpus_docs

    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    _corpus_docs = [
        Document(
            page_content=p.payload.get("page_content", ""),
            metadata=p.payload.get("metadata", {}),
        )
        for p in points
    ]
    return _corpus_docs


def get_bm25_retriever():
    """Build or retrieve cached BM25 index from document chunks."""
    global _bm25_instance
    if _bm25_instance is not None:
        return _bm25_instance

    docs = get_all_documents()
    if not docs:
        return None

    # Tokenize corpus for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    _bm25_instance = BM25Okapi(tokenized_corpus)
    return _bm25_instance


def bm25_search(query: str, top_k: int = 5) -> List[Document]:
    """Perform keyword-based sparse search using BM25."""
    bm25 = get_bm25_retriever()
    docs = get_all_documents()

    if not bm25 or not docs:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Sort indices by BM25 score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Filter out chunks with 0 keyword overlap if desired, but keep top_k candidates
    return [docs[i] for i in ranked_indices[:top_k]]


def reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_n: int = 5,
) -> List[Document]:
    """
    Combines dense and sparse results using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(doc) = sum(1 / (k + rank))
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Helper to calculate RRF score
    def add_ranks(docs: List[Document]):
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content.strip()
            if key not in doc_map:
                doc_map[key] = doc
                doc_scores[key] = 0.0
            doc_scores[key] += 1.0 / (k + rank)

    add_ranks(vector_docs)
    add_ranks(bm25_docs)

    # Sort documents by fused RRF score
    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


def hybrid_search(query: str, top_k: int = 3, candidate_k: int = 5) -> List[Document]:
    """
    Executes Hybrid Retrieval:
    1. Vector Similarity Search (Dense)
    2. BM25 Search (Sparse)
    3. Merges and ranks via Reciprocal Rank Fusion (RRF)
    """
    # 1. Dense Vector Search
    vector_store = get_vector_store()
    vector_candidates = vector_store.similarity_search(query, k=candidate_k)

    # 2. Sparse BM25 Search
    bm25_candidates = bm25_search(query, top_k=candidate_k)

    # 3. Reciprocal Rank Fusion
    fused_candidates = reciprocal_rank_fusion(
        vector_docs=vector_candidates,
        bm25_docs=bm25_candidates,
        k=60,
        top_n=top_k,
    )

    return fused_candidates


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Document]:
    """Main retrieval entrypoint using hybrid search."""
    return hybrid_search(query, top_k=top_k)


# Alias for backwards compatibility
retrieve_documents = retrieve_relevant_chunks



    
if __name__ == "__main__":

    print("\n\n Retrieval Started ->  ")
    query = input("\n Ask a question -> ")

    documents = retrieve_relevant_chunks(query)
    # print("\n\n Getting relevant chunks -> ")

    #for i, document in enumerate(documents, start=1):

        # print(f"\n--- Result {i} ---")

        # print("Content:")
        # print(document.page_content)

        # print("\nMetadata:")
        # print(document.metadata)

    answer = generate_answer(query, documents)
    print("\n Final Answer using LLM --> ")
    print(answer)
