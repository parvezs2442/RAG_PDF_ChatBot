
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from app.llm import generate_answer

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "resume_data"

_qdrant_client = None
_vector_store = None
_bm25_instance = None
_corpus_docs = None
_reranker_instance = None


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


def get_reranker():
    """Singleton Cross-Encoder reranker model (MS MARCO MiniLM)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_instance


def get_all_documents(metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
    """Fetch all document chunks and metadata stored in Qdrant with optional filtering."""
    global _corpus_docs
    if _corpus_docs is None:
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

    if not metadata_filter:
        return _corpus_docs

    # Filter documents based on metadata criteria
    return [
        doc
        for doc in _corpus_docs
        if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())
    ]


def get_bm25_retriever(metadata_filter: Optional[Dict[str, Any]] = None):
    """Build or retrieve cached BM25 index from document chunks."""
    global _bm25_instance
    docs = get_all_documents(metadata_filter=metadata_filter)
    if not docs:
        return None, []

    if metadata_filter is None and _bm25_instance is not None:
        return _bm25_instance, docs

    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    if metadata_filter is None:
        _bm25_instance = bm25
    return bm25, docs


def bm25_search(
    query: str,
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Perform keyword-based sparse search using BM25."""
    bm25, docs = get_bm25_retriever(metadata_filter=metadata_filter)
    if not bm25 or not docs:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked_indices[:top_k]]


def reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_n: int = 8,
) -> List[Document]:
    """Combines dense and sparse results using Reciprocal Rank Fusion (RRF)."""
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def add_ranks(docs: List[Document]):
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content.strip()
            if key not in doc_map:
                doc_map[key] = doc
                doc_scores[key] = 0.0
            doc_scores[key] += 1.0 / (k + rank)

    add_ranks(vector_docs)
    add_ranks(bm25_docs)

    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


def hybrid_search(
    query: str,
    candidate_k: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Generates a wide candidate pool via Hybrid Retrieval (Dense Vector + Sparse BM25)."""
    vector_store = get_vector_store()
    vector_candidates = vector_store.similarity_search(query, k=candidate_k)
    bm25_candidates = bm25_search(query, top_k=candidate_k, metadata_filter=metadata_filter)

    fused_candidates = reciprocal_rank_fusion(
        vector_docs=vector_candidates,
        bm25_docs=bm25_candidates,
        k=60,
        top_n=candidate_k,
    )
    return fused_candidates


def rerank_candidates(
    query: str,
    candidates: List[Document],
    top_k: int = 3,
    relevance_threshold: Optional[float] = None,
    verbose: bool = True,
) -> List[Document]:
    """
    Cross-Encoder Reranking and Relevance Thresholding.
    Deeply evaluates [Query + Chunk] mutual attention and ranks candidates by true relevance.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = []
    for doc, score in zip(candidates, scores):
        doc.metadata["rerank_score"] = float(score)
        if relevance_threshold is None or score >= relevance_threshold:
            scored_candidates.append((doc, float(score)))

    # Sort descending by Cross-Encoder score
    scored_candidates.sort(key=lambda item: item[1], reverse=True)

    if verbose and scored_candidates:
        print("\n" + "=" * 55)
        print("[Retrieval Pipeline Diagnostics - Reranker Scores]")
        print("=" * 55)
        for rank, (doc, sc) in enumerate(scored_candidates[:top_k], start=1):
            clean_snippet = (
                doc.page_content.encode("ascii", "ignore")
                .decode("ascii")
                .replace("\n", " ")[:85]
            )
            print(f"Rank {rank} | Score: {sc:6.2f} | Snippet: {clean_snippet}...")
        print("=" * 55 + "\n")

    return [doc for doc, _ in scored_candidates[:top_k]]


def advanced_retrieval(
    query: str,
    top_k: int = 3,
    candidate_k: int = 8,
    relevance_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Document]:
    """
    Complete Retrieval Pipeline:
    Query -> Hybrid Retrieval (Dense + Sparse) -> Candidate Pool -> Cross-Encoder Reranker -> Threshold -> Top-K Context
    """
    # 1. Candidate Generation via Hybrid Search
    candidates = hybrid_search(
        query=query,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
    )

    # 2. Deep Reranking & Quality Threshold Filtering
    best_chunks = rerank_candidates(
        query=query,
        candidates=candidates,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        verbose=verbose,
    )

    return best_chunks


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    candidate_k: int = 8,
    relevance_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Document]:
    """Main retrieval entrypoint using the advanced hybrid and reranking pipeline."""
    return advanced_retrieval(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        relevance_threshold=relevance_threshold,
        metadata_filter=metadata_filter,
        verbose=verbose,
    )


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
