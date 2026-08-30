
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.llm import generate_answer

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "resume_data"

def get_vector_store():

    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    qdrant_client = QdrantClient(
        path=QDRANT_PATH
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    return vector_store


def retrieve_relevant_chunks(query):


    vector_store = get_vector_store()
    relevant_chunks = vector_store.similarity_search(
        query,
        3
    )
    return relevant_chunks



    
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
