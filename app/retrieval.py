from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


QDRANT_PATH="qdrant_db"
COLLECTION_NAME="pdfData"


#Creating vector store
def get_vector_store():
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLm-L6-v2"
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


#retrieve relevcant chunks

def retrieve_documents(query):
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=3
    )
    return documents


#main

if __name__ == "__main__":

    query = input("\nAsk a question: ")
    documents = retrieve_documents(query)

    print("\nRetrieved Documents:\n")

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Result {i} ---")

        print("Content:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)