from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

PDF_PATH = "data/data.pdf"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "pdfData"


#Loading PDF
def load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents

#Splitting documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

#Creating Qdrant Vector Store


def create_vector_store(chunks):
    print("Loading embedding model...")
    # Local HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Connecting to Qdrant...")

    # Local Qdrant database
    qdrant_client = QdrantClient(
        path=QDRANT_PATH
    )

    # Create collection if it doesn't exist
    if not qdrant_client.collection_exists(COLLECTION_NAME):

        print("Creating Qdrant collection...")

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    # Connect LangChain with Qdrant
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    print("Adding chunks to Qdrant...")

    # Convert chunks into embeddings
    # and store them in Qdrant
    vector_store.add_documents(chunks)
    return vector_store


if __name__ == "__main__":

    print("Loading PDF...")
    documents = load_pdf()

    print(f"Loaded {len(documents)} pages")
    print("\nSplitting documents...")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nCreating vector store...")
    vector_store = create_vector_store(chunks)

    print("\nVector store created successfully!")