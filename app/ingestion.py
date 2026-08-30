
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

#load pdf 
FILE_PATH = "data/Parvez_FullStack_AI (1).pdf"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "resume_data"

def upload_pdf():
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    return documents 


# splitting the data into chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 100
    )
    chunks = splitter.split_documents(documents)
    return chunks

#Embeddings & stroring in Vector DB

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    qdrant_client = QdrantClient(
        path=QDRANT_PATH
    )

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)
    return vector_store




if __name__ == "__main__":
    print("Uploading Pdf .....")
    documents = upload_pdf()

    print("\n\n CHunking Starts")
    chunks = split_text(documents)

    print("\n\n vectoe db creation starts")
    vector_store = create_vector_store(chunks)

    print("\n\n vector db created")

















































