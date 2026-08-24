from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "data/data.pdf"

#load pdf & extract text
def load_pdf():
    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    return documents


if __name__ == "__main__":
    documents = load_pdf()

    print(f"Loaded {len(documents)} pages")

    for document in documents:
        print(f"\nPage: {document.metadata['page'] + 1}")
        print(document.page_content[:300])