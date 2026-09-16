from langchain_community.document_loaders import PyPDFLoader


PDF_PATH = "research_paper.pdf"


def load_document():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    return documents


if __name__ == "__main__":
    documents = load_document()

    print("\nFirst page preview:\n")
    print(documents[0].page_content[:1000])