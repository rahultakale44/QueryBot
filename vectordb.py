from langchain_chroma import Chroma

from pdf_loader import load_document
from splitter import split_documents
from embeddings import create_embeddings


CHROMA_PATH = "chroma_db"


def create_vector_database():
    documents = load_document()
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="research_papers"
    )

    print(f"Stored {len(chunks)} chunks in Chroma.")
    print(f"Database location: {CHROMA_PATH}")

    return vector_db


if __name__ == "__main__":
    create_vector_database()