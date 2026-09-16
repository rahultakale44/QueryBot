from langchain_huggingface import HuggingFaceEmbeddings
from splitter import split_documents
from pdf_loader import load_document


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    documents = load_document()
    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector = embeddings.embed_query(
        "What this paper is talking about?"
    )

    print(f"Embedding dimensions: {len(vector)}")
    print("\nFirst 10 values:")
    print(vector[:10])