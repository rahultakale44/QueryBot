from langchain_chroma import Chroma

from embeddings import create_embeddings


CHROMA_PATH = "chroma_db"


def create_retriever():
    embeddings = create_embeddings()

    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="research_papers"
    )

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


if __name__ == "__main__":
    retriever = create_retriever()

    query = "What this paper is talking about?"

    documents = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Retrieved documents: {len(documents)}")

    print("\n--- Retrieved Document Preview ---\n")

    for i, document in enumerate(documents, start=1):
        print(f"Document {i}")
        print(document.page_content[:500])
        print("-" * 80)