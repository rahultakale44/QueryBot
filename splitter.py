from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_document


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Original pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    documents = load_document()
    chunks = split_documents(documents)

    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content[:1000])