from src.loader import load_documents
from src.chunking import build_chunks
from src.rag_agent import RAGAgent
import os

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    pdf_path = os.path.join(BASE_DIR, "data", "docs.pdf")

    docs = load_documents(pdf_path)
    
    if not docs:
        raise ValueError("No documents found in data/docs.pdf")

    chunks = build_chunks(docs, chunk_size=700, chunk_overlap=70)
    agent = RAGAgent(chunks)

    query = input("Enter your query: ").strip()

    result = agent.query(query)

    print("ANSWER:\n",result)


if __name__ == "__main__":
    main()