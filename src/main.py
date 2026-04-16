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
    

    while True:
        query = input("Enter your query: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        if not query:
            print("Please enter a valid query.\n")
            continue

        try:
            result = agent.query(query)
            print("\nANSWER:\n", result, "\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    
    main()