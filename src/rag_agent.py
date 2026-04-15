from src.retriever import HybridRetriever
from src.rerank import Reranker
from src.chain import select_and_plan, generate_answer

class RAGAgent:
    def __init__(self, chunks):
        """
        Initialize the full RAG pipeline.
        """
        self.retriever = HybridRetriever(chunks)
        self.reranker = Reranker()

    def query(self, user_query, retrieve_k=20, rerank_k=5, debug=False):
        """
        Full RAG pipeline:
        1. Retrieve
        2. Rerank
        3. Generate Answer
        """

        
        retrieved = self.retriever.hybrid_search(
            user_query,
            bm25_k=retrieve_k,
            dense_k=retrieve_k,
            final_k=retrieve_k
        )

        if debug:
            print(f"[DEBUG] Retrieved {len(retrieved)} chunks")

        
        reranked = self.reranker.rerank(
            user_query,
            retrieved,
            top_k=rerank_k
        )

        if debug:
            print(f"[DEBUG] Reranked to {len(reranked)} chunks")

        
        plan = select_and_plan(user_query, reranked)

        
        answer = generate_answer(plan)

        if debug:
            return {
                "query": user_query,
                "retrieved": retrieved,
                "reranked": reranked,
                "plan": plan,
                "answer": answer
            }

        return answer