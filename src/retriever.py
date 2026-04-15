import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

class HybridRetriever:

    def __init__(self, chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        self.tokenized_corpus = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.embedder = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedder.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)  # type: ignore

    def _tokenize(self, text):
        return text.lower().split()

    def bm25_search(self, query, top_k=10):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), self.chunks[i]) for i in idxs]

    def dense_search(self, query, top_k=10):
        q_emb = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k) # type: ignore

        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i != -1:
                results.append((int(i), float(s), self.chunks[i]))
        return results

    
    def hybrid_search(self, query, bm25_k=20, dense_k=20, final_k=10):
        bm25_hits = self.bm25_search(query, top_k=bm25_k)
        dense_hits = self.dense_search(query, top_k=dense_k)

        merged = {}

        for idx, score, chunk in bm25_hits:
            merged[idx] = {
                "idx": idx,
                "bm25_score": score,
                "dense_score": 0.0,
                "chunk": chunk
            }

        for idx, score, chunk in dense_hits:
            if idx not in merged:
                merged[idx] = {
                    "idx": idx,
                    "bm25_score": 0.0,
                    "dense_score": score,
                    "chunk": chunk
                }
            else:
                merged[idx]["dense_score"] = score

        candidates = list(merged.values())

        def normalize(scores):
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s == 0:
                return [0 for _ in scores]
            return [(s - min_s) / (max_s - min_s) for s in scores]

        bm25_scores = [c["bm25_score"] for c in candidates]
        dense_scores = [c["dense_score"] for c in candidates]

        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)

        for i, c in enumerate(candidates):
            c["hybrid_score"] = 0.5 * bm25_norm[i] + 0.5 * dense_norm[i]

        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return [
            {
                "score": c["hybrid_score"],
                "text": c["chunk"]["text"],
                "source": c["chunk"].get("source"),
                "page": c["chunk"].get("page"),
                "chunk_id": c["chunk"].get("chunk_id")
            }
            for c in candidates[:final_k]
        ]