"""Поиск по индексу. Запуск: .venv/bin/python Task3/search.py "запрос" [k]"""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def search(query: str, k: int = 3):
    model = SentenceTransformer(MODEL)
    emb = model.encode([query], prompt_name="query", normalize_embeddings=True)
    col = chromadb.PersistentClient(path="Task3/chroma_db").get_collection("knowledge_base")
    res = col.query(query_embeddings=emb.tolist(), n_results=k)
    return list(zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]))


if __name__ == "__main__":
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"Запрос: {q}\n")
    for i, (cid, dist, meta, doc) in enumerate(search(q, k), 1):
        print(f"{i}. {cid} | {meta['title']} | расстояние {dist:.3f}\n   {doc[:300]}\n")
