"""Построение векторного индекса базы знаний (Задание 3).

Запуск из корня проекта: .venv/bin/python Task3/build_index.py
"""
import glob
import os
import time

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

KB_DIR = "Task2/knowledge_base"
DB_DIR = "Task3/chroma_db"
COLLECTION = "knowledge_base"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_WORDS = 150
OVERLAP_WORDS = 15


def word_count(text: str) -> int:
    return len(text.split())


def load_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_WORDS,
        chunk_overlap=OVERLAP_WORDS,
        length_function=word_count,
        add_start_index=True,
    )
    ids, texts, metas = [], [], []
    for path in sorted(glob.glob(f"{KB_DIR}/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = text.splitlines()[0].lstrip("# ").strip()
        docs = splitter.create_documents([text])
        for i, d in enumerate(docs):
            name = os.path.basename(path)
            ids.append(f"{name}#{i}")
            texts.append(d.page_content)
            metas.append({
                "source": path,
                "title": title,
                "chunk_id": i,
                "chunks_in_doc": len(docs),
                "start_char": d.metadata["start_index"],
            })
    return ids, texts, metas


def main():
    ids, texts, metas = load_chunks()
    print(f"Чанков: {len(texts)}")

    model = SentenceTransformer(MODEL)
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    print(f"Эмбеддинги: {elapsed:.1f} с, размерность {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metas)
    print(f"В индексе: {col.count()}")


if __name__ == "__main__":
    main()
