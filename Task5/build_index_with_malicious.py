"""Копирует основной индекс в Task5/chroma_db и добавляет «злонамеренный» файл так же, как остальные документы.

Запуск из корня проекта: .venv/bin/python Task5/build_index_with_malicious.py
"""
import os
import shutil
import sys

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "Task3")
from build_index import CHUNK_WORDS, COLLECTION, MODEL, OVERLAP_WORDS, word_count  # noqa: E402

SRC_DB = "Task3/chroma_db"
DST_DB = "Task5/chroma_db"
FILE = "Task5/malicious.txt"


def main():
    shutil.rmtree(DST_DB, ignore_errors=True)
    shutil.copytree(SRC_DB, DST_DB)
    col = chromadb.PersistentClient(path=DST_DB).get_collection(COLLECTION)
    before = col.count()

    with open(FILE, encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_WORDS, chunk_overlap=OVERLAP_WORDS, length_function=word_count, add_start_index=True
    )
    docs = splitter.create_documents([text])
    name = os.path.basename(FILE)
    model = SentenceTransformer(MODEL)
    emb = model.encode([d.page_content for d in docs], normalize_embeddings=True)
    col.add(
        ids=[f"{name}#{i}" for i in range(len(docs))],
        embeddings=emb.tolist(),
        documents=[d.page_content for d in docs],
        metadatas=[{"source": FILE, "title": name, "chunk_id": i, "chunks_in_doc": len(docs),
                    "start_char": d.metadata["start_index"]} for i, d in enumerate(docs)],
    )
    print(f"Было чанков: {before}, стало: {col.count()}")


if __name__ == "__main__":
    main()
