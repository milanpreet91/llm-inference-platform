import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.ingest")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "eda_docs"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def ingest_file(path: str, collection) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    filename = Path(path).name

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    # Upsert in batches of 50
    batch = 50
    for i in range(0, len(chunks), batch):
        collection.upsert(
            ids=ids[i:i+batch],
            documents=chunks[i:i+batch],
            metadatas=metadatas[i:i+batch],

