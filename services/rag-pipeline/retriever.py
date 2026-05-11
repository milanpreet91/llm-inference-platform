import os
import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("rag.retriever")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "eda_docs"
TOP_K = 3


class RAGRetriever:
    def __init__(self):
        self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("RAGRetriever initialised")

    def retrieve(self, query: str) -> str:
        """Retrieve top-k relevant chunks and return as context string."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=TOP_K,
                include=["documents", "metadatas"],
            )
            docs = results["documents"][0]
            sources = [m["source"] for m in results["metadatas"][0]]

            if not docs:
                return ""

            context_parts = []
            for doc, src in zip(docs, sources):
                context_parts.append(f"[Source: {src}]\n{doc}")

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""

    def build_prompt(self, query: str) -> str:
        """Build full prompt with retrieved context."""
        context = self.retrieve(query)
        if not context:
            return query

        return f"""You are an EDA (Electronic Design Automation) assistant.
Use the following documentation excerpts to answer the question.
If the answer is not in the documentation, say so clearly.

DOCUMENTATION:
{context}

QUESTION: {query}

ANSWER:"""
