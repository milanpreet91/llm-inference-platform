import logging
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("cache.semantic")

SIMILARITY_THRESHOLD = 0.92   # cosine similarity — tune this
COLLECTION_NAME = "semantic_cache"
SEMANTIC_CACHE_TTL = 100      # max docs kept in collection


class SemanticCache:
    def __init__(self, chroma_host: str, chroma_port: int):
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("SemanticCache initialised")

    def get(self, query: str) -> Optional[dict]:
        """Returns cached answer if a semantically similar query exists."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            if not results["ids"][0]:
                return None

            distance = results["distances"][0][0]
            similarity = 1 - distance          # cosine distance → similarity

            if similarity >= SIMILARITY_THRESHOLD:
                metadata = results["metadatas"][0][0]
                logger.info(
                    f"Semantic cache HIT (similarity={similarity:.3f}) "
                    f"for query: {query[:60]}"
                )
                return {"answer": metadata["answer"], "model": metadata["model"]}

            logger.info(f"Semantic cache MISS (similarity={similarity:.3f})")
        except Exception as e:
            logger.warning(f"Semantic cache GET failed: {e}")
        return None

