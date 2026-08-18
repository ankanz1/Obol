from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import logging
import uuid
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant connections and collections for all 18 languages."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
            check_compatibility=False,  # Skip version check
        )
        self.collection_prefix = "msmarco_"
        self.embedding_dim = settings.embedding_dim
    
    def get_collection_name(self, language: str) -> str:
        """Get collection name for a language."""
        return f"{self.collection_prefix}{language}"
    
    def create_all_collections(self, recreate: bool = False):
        """Create collections for all supported languages."""
        for lang in settings.supported_languages:
            collection_name = self.get_collection_name(lang)
            try:
                if recreate:
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
            except:
                pass
            
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection exists: {collection_name}")
            except:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=128,
                        full_scan_threshold=10000
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,
                        memmap_threshold=50000
                    )
                )
                logger.info(f"Created collection: {collection_name}")
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], language: str, batch_size: int = 100):
        """Upsert chunks to language-specific collection."""
        collection_name = self.get_collection_name(language)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                # Ensure we have embedding
                if "vector" not in chunk:
                    logger.warning(f"Chunk {chunk.get('id')} missing vector, skipping")
                    continue
                
                payload = {k: v for k, v in chunk.items() if k not in ["id", "vector"]}
                payload["content"] = chunk.get("content", "")
                
                points.append(PointStruct(
                    id=chunk["id"],
                    vector=chunk["vector"].tolist() if hasattr(chunk["vector"], "tolist") else chunk["vector"],
                    payload=payload
                ))
            
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                logger.info(f"Upserted {len(points)} points to {collection_name}")
    
    def search(
        self,
        query_vector: List[float],
        language: str,
        limit: int = 5,
        score_threshold: float = None,
        query_filter: models.Filter = None
    ) -> List[models.ScoredPoint]:
        """Search in language-specific collection."""
        collection_name = self.get_collection_name(language)
        
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold or settings.retrieval_score_threshold,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            with_vectors=False
        )
        
        return results.points
    
    def search_multi_language(
        self,
        query_vector: List[float],
        languages: List[str],
        limit: int = 5,
        score_threshold: float = None
    ) -> Dict[str, List[models.ScoredPoint]]:
        """Search across multiple language collections."""
        results = {}
        for lang in languages:
            try:
                results[lang] = self.search(query_vector, lang, limit, score_threshold)
            except Exception as e:
                logger.warning(f"Search failed for {lang}: {e}")
                results[lang] = []
        return results
    
    def get_collection_stats(self, language: str) -> Dict[str, Any]:
        """Get collection statistics."""
        collection_name = self.get_collection_name(language)
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": getattr(info, 'vectors_count', 'N/A'),
                "points_count": getattr(info, 'points_count', 'N/A'),
                "status": getattr(info, 'status', 'N/A')
            }
        except Exception as e:
            return {"name": collection_name, "error": str(e)}
    
    def close(self):
        self.client.close()


# Global instance
_qdrant = None

def get_qdrant() -> QdrantManager:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantManager()
    return _qdrant