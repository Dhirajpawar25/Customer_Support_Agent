"""Local embeddings and retrieval."""
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from src.models import DocumentChunk, RetrievalResult
from src.config import config


class EmbeddingStore:
    """Simple in-memory embedding store with persistence."""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.array([])
        self._index_path = config.project_root / ".embeddings_cache.json"
    
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build embedding index from chunks."""
        self.chunks = chunks
        if not chunks:
            self.embeddings = np.array([])
            return
        
        if self._load_from_cache(chunks):
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        self.embeddings = np.array(embeddings)
        self._save_to_cache()
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._local_embedding(text) for text in texts]

    @staticmethod
    def _local_embedding(text: str, dimensions: int = 512) -> List[float]:
        vector = np.zeros(dimensions, dtype=float)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            vector[hash(token) % dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()
    
    def _save_to_cache(self) -> None:
        if len(self.chunks) == 0:
            return
        
        cache_data = {
            "chunks": [
                {
                    "content": c.content,
                    "metadata": {
                        "document_id": c.metadata.document_id,
                        "title": c.metadata.title,
                        "status": c.metadata.status.value,
                        "effective_date": str(c.metadata.effective_date) if c.metadata.effective_date else None,
                        "last_reviewed": str(c.metadata.last_reviewed) if c.metadata.last_reviewed else None,
                        "audience": c.metadata.audience,
                        "policy_authority": c.metadata.policy_authority.value,
                        "supersedes": c.metadata.supersedes,
                        "source_file": c.metadata.source_file
                    },
                    "chunk_index": c.chunk_index,
                    "heading": c.heading,
                    "heading_level": c.heading_level
                }
                for c in self.chunks
            ],
            "embeddings": self.embeddings.tolist()
        }
        self._index_path.write_text(json.dumps(cache_data))
    
    def _load_from_cache(self, chunks: List[DocumentChunk]) -> bool:
        if not self._index_path.exists():
            return False
        
        try:
            cache_data = json.loads(self._index_path.read_text())
            
            if len(cache_data["chunks"]) != len(chunks):
                return False
            
            for i, (cached, current) in enumerate(zip(cache_data["chunks"], chunks)):
                if cached["content"] != current.content:
                    return False
            
            self.chunks = chunks
            self.embeddings = np.array(cache_data["embeddings"])
            return True
        except Exception:
            return False
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._local_embedding(query)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> RetrievalResult:
        if len(self.chunks) == 0 or self.embeddings.size == 0:
            return RetrievalResult(chunks=[], query=query, scores=[])
        
        query_embedding = self._get_query_embedding(query)
        query_vec = np.array(query_embedding)
        
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return RetrievalResult(chunks=[], query=query, scores=[])
        
        similarities = np.dot(self.embeddings, query_vec) / (doc_norms * query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        scores = []
        for idx in top_indices:
            if similarities[idx] >= min_score:
                results.append(self.chunks[idx])
                scores.append(float(similarities[idx]))
        
        return RetrievalResult(chunks=results, query=query, scores=scores)
    
    def search_authoritative(self, query: str, top_k: int = 5, min_score: float = 0.3) -> RetrievalResult:
        auth_chunks = [c for c in self.chunks if c.metadata.is_authoritative and not c.metadata.is_superseded]
        if not auth_chunks:
            return RetrievalResult(chunks=[], query=query, scores=[])
        
        auth_indices = [i for i, c in enumerate(self.chunks) if c in auth_chunks]
        auth_embeddings = self.embeddings[auth_indices]
        
        query_embedding = self._get_query_embedding(query)
        query_vec = np.array(query_embedding)
        
        doc_norms = np.linalg.norm(auth_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return RetrievalResult(chunks=[], query=query, scores=[])
        
        similarities = np.dot(auth_embeddings, query_vec) / (doc_norms * query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        scores = []
        for idx in top_indices:
            if similarities[idx] >= min_score:
                results.append(auth_chunks[idx])
                scores.append(float(similarities[idx]))
        
        return RetrievalResult(chunks=results, query=query, scores=scores)


embedding_store = EmbeddingStore()