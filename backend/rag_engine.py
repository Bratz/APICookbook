# backend/rag_engine.py
"""
Lightweight RAG engine optimized for API documentation
Uses latest efficient embedding models and vector stores
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document chunk for RAG"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""

class LightweightRAG:
    """
    Efficient RAG system using small models
    Implements latest techniques from 2024
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.documents: List[Document] = []
        self.embeddings_matrix = None
        self.use_matryoshka = True  # Matryoshka embeddings for efficiency
        self.use_quantization = True  # Quantize embeddings to int8
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Latest small embedding models (2024)
            small_models = {
                "BAAI/bge-small-en-v1.5": 33,  # 33M params
                "BAAI/bge-micro-v2": 17,  # 17M params
                "thenlper/gte-small": 33,  # 33M params
                "sentence-transformers/all-MiniLM-L6-v2": 22,  # 22M params
                "jinaai/jina-embeddings-v3-small": 33,  # New Jina v3
                "nomic-ai/nomic-embed-text-v1.5": 137,  # Nomic's latest
                "Snowflake/snowflake-arctic-embed-s": 33,  # Snowflake's small
            }
            
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Enable optimizations
            if hasattr(self.embedding_model, "half"):
                self.embedding_model.half()  # FP16 for speed
            
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers not installed, using numpy fallback")
            self._use_numpy_embeddings = True
    
    def add_documents_from_apis(self, api_endpoints: Dict[str, Any]):
        """Add API documentation to RAG"""
        for endpoint_name, endpoint in api_endpoints.items():
            # Create structured document
            content = self._create_api_document(endpoint_name, endpoint)
            
            doc = Document(
                content=content,
                metadata={
                    "endpoint_name": endpoint_name,
                    "method": endpoint.get("method"),
                    "path": endpoint.get("path"),
                    "category": endpoint.get("category"),
                    "tags": endpoint.get("tags", [])
                },
                chunk_id=f"api_{endpoint_name}"
            )
            
            self.documents.append(doc)
        
        # Generate embeddings for all documents
        self._generate_embeddings()
        logger.info(f"Added {len(self.documents)} API documents to RAG")
    
    def _create_api_document(self, name: str, endpoint: Dict[str, Any]) -> str:
        """Create searchable document from API endpoint"""
        parts = [
            f"API: {name}",
            f"Method: {endpoint.get('method', 'GET')}",
            f"Path: {endpoint.get('path', '')}",
            f"Description: {endpoint.get('description', '')}",
            f"Category: {endpoint.get('category', 'general')}"
        ]
        
        # Add parameters
        if endpoint.get("parameters"):
            param_strs = []
            for param_name, param_info in endpoint["parameters"].items():
                param_str = f"{param_name} ({param_info.get('type', 'string')}): {param_info.get('description', '')}"
                param_strs.append(param_str)
            parts.append(f"Parameters: {', '.join(param_strs)}")
        
        # Add examples if available
        if endpoint.get("examples"):
            parts.append(f"Examples available: {', '.join(endpoint['examples'].keys())}")
        
        return "\n".join(parts)
    
    def _generate_embeddings(self):
        """Generate embeddings for all documents"""
        if not self.embedding_model:
            # Fallback to random embeddings for testing
            for doc in self.documents:
                doc.embedding = np.random.randn(384).astype(np.float32)
            self.embeddings_matrix = np.vstack([doc.embedding for doc in self.documents])
            return
        
        # Batch encode all documents
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        # Apply optimizations
        if self.use_matryoshka:
            # Use only first 256 dimensions (Matryoshka representation)
            embeddings = embeddings[:, :256]
        
        if self.use_quantization:
            # Quantize to int8 for memory efficiency
            embeddings = self._quantize_embeddings(embeddings)
        
        # Store embeddings
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
        
        self.embeddings_matrix = embeddings
    
    def _quantize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Quantize embeddings to int8"""
        # Scale to int8 range
        scale = 127.0 / np.max(np.abs(embeddings))
        quantized = (embeddings * scale).astype(np.int8)
        
        # Store scale for dequantization
        self.embedding_scale = scale
        
        return quantized
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_category: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        if not self.embedding_model:
            # Fallback to keyword search
            return self._keyword_search(query, top_k, filter_category)
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Apply same optimizations as documents
        if self.use_matryoshka:
            query_embedding = query_embedding[:256]
        
        if self.use_quantization:
            query_embedding = (query_embedding * self.embedding_scale).astype(np.int8)
        
        # Calculate similarities
        if self.use_quantization:
            # Integer dot product for speed
            similarities = np.dot(self.embeddings_matrix, query_embedding) / (self.embedding_scale ** 2)
        else:
            similarities = np.dot(self.embeddings_matrix, query_embedding)
        
        # Apply category filter if specified
        if filter_category:
            for i, doc in enumerate(self.documents):
                if doc.metadata.get("category") != filter_category:
                    similarities[i] = -1
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return positive similarities
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def _keyword_search(
        self, 
        query: str, 
        top_k: int,
        filter_category: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Fallback keyword search using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create corpus
        corpus = [doc.content for doc in self.documents]
        corpus.append(query)
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate similarities
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        # Apply filter
        if filter_category:
            for i, doc in enumerate(self.documents):
                if doc.metadata.get("category") != filter_category:
                    similarities[i] = 0
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save_index(self, path: str):
        """Save RAG index to disk"""
        data = {
            "documents": self.documents,
            "embeddings_matrix": self.embeddings_matrix,
            "embedding_scale": getattr(self, "embedding_scale", 1.0)
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved RAG index to {path}")
    
    def load_index(self, path: str):
        """Load RAG index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings_matrix = data["embeddings_matrix"]
        self.embedding_scale = data.get("embedding_scale", 1.0)
        
        logger.info(f"Loaded RAG index from {path}")

class HybridSearch:
    """Hybrid search combining dense and sparse retrieval"""
    
    def __init__(self, rag_engine: LightweightRAG):
        self.rag_engine = rag_engine
        self.bm25 = None
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 for sparse retrieval"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents
            tokenized_docs = [
                doc.content.lower().split() 
                for doc in self.rag_engine.documents
            ]
            
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info("Initialized BM25 for hybrid search")
            
        except ImportError:
            logger.warning("rank-bm25 not installed")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining dense and sparse retrieval
        alpha: weight for dense retrieval (1-alpha for sparse)
        """
        # Dense retrieval
        dense_results = self.rag_engine.search(query, top_k * 2)
        
        # Sparse retrieval
        sparse_results = []
        if self.bm25:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            
            top_sparse_indices = np.argsort(sparse_scores)[-top_k*2:][::-1]
            for idx in top_sparse_indices:
                if sparse_scores[idx] > 0:
                    sparse_results.append((
                        self.rag_engine.documents[idx],
                        float(sparse_scores[idx])
                    ))
        
        # Combine and rerank
        combined_scores = {}
        
        # Add dense scores
        for doc, score in dense_results:
            combined_scores[doc.chunk_id] = alpha * score
        
        # Add sparse scores
        for doc, score in sparse_results:
            if doc.chunk_id in combined_scores:
                combined_scores[doc.chunk_id] += (1 - alpha) * (score / 10)  # Normalize BM25
            else:
                combined_scores[doc.chunk_id] = (1 - alpha) * (score / 10)
        
        # Sort and return top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Get documents
        final_results = []
        for chunk_id, score in sorted_results:
            doc = next(d for d in self.rag_engine.documents if d.chunk_id == chunk_id)
            final_results.append((doc, score))
        
        return final_results