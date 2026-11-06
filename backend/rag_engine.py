"""
Lightweight RAG engine optimized for API documentation
Uses latest efficient embedding models and vector stores
Enhanced with auto-training and knowledge base integration
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import hashlib
from datetime import datetime

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
    Enhanced with auto-training and caching
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    # bge cosine scores are compressed (~0.5-0.9). On the BaNCS spec, good top hits
    # scored >= 0.71, weak ones 0.65-0.68 (keyword search did better on those) and
    # nonsense ~0.54. Below this, callers fall back to keyword search.
    # ponytail: tuned on 8 sample queries; retune with real query logs or a new model.
    MIN_SCORE = 0.7
    INDEX_VERSION = 2  # bump when document construction changes, to rebuild the cache

    def __init__(self, 
                 knowledge_base=None,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 auto_train: bool = True):
        self.kb = knowledge_base
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.documents: List[Document] = []
        # ponytail: plain float32 matrix (~3k x 384, a few MB). int8/Matryoshka tricks
        # removed - they saved nothing here and the int8 dot product overflowed.
        self.embeddings_matrix = None

        # Caching paths
        self.cache_dir = Path("models/rag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "rag_index.pkl"
        self.hash_path = self.cache_dir / "kb_hash.txt"
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Auto-train if knowledge base provided
        if auto_train and self.kb:
            self._auto_train_if_needed()
    
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
            
            # Choose the best available model
            model_to_use = self.embedding_model_name
            if model_to_use not in small_models:
                logger.warning(f"Unknown model {model_to_use}, using default")
                model_to_use = "BAAI/bge-small-en-v1.5"
            
            self.embedding_model = SentenceTransformer(model_to_use)
            self.embedding_model_name = model_to_use

            # FP16 only helps on GPU; on CPU it is slower and loses precision
            if self.embedding_model.device.type == "cuda":
                self.embedding_model.half()

            logger.info(f"Loaded embedding model: {model_to_use} ({small_models.get(model_to_use, 'Unknown')}M params)")
            
        except ImportError:
            logger.warning("sentence-transformers not installed, using TF-IDF fallback")
            self._use_numpy_embeddings = True
            self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF as fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            use_idf=True,
            sublinear_tf=True
        )
    
    def _get_kb_hash(self) -> str:
        """Generate hash of knowledge base for cache invalidation"""
        if not self.kb or not hasattr(self.kb, 'endpoints'):
            return "empty"
        
        # Create deterministic hash of KB content + how it was embedded,
        # so switching model (or TF-IDF <-> embeddings) rebuilds the cache
        kb_summary = {
            'model': self.embedding_model_name if self.embedding_model else 'tfidf',
            'version': self.INDEX_VERSION,
            'count': len(self.kb.endpoints),
            'endpoints': sorted([
                (name, data.get('url', ''), data.get('method', ''))
                for name, data in self.kb.endpoints.items()
            ])
        }
        
        kb_str = json.dumps(kb_summary, sort_keys=True)
        return hashlib.sha256(kb_str.encode()).hexdigest()[:16]
    
    def _needs_training(self) -> bool:
        """Check if retraining is needed"""
        if not self.index_path.exists() or not self.hash_path.exists():
            return True
        
        try:
            with open(self.hash_path, 'r') as f:
                stored_hash = f.read().strip()
            
            current_hash = self._get_kb_hash()
            needs_update = stored_hash != current_hash
            
            if needs_update:
                logger.info(f"RAG: KB changed (hash: {stored_hash[:8]}... -> {current_hash[:8]}...)")
            
            return needs_update
        except Exception as e:
            logger.error(f"RAG: Error checking hash: {e}")
            return True
    
    def _auto_train_if_needed(self):
        """Auto-train if KB has changed"""
        if self._needs_training():
            start_time = datetime.now()
            logger.info(f"RAG: Training on {len(self.kb.endpoints)} endpoints...")
            
            # Clear existing documents
            self.documents = []
            
            # Add documents from KB
            self.add_documents_from_apis(self.kb.endpoints)
            
            # Save index and hash
            self.save_index(str(self.index_path))
            with open(self.hash_path, 'w') as f:
                f.write(self._get_kb_hash())
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"RAG: Training completed in {elapsed:.2f}s")
        else:
            # Load existing index
            if self.index_path.exists():
                self.load_index(str(self.index_path))
                logger.info(f"RAG: Loaded cached index with {len(self.documents)} documents")
    
    def add_documents_from_apis(self, api_endpoints: Dict[str, Any]):
        """Add one document per API endpoint.

        ponytail: one chunk per endpoint. Measured on the BaNCS spec, the extra
        parameter/keyword chunks ranked worse than the main chunk alone and the
        parameter chunks tripled CPU build time. The main chunk lists the first 5 params.
        """
        for endpoint_name, endpoint in api_endpoints.items():
            self.documents.append(Document(
                content=self._create_api_document(endpoint_name, endpoint),
                metadata={
                    "endpoint_name": endpoint_name,
                    "method": endpoint.get("method"),
                    "path": endpoint.get("url"),
                    "category": endpoint.get("category"),
                    "tags": endpoint.get("tags", []),
                    "chunk_type": "main"
                },
                chunk_id=f"api_{endpoint_name}_main"
            ))

        # Generate embeddings for all documents
        self._generate_embeddings()
        logger.info(f"Added {len(self.documents)} document chunks from {len(api_endpoints)} APIs")
    
    def _create_api_document(self, name: str, endpoint: Dict[str, Any]) -> str:
        """Create searchable document from API endpoint"""
        import re
        
        parts = []
        
        # Tokenize name properly
        name_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', name)
        parts.append(f"API Endpoint: {' '.join(name_tokens)}")
        
        parts.append(f"Method: {endpoint.get('method', 'GET')}")
        parts.append(f"Path: {endpoint.get('url', '')}")
        parts.append(f"Description: {endpoint.get('description', '')}")
        parts.append(f"Category: {endpoint.get('category', 'general')}")
        
        # Add parameters with details
        if endpoint.get("parameters"):
            if isinstance(endpoint["parameters"], dict):
                param_strs = []
                for param_name, param_info in endpoint["parameters"].items():
                    if isinstance(param_info, dict):
                        param_str = f"{param_name} ({param_info.get('type', 'string')}): {param_info.get('description', '')}"
                        param_strs.append(param_str)
                if param_strs:
                    parts.append(f"Parameters: {', '.join(param_strs[:5])}")  # Limit to 5 for main doc
        
        # Add authentication info
        parts.append("Authentication: Required headers - entity, userId, languageCode")
        
        return "\n".join(parts)
    
    def _generate_embeddings(self):
        """Generate embeddings for all documents"""
        if hasattr(self, '_use_numpy_embeddings') and self._use_numpy_embeddings:
            # Use TF-IDF fallback
            self._generate_tfidf_embeddings()
            return
        
        if not self.embedding_model:
            # Fallback to random embeddings for testing
            for doc in self.documents:
                doc.embedding = np.random.randn(384).astype(np.float32)
            self.embeddings_matrix = np.vstack([doc.embedding for doc in self.documents])
            return
        
        # Batch encode all documents
        texts = [doc.content for doc in self.documents]
        
        # Process in batches for memory efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Store embeddings
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
        
        self.embeddings_matrix = embeddings
    
    def _generate_tfidf_embeddings(self):
        """Generate TF-IDF embeddings as fallback"""
        texts = [doc.content for doc in self.documents]
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Convert to dense and store
        embeddings = tfidf_matrix.toarray().astype(np.float32)
        
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
        
        self.embeddings_matrix = embeddings
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_category: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for relevant documents.

        Returns only matches at or above MIN_SCORE, so weak queries come back empty
        and callers can fall back to keyword search.
        """

        if hasattr(self, '_use_numpy_embeddings') and self._use_numpy_embeddings:
            # Use TF-IDF search
            return self._tfidf_search(query, top_k, filter_category)
        
        if not self.embedding_model:
            # Fallback to keyword search
            return self._keyword_search(query, top_k, filter_category)
        
        # bge models expect this instruction on short retrieval queries (not on documents)
        query_embedding = self.embedding_model.encode(
            self.QUERY_PREFIX + query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Embeddings are L2-normalized, so a dot product is the cosine similarity
        similarities = self.embeddings_matrix @ query_embedding
        
        # Apply category filter if specified
        if filter_category:
            for i, doc in enumerate(self.documents):
                if doc.metadata.get("category") != filter_category:
                    similarities[i] = -1
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], float(similarities[i]))
                for i in top_indices if similarities[i] >= self.MIN_SCORE]
    
    def _tfidf_search(self, query: str, top_k: int, filter_category: Optional[str]) -> List[Tuple[Document, float]]:
        """TF-IDF based search"""
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = (self.embeddings_matrix @ query_vec.T).toarray().flatten()
        
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
    
    def _keyword_search(self, query: str, top_k: int, filter_category: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Fallback keyword search"""
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
            "model_name": self.embedding_model_name,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved RAG index to {path} ({len(self.documents)} documents)")
    
    def load_index(self, path: str):
        """Load RAG index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings_matrix = data["embeddings_matrix"]

        logger.info(f"Loaded RAG index from {path} ({len(self.documents)} documents)")
