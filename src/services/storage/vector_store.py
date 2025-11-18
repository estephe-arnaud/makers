"""
Vector Store Module

Provides the RetrievalEngine for vector similarity search with ChromaDB.
Handles document insertion and retrieval using LlamaIndex and ChromaDB.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LlamaHuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaOllamaEmbedding

from config.settings import settings

logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry logger to suppress PostHog errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

class RetrievedNode:
    """Represents a retrieved document node with its metadata and relevance score."""
    
    def __init__(self, text: str, score: Optional[float], metadata: Dict[str, Any]):
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"RetrievedNode(score={self.score:.4f}, text='{self.text[:100]}...', metadata={self.metadata})"


class RetrievalEngine:
    """Engine for retrieving relevant documents using vector similarity search with ChromaDB."""
    
    DEFAULT_COLLECTION_NAME = "arxiv_chunks"

    def __init__(
        self,
        chroma_db_path: Optional[Path] = None,
        collection_name: str = None,
    ):
        self.chroma_db_path = chroma_db_path or settings.CHROMA_DB_PATH
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # Ensure ChromaDB directory exists
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)

        self._vector_store: Optional[ChromaVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[BaseRetriever] = None
        self._chroma_collection = None  # Store ChromaDB collection for direct access

        self._setup_llamaindex_components()
        logger.info("RetrievalEngine initialized with ChromaDB and LlamaIndex components")

    @staticmethod
    def configure_embedding_model() -> None:
        """Configure the global embedding model based on the selected provider.
        
        This is a static method that can be called independently to configure
        LlamaIndex embeddings without creating a RetrievalEngine instance.
        """
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        logger.info(f"Configuring LlamaIndex embed_model for provider: {provider}")

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is missing")

            model_kwargs = {}
            if settings.OPENAI_EMBEDDING_MODEL_NAME in ["text-embedding-3-small", "text-embedding-3-large"]:
                native_max_dim = 1536 if settings.OPENAI_EMBEDDING_MODEL_NAME == "text-embedding-3-small" else 3072
                if settings.OPENAI_EMBEDDING_DIMENSION < native_max_dim:
                    model_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSION

            LlamaSettings.embed_model = LlamaOpenAIEmbedding(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL_NAME,
                **model_kwargs
            )
            logger.info(f"Configured OpenAI embedding model: {settings.OPENAI_EMBEDDING_MODEL_NAME}")

        elif provider == "huggingface":
            LlamaSettings.embed_model = LlamaHuggingFaceEmbedding(
                model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
            )
            logger.info(f"Configured HuggingFace embedding model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")

        elif provider == "ollama":
            if not settings.OLLAMA_BASE_URL:
                raise ValueError("Ollama base URL is missing")

            LlamaSettings.embed_model = LlamaOllamaEmbedding(
                model_name=settings.OLLAMA_EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL
            )
            logger.info(f"Configured Ollama embedding model: {settings.OLLAMA_EMBEDDING_MODEL_NAME}")

        else:
            raise NotImplementedError(f"Unsupported embedding provider: {provider}")

    def _setup_llamaindex_components(self) -> None:
        """Initialize LlamaIndex components for vector search with ChromaDB."""
        try:
            self.configure_embedding_model()

            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
            
            # Get or create collection
            chroma_collection = chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Store collection for direct access (bypassing LlamaIndex filter issues)
            self._chroma_collection = chroma_collection

            # Create ChromaDB vector store
            self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            logger.info(f"Configured ChromaDB vector store at: {self.chroma_db_path}")
            logger.info(f"Using collection: {self.collection_name}")

            if LlamaSettings.embed_model is None:
                raise ValueError("Embedding model not configured")

            self._index = VectorStoreIndex.from_vector_store(self._vector_store)
            self._retriever = self._index.as_retriever(similarity_top_k=5)
            logger.info("Initialized vector store index and retriever with ChromaDB")

        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex components: {e}", exc_info=True)
            self._vector_store = None
            self._index = None
            self._retriever = None
            raise

    def retrieve_simple_vector_search(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filters: Optional[List[Dict[str, Any]]] = None  # Deprecated: kept for compatibility but ignored
    ) -> List[RetrievedNode]:
        """Retrieve relevant documents using vector similarity search."""
        if not self._index or not self._retriever:
            logger.error("RetrievalEngine not properly initialized")
            if not self._index:
                try:
                    self._setup_llamaindex_components()
                except Exception:
                    return []
            if not self._retriever:
                logger.error("Retriever not available after re-initialization")
                return []

        try:
            # Use ChromaDB directly to avoid LlamaIndex filter issues
            # Embed the query text first
            if LlamaSettings.embed_model is None:
                raise ValueError("Embedding model not configured")
            
            # Get query embedding - try get_query_embedding first, fallback to get_text_embedding
            if hasattr(LlamaSettings.embed_model, 'get_query_embedding'):
                query_embedding = LlamaSettings.embed_model.get_query_embedding(query_text)
            elif hasattr(LlamaSettings.embed_model, 'get_text_embedding'):
                query_embedding = LlamaSettings.embed_model.get_text_embedding(query_text)
            else:
                # Fallback: use get_text_embedding_batch
                embeddings = LlamaSettings.embed_model.get_text_embedding_batch([query_text])
                query_embedding = embeddings[0] if embeddings else None
                if query_embedding is None:
                    raise ValueError("Failed to generate query embedding")
            
            # Query ChromaDB directly (bypassing LlamaIndex to avoid empty filter issues)
            # ChromaDB query with query_embeddings (not query_texts) doesn't require filters
            if self._chroma_collection is None:
                # Fallback: reinitialize if needed
                self._setup_llamaindex_components()
            
            chroma_results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                # Don't pass where/filters parameter at all - ChromaDB will handle it correctly
            )
            
            # Convert ChromaDB results to RetrievedNode format
            results = []
            if chroma_results and chroma_results.get('ids') and len(chroma_results['ids']) > 0:
                ids = chroma_results['ids'][0]
                documents = chroma_results.get('documents', [[]])[0] if chroma_results.get('documents') else []
                metadatas = chroma_results.get('metadatas', [[]])[0] if chroma_results.get('metadatas') else []
                distances = chroma_results.get('distances', [[]])[0] if chroma_results.get('distances') else []
                
                for i, doc_id in enumerate(ids):
                    # Convert distance to similarity score (cosine distance -> similarity)
                    # ChromaDB returns distances (lower is better), convert to similarity (higher is better)
                    distance = distances[i] if i < len(distances) else None
                    score = 1.0 - distance if distance is not None else None
                    
                    # Get document text and metadata
                    doc_text = documents[i] if i < len(documents) else ""
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    results.append(
                        RetrievedNode(
                            text=doc_text,
                            score=score,
                            metadata=metadata
                        )
                    )

            logger.info(f"Retrieved {len(results)} nodes for query: '{query_text[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    @staticmethod
    def get_embedding_model_name() -> str:
        """Get the embedding model name based on the configured provider."""
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            return settings.OPENAI_EMBEDDING_MODEL_NAME
        elif provider == "huggingface":
            return settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        elif provider == "ollama":
            return settings.OLLAMA_EMBEDDING_MODEL_NAME
        else:
            logger.warning(f"Unknown embedding provider: {provider}, using 'unknown' as model name")
            return "unknown"

    def insert_documents(self, documents: List[Document], show_progress: bool = False) -> None:
        """Insert documents into the vector store.
        
        Args:
            documents: List of LlamaIndex Document objects to insert
            show_progress: Whether to show a progress bar (requires tqdm)
        """
        if not documents:
            logger.warning("No documents provided for insertion")
            return
        
        if not self._index:
            logger.error("RetrievalEngine index not initialized")
            raise ValueError("RetrievalEngine index not properly initialized")
        
        logger.info(f"Inserting {len(documents)} document(s) into collection: {self.collection_name}")
        
        if show_progress:
            try:
                from tqdm import tqdm
                with tqdm(total=len(documents), desc="Inserting", unit="doc", leave=False) as pbar:
                    for doc in documents:
                        self._index.insert(document=doc)
                        pbar.update(1)
            except ImportError:
                logger.warning("tqdm not available, inserting without progress bar")
                for doc in documents:
                    self._index.insert(document=doc)
        else:
            for doc in documents:
                self._index.insert(document=doc)
        
        logger.info(f"Successfully inserted {len(documents)} document(s) into ChromaDB")