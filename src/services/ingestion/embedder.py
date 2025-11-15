"""
Embedder Module

This module provides functionality for generating embeddings from text chunks using various
embedding providers (OpenAI, HuggingFace, Ollama). It handles the embedding generation process,
metadata management, and batch processing of text chunks.

Key features:
- Multiple embedding provider support
- Batch processing with configurable batch size
- Comprehensive metadata management
- Robust error handling and logging

Provider Notes:
- HuggingFace: Runs locally - UNLIMITED and FREE (model downloaded once, cached locally)
- Ollama: Runs locally - UNLIMITED and FREE (requires Ollama installed)
- OpenAI: Uses API - has rate limits and costs per token

Recommendation: Use HuggingFace or Ollama for unlimited, free embeddings.
"""

import logging
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

from config.settings import settings
from src.services.ingestion.preprocessor import ProcessedChunk 

logger = logging.getLogger(__name__)

# Provider to model name mapping
_PROVIDER_MODEL_MAP = {
    "openai": settings.OPENAI_EMBEDDING_MODEL_NAME,
    "huggingface": settings.HUGGINGFACE_EMBEDDING_MODEL_NAME,
    "ollama": settings.OLLAMA_EMBEDDING_MODEL_NAME,
}

def _get_model_name(provider: str) -> str:
    """Get the model name for a given provider."""
    return _PROVIDER_MODEL_MAP.get(provider, "")

def get_embedding_client() -> Any:
    """
    Initialize and return an embedding client based on the configured provider.
    
    Returns:
        An embedding client instance (OpenAIEmbeddings, HuggingFaceEmbeddings, or OllamaEmbeddings)
        
    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    logger.info(f"Initializing embedding client for provider: {provider}")

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured in settings for embeddings.")
            raise ValueError("OpenAI API key is missing for OpenAI embeddings.")
        
        model_kwargs = {}
        if settings.OPENAI_EMBEDDING_MODEL_NAME in ["text-embedding-3-small", "text-embedding-3-large"]:
            native_max_dim = 1536 if settings.OPENAI_EMBEDDING_MODEL_NAME == "text-embedding-3-small" else 3072
            if settings.OPENAI_EMBEDDING_DIMENSION < native_max_dim:
                model_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSION
        
        logger.info(
            f"Using OpenAIEmbeddings with model: {settings.OPENAI_EMBEDDING_MODEL_NAME}, "
            f"dimension: {settings.OPENAI_EMBEDDING_DIMENSION}"
        )
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL_NAME,
            **model_kwargs
        )
    elif provider == "huggingface":
        # HuggingFace embeddings run locally - unlimited and free!
        # The model is downloaded once and cached, then runs on your machine
        logger.info(
            f"Using HuggingFaceEmbeddings (local, unlimited) with model: "
            f"{settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}"
        )
        return HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        )
    
    if provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            logger.error("Ollama base URL is not configured for embeddings.")
            raise ValueError("Ollama base URL is missing for Ollama embeddings.")
        logger.info(f"Using OllamaEmbeddings with model: {settings.OLLAMA_EMBEDDING_MODEL_NAME} via {settings.OLLAMA_BASE_URL}")
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBEDDING_MODEL_NAME
        )
    else:
        logger.error(f"Unsupported embedding provider: {provider}")
        raise ValueError(f"Unsupported embedding provider: {provider}")

def generate_embeddings_for_chunks(
    processed_chunks: List[ProcessedChunk],
    batch_size: int = 32 
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of processed text chunks.
    
    Args:
        processed_chunks: List of processed text chunks to embed
        batch_size: Number of chunks to process in each batch
        
    Returns:
        List of dictionaries containing the embedded chunks with their metadata
    """
    if not processed_chunks:
        logger.warning("No processed chunks provided for embedding.")
        return []

    logger.info(f"Starting embedding generation for {len(processed_chunks)} chunks.")
    
    embedding_client = get_embedding_client()
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    model_name = _get_model_name(provider)
    
    if not model_name:
        logger.error(f"Unknown embedding provider '{provider}' in generate_embeddings_for_chunks.")
        return []
    
    texts_to_embed: List[str] = [chunk["text_chunk"] for chunk in processed_chunks]
    final_chunks_for_db: List[Dict[str, Any]] = []
    total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[batch_idx:batch_idx + batch_size]
        batch_original_chunks = processed_chunks[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(
            f"Embedding batch {batch_num}/{total_batches} "
            f"(size: {len(batch_texts)}) using {provider} provider."
        )
        
        try:
            embeddings = embedding_client.embed_documents(batch_texts)
            
            if len(embeddings) != len(batch_texts):
                logger.error(
                    f"Mismatch in number of embeddings ({len(embeddings)}) "
                    f"and texts ({len(batch_texts)}) in batch {batch_num}."
                )
                continue

            # Process batch results
            for original_chunk, embedding_vector in zip(batch_original_chunks, embeddings):
                # Build base metadata (priority fields)
                metadata = {
                    "chunk_id": original_chunk['chunk_id'],
                    "arxiv_id": original_chunk['arxiv_id'],
                    "original_document_title": original_chunk.get('original_document_title'),
                    "embedding_model": model_name,
                    "embedding_provider": provider,
                    "embedding_dimension": len(embedding_vector) if embedding_vector else 0
                }
                
                # Merge source metadata (preserve priority fields, map title if needed)
                source_meta = original_chunk.get("source_document_metadata")
                if isinstance(source_meta, dict):
                    # Add all source metadata except conflicting keys
                    for key, value in source_meta.items():
                        if key not in metadata:
                            metadata[key] = value
                        elif key == "title" and not metadata.get("original_document_title"):
                            metadata["original_document_title"] = value

                final_chunks_for_db.append({
                    "chunk_id": original_chunk['chunk_id'],
                    "text_chunk": original_chunk['text_chunk'],
                    "embedding": embedding_vector,
                    "metadata": metadata
                })

        except Exception as e:
            logger.error(
                f"Error embedding batch {batch_num} with {provider}: {e}",
                exc_info=True
            )
        
    logger.info(f"Finished embedding generation. Successfully structured {len(final_chunks_for_db)} chunks for DB out of {len(processed_chunks)}.")
    return final_chunks_for_db