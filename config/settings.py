"""
MAKERS Configuration Module

This module defines the central configuration for the MAKERS system using Pydantic settings.
It handles:
- Environment variable loading
- Default configuration values
- Configuration validation
- Provider-specific settings (LLM, Embeddings, etc.)

Configuration is loaded from:
1. Environment variables
2. .env file
3. Default values (as fallback)
"""

from typing import List, Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Centralized application settings.
    
    Settings are loaded from environment variables and/or a .env file.
    All settings can be overridden by environment variables using the same name.
    """
    
    # --- General Project Settings ---
    PROJECT_NAME: str = "MAKERS: Multi Agent Knowledge Exploration & Retrieval System"
    DEBUG: bool = False
    PYTHON_ENV: str = "development"  # development, staging, production

    # --- API Keys & Authentication ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    # --- SQLite Configuration (for LangGraph checkpoints) ---
    SQLITE_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"

    # --- ChromaDB Configuration (for vector storage) ---
    CHROMA_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "arxiv_chunks"

    # --- LLM Provider Configuration ---
    DEFAULT_LLM_MODEL_PROVIDER: str = "groq"  # Groq for unlimited/free tier usage
    DEFAULT_OPENAI_GENERATIVE_MODEL: str = "gpt-4"
    HUGGINGFACE_REPO_ID: Optional[str] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    OLLAMA_GENERATIVE_MODEL_NAME: Optional[str] = "mistral"
    GROQ_MODEL_NAME: Optional[str] = "llama-3.3-70b-versatile"  # Default Groq model
    GOOGLE_GEMINI_MODEL_NAME: Optional[str] = "gemini-pro"  # Default Gemini model

    # --- Embedding Configuration ---
    DEFAULT_EMBEDDING_PROVIDER: str = "huggingface"  # Local, unlimited, free
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSION: int = 1536
    
    # HuggingFace Embeddings (local, unlimited, free)
    # Models run locally on your machine - no API limits or costs
    HUGGINGFACE_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_EMBEDDING_MODEL_DIMENSION: int = 384
    
    # Ollama Embeddings
    OLLAMA_EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_MODEL_DIMENSION: int = 768

    # --- Data Processing Configuration ---
    CHUNK_SIZE: int = 512  # Optimal for embedding models: better semantic precision, less information dilution
    CHUNK_OVERLAP: int = 100  # ~20% overlap for context preservation between chunks

    # --- Core Workflow Configuration ---
    # Summary and Memory Management
    SUMMARY_THRESHOLD: int = 15  # Number of messages before triggering summarization (more frequent = better memory management)
    MESSAGES_TO_KEEP_AFTER_SUMMARY: int = 3  # Number of recent messages to keep after summarization
    
    # Workflow Safety
    MAX_ITERATIONS: int = 100  # Maximum iterations to prevent infinite loops (higher for complex workflows)
    
    # LLM Temperature Settings
    AGENT_TEMPERATURE: float = 0.3  # Main agent: balanced creativity and accuracy
    SUMMARY_LLM_TEMPERATURE: float = 0.1  # Summary agent: low temperature for factual consistency
    DOCUMENT_ANALYST_TEMPERATURE: float = 0.2  # Document analyst: low temperature for analytical precision
    DOCUMENT_SYNTHESIZER_TEMPERATURE: float = 0.4  # Document synthesizer: higher temperature for creative synthesis
    
    # RAG Configuration
    RAG_TOP_K: int = 5  # Number of top results to retrieve from knowledge base
    RAG_SIMILARITY_THRESHOLD: float = 0.0  # Minimum similarity score threshold (0.0 = no threshold)

    # --- ArXiv Configuration ---
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    ARXIV_DEFAULT_QUERY: str = "What are the latest advancements in face analysis"
    ARXIV_MAX_RESULTS: int = 10
    ARXIV_SORT_BY: str = "SubmittedDate"
    ARXIV_SORT_ORDER: str = "Descending"
    ARXIV_DOWNLOAD_DELAY_SECONDS: int = 3

    # --- Evaluation Configuration ---
    EVALUATION_DATASET_PATH: Optional[str] = str(DATA_DIR / "evaluation/rag_eval_dataset.json")

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]  # Override in production with specific origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create global settings instance
settings = Settings()

if __name__ == "__main__":
    """Print current configuration when run directly."""
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")

    print("\n--- Generative LLM Configuration ---")
    print(f"Default Generative LLM Provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"  OpenAI Model: {settings.DEFAULT_OPENAI_GENERATIVE_MODEL}")
    print(f"  HuggingFace Repo ID: {settings.HUGGINGFACE_REPO_ID}")
    print(f"  Ollama Model: {settings.OLLAMA_GENERATIVE_MODEL_NAME}")
    print(f"  Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    print(f"  Groq Model: {settings.GROQ_MODEL_NAME}")
    print(f"  Google Gemini Model: {settings.GOOGLE_GEMINI_MODEL_NAME}")

    print("\n--- Embedding Configuration ---")
    print(f"Default Embedding Provider: {settings.DEFAULT_EMBEDDING_PROVIDER}")
    print(f"  OpenAI Model: {settings.OPENAI_EMBEDDING_MODEL_NAME}")
    print(f"  OpenAI Dimension: {settings.OPENAI_EMBEDDING_DIMENSION}")
    print(f"  HuggingFace Model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
    print(f"  HuggingFace Dimension: {settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION}")
    print(f"  Ollama Model: {settings.OLLAMA_EMBEDDING_MODEL_NAME}")
    print(f"  Ollama Dimension: {settings.OLLAMA_EMBEDDING_MODEL_DIMENSION}")

    print("\n--- API Keys (Presence) ---")
    print(f"OpenAI API Key: {'✓' if settings.OPENAI_API_KEY else '✗'}")
    print(f"HuggingFace API Key: {'✓' if settings.HUGGINGFACE_API_KEY else '✗'}")
    print(f"Anthropic API Key: {'✓' if settings.ANTHROPIC_API_KEY else '✗'}")
    print(f"Groq API Key: {'✓' if settings.GROQ_API_KEY else '✗'}")
    print(f"Google API Key: {'✓' if settings.GOOGLE_API_KEY else '✗'}")
    print(f"Tavily API Key: {'✓' if settings.TAVILY_API_KEY else '✗'}")
    print(f"Weights & Biases API Key: {'✓' if settings.WANDB_API_KEY else '✗'}")

    print("\n--- ChromaDB Configuration (Vector Storage) ---")
    print(f"ChromaDB Path: {settings.CHROMA_DB_PATH}")
    print(f"Collection Name: {settings.CHROMA_COLLECTION_NAME}")
    
    print("\n--- SQLite Configuration (Checkpoints) ---")
    print(f"SQLite DB Path: {settings.SQLITE_DB_PATH}")

    print("\n--- Data & Paths ---")
    print(f"Data Directory: {settings.DATA_DIR}")
    print(f"Evaluation Dataset: {settings.EVALUATION_DATASET_PATH}")
    
    print("\n--- Core Workflow Configuration ---")
    print(f"Summary Threshold: {settings.SUMMARY_THRESHOLD} messages")
    print(f"Messages to Keep After Summary: {settings.MESSAGES_TO_KEEP_AFTER_SUMMARY}")
    print(f"Max Iterations: {settings.MAX_ITERATIONS}")
    print(f"Agent Temperature: {settings.AGENT_TEMPERATURE}")
    print(f"Summary LLM Temperature: {settings.SUMMARY_LLM_TEMPERATURE}")
    print(f"Document Analyst Temperature: {settings.DOCUMENT_ANALYST_TEMPERATURE}")
    print(f"Document Synthesizer Temperature: {settings.DOCUMENT_SYNTHESIZER_TEMPERATURE}")
    print(f"RAG Top K: {settings.RAG_TOP_K}")
    print(f"RAG Similarity Threshold: {settings.RAG_SIMILARITY_THRESHOLD}")
    
    print("\n--- Data Processing Configuration ---")
    print(f"Chunk Size: {settings.CHUNK_SIZE} tokens")
    print(f"Chunk Overlap: {settings.CHUNK_OVERLAP} tokens")