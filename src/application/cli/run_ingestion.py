"""Document ingestion pipeline for MAKERS."""
import argparse
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging
from src.services.ingestion.arxiv_downloader import (
    search_arxiv_papers, 
    download_papers as execute_arxiv_downloads
)
from src.services.ingestion.document_parser import parse_document_collection
from src.services.ingestion.preprocessor import preprocess_parsed_documents
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LlamaHuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaOllamaEmbedding
import chromadb
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)



@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    pdf_dir: Optional[Path]
    download_from_arxiv: bool
    query: Optional[str]
    arxiv_keywords: Optional[str]
    max_results: int
    sort_by: str
    sort_order: str
    corpus_name: Optional[str]
    collection_name: str
    vector_index_name: str
    text_index_name: str
    
    @property
    def effective_query(self) -> str:
        """Get the effective query (keywords or query, with fallback to default)."""
        return self.arxiv_keywords or self.query or settings.ARXIV_DEFAULT_QUERY
    
    @property
    def effective_corpus_name(self) -> str:
        """Determine the corpus name from available sources."""
        if self.corpus_name:
            return sanitize_query_for_directory_name(self.corpus_name)
        if self.download_from_arxiv:
            # Use effective_query which includes fallback to default
            return sanitize_query_for_directory_name(self.effective_query)
        if self.pdf_dir:
            # Extract corpus name from path if it's already in a corpus structure
            corpus_name = self._extract_corpus_name_from_path(self.pdf_dir)
            return sanitize_query_for_directory_name(corpus_name)
        return "default_corpus"
    
    def _extract_corpus_name_from_path(self, pdf_dir: Path) -> str:
        """Extract corpus name from PDF directory path.
        
        Examples:
        - data/corpus/face_analysis/pdfs -> face_analysis
        - data/face_analysis -> face_analysis
        - corpus/face_analysis/pdfs -> face_analysis
        """
        # If we're in a "pdfs" subdirectory, use parent name
        if pdf_dir.name == "pdfs" and pdf_dir.parent:
            return pdf_dir.parent.name
        
        # Check if path contains "corpus/{name}" pattern
        try:
            parts = pdf_dir.parts
            if "corpus" in parts:
                idx = parts.index("corpus")
                if idx + 1 < len(parts) and parts[idx + 1] != "pdfs":
                    return parts[idx + 1]
        except (ValueError, IndexError):
            pass
        
        # Default: use directory name
        return pdf_dir.name


def sanitize_query_for_directory_name(query: str) -> str:
    """Convert query to a valid directory name."""
    if not query:
        return "default_corpus"
    name = query.lower()
    name = re.sub(r'[\s\W-]+', '_', name)
    name = name.strip('_')
    return name[:50]


def setup_corpus_directories(corpus_name: str) -> tuple[Path, Path]:
    """Create and return paths for PDF and metadata directories."""
    corpus_path = Path(settings.DATA_DIR) / "corpus" / corpus_name
    pdf_path = corpus_path / "pdfs"
    metadata_path = corpus_path / "metadata"

    pdf_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    return pdf_path, metadata_path

def use_local_pdfs(local_pdf_dir: Path, pdf_path: Path, metadata_path: Path) -> dict:
    """Use local PDF files from the specified directory."""
    if not local_pdf_dir.exists() or not local_pdf_dir.is_dir():
        raise ValueError(f"Directory does not exist: {local_pdf_dir}")
    
    local_pdfs = list(local_pdf_dir.glob("*.pdf"))
    if not local_pdfs:
        logger.warning(f"No PDF files found in {local_pdf_dir}")
        return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}

    logger.info(f"Found {len(local_pdfs)} PDF file(s)")
    
    # Check if source and destination are the same
    try:
        if local_pdf_dir.resolve() == pdf_path.resolve():
            return {"downloaded_count": len(local_pdfs), "failed_count": 0, "skipped_count": 0}
    except Exception:
        pass
    
    # Copy files
    stats = {"copied": 0, "skipped": 0, "failed": 0}
    for pdf_file in local_pdfs:
        dest_pdf = pdf_path / pdf_file.name
        try:
            if pdf_file.resolve() == dest_pdf.resolve() or dest_pdf.exists():
                stats["skipped"] += 1
            else:
                shutil.copy2(pdf_file, dest_pdf)
                stats["copied"] += 1
        except Exception as e:
            logger.error(f"Error copying {pdf_file.name}: {e}")
            stats["failed"] += 1
    
    logger.info(f"Copied: {stats['copied']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
    return {
        "downloaded_count": stats["copied"],
        "failed_count": stats["failed"],
        "skipped_count": stats["skipped"]
    }

def download_papers(config: IngestionConfig, pdf_path: Path, metadata_path: Path, skip_download: bool = False) -> dict:
    """Download papers from ArXiv or use local PDFs."""
    if config.download_from_arxiv:
        if skip_download:
            return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}
        return _download_from_arxiv(config, pdf_path, metadata_path)
    
    if not config.pdf_dir:
        raise ValueError("--pdf_dir is required. Use --download_from_arxiv to download from ArXiv.")
    return use_local_pdfs(config.pdf_dir, pdf_path, metadata_path)


def _download_from_arxiv(config: IngestionConfig, pdf_path: Path, metadata_path: Path) -> dict:
    """Download papers from ArXiv."""
    query = config.effective_query
    logger.info(f"Searching ArXiv: '{query}'")
    
    search_results = search_arxiv_papers(
        query=query,
        max_results=config.max_results,
        sort_by=config.sort_by,
        sort_order=config.sort_order
    )

    if not search_results:
        logger.info("No papers found on ArXiv")
        return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}

    logger.info(f"Found {len(search_results)} paper(s). Downloading...")
    
    stats = execute_arxiv_downloads(
        results=search_results,
        pdf_dir=pdf_path,
        metadata_dir=metadata_path
    )
        
    logger.info(f"Downloaded: {stats['successful']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
    return {
        "downloaded_count": stats['successful'],
        "failed_count": stats['failed'],
        "skipped_count": stats['skipped']
    }

def process_documents(pdf_path: Path, metadata_path: Path) -> list:
    """Parse and preprocess documents into chunks."""
    logger.info("Parsing documents...")
    documents = parse_document_collection(pdf_path, metadata_path)
    if not documents:
        raise ValueError("No documents parsed")

    logger.info(f"Parsed {len(documents)} document(s)")
    logger.info("Preprocessing documents...")
        
    chunks = preprocess_parsed_documents(documents)
    if not chunks:
        raise ValueError("No chunks generated")

    logger.info(f"Generated {len(chunks)} chunk(s)")
    return chunks


def setup_chromadb(config: IngestionConfig, chunks: list) -> None:
    """Set up ChromaDB with chunks (embeddings are generated automatically)."""
    if not chunks:
        raise ValueError("No chunks to store in ChromaDB")
    
    logger.info("Setting up ChromaDB...")
    
    chroma_db_path = settings.CHROMA_DB_PATH
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    _configure_embedding_model()
    
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Get the embedding model name based on the configured provider
    embedding_provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    if embedding_provider == "openai":
        embedding_model_name = settings.OPENAI_EMBEDDING_MODEL_NAME
    elif embedding_provider == "huggingface":
        embedding_model_name = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
    elif embedding_provider == "ollama":
        embedding_model_name = settings.OLLAMA_EMBEDDING_MODEL_NAME
    else:
        embedding_model_name = "unknown"
        logger.warning(f"Unknown embedding provider: {embedding_provider}, using 'unknown' as model name")
    
    # Convert chunks to LlamaIndex Documents
    documents = []
    for chunk in chunks:
        source_metadata = chunk.get("source_document_metadata", {}) or {}
        doc = Document(
            text=chunk.get("text_chunk", ""),
            metadata={
                "chunk_id": chunk.get("chunk_id", ""),
                "arxiv_id": chunk.get("arxiv_id", ""),
                "original_document_title": chunk.get("original_document_title", ""),
                "primary_category": source_metadata.get("primary_category", ""),
                "summary": source_metadata.get("summary", ""),
                "embedding_provider": settings.DEFAULT_EMBEDDING_PROVIDER,
                "embedding_model": embedding_model_name,
            }
        )
        documents.append(doc)
    
    logger.info(f"Inserting {len(documents)} chunk(s) into collection: {config.collection_name}")
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    with tqdm(total=len(documents), desc="Inserting", unit="doc", leave=False) as pbar:
        for doc in documents:
            index.insert(document=doc)
            pbar.update(1)
    
    logger.info(f"Stored {len(documents)} chunk(s) in ChromaDB ({chroma_db_path})")


def _configure_embedding_model() -> None:
    """Configure the embedding model for LlamaIndex."""
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing")
        LlamaSettings.embed_model = LlamaOpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL_NAME
        )
    elif provider == "huggingface":
        LlamaSettings.embed_model = LlamaHuggingFaceEmbedding(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        )
    elif provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            raise ValueError("Ollama base URL is missing")
        LlamaSettings.embed_model = LlamaOllamaEmbedding(
            model_name=settings.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def main():
    """Main entry point for the ingestion pipeline."""
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    setup_logging(level=args.log_level.upper())
    logger.info("Starting ingestion pipeline")

    try:
        config = _validate_and_create_config(args, parser)
        corpus_name = config.effective_corpus_name
        pdf_path, metadata_path = setup_corpus_directories(corpus_name)
        logger.info(f"Corpus: {corpus_name}")

        download_papers(config, pdf_path, metadata_path, skip_download=args.skip_download)
        chunks = process_documents(pdf_path, metadata_path)
        setup_chromadb(config, chunks)

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="MAKERS: Data Ingestion Pipeline")
    
    parser.add_argument("--pdf_dir", type=str, help="Path to directory containing PDF files (required by default)")
    parser.add_argument("--download_from_arxiv", action="store_true", help="Download PDFs from ArXiv instead of local directory")
    
    parser.add_argument("--query", type=str, help="Natural language query for ArXiv search and corpus naming")
    parser.add_argument("--arxiv_keywords", type=str, help="Optimized keywords for ArXiv search (e.g., 'machine learning AND neural networks')")
    parser.add_argument("--max_results", type=int, default=settings.ARXIV_MAX_RESULTS, help=f"Maximum papers to download (default: {settings.ARXIV_MAX_RESULTS})")
    parser.add_argument("--sort_by", type=str, default=settings.ARXIV_SORT_BY, choices=["Relevance", "LastUpdatedDate", "SubmittedDate"], help=f"Sort criterion (default: {settings.ARXIV_SORT_BY})")
    parser.add_argument("--sort_order", type=str, default=settings.ARXIV_SORT_ORDER, choices=["Ascending", "Descending"], help=f"Sort order (default: {settings.ARXIV_SORT_ORDER})")
    parser.add_argument("--skip_download", action="store_true", help="Skip ArXiv download and use existing PDFs")
    
    parser.add_argument("--corpus_name", type=str, help="Specific corpus name (default: derived from query or PDF directory)")
    parser.add_argument("--collection_name", type=str, default=settings.CHROMA_COLLECTION_NAME, help=f"ChromaDB collection name (default: {settings.CHROMA_COLLECTION_NAME})")
    parser.add_argument("--vector_index_name", type=str, default="", help="Not used with ChromaDB (auto-indexed)")
    parser.add_argument("--text_index_name", type=str, default="", help="Not used with ChromaDB")
    
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO)")
    
    return parser


def _validate_and_create_config(args, parser: argparse.ArgumentParser) -> IngestionConfig:
    """Validate arguments and create configuration."""
    pdf_dir = None
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir).resolve()
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            parser.error(f"Directory does not exist: {pdf_dir}")
    
    if not args.download_from_arxiv:
        if not pdf_dir:
            parser.error("--pdf_dir is required. Use --download_from_arxiv to download from ArXiv.")
    elif not args.arxiv_keywords and not args.query:
        parser.error("--query or --arxiv_keywords is required with --download_from_arxiv")
    
    return IngestionConfig(
        pdf_dir=pdf_dir,
        download_from_arxiv=args.download_from_arxiv,
        query=args.query,
        arxiv_keywords=args.arxiv_keywords,
        max_results=args.max_results,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        corpus_name=args.corpus_name,
        collection_name=args.collection_name,
        vector_index_name=args.vector_index_name,
        text_index_name=args.text_index_name
    )

if __name__ == "__main__":
    main()