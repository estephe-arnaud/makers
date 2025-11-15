# makers/scripts/run_ingestion.py
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
from src.services.ingestion.embedder import generate_embeddings_for_chunks
from src.services.storage.mongodb import MongoDBManager
from pymongo.errors import ConnectionFailure

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
            return sanitize_query_for_directory_name(self.pdf_dir.name)
        return "default_corpus"


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
    """Copy or use local PDF files from the specified directory."""
    logger.info(f"Using local PDF directory: {local_pdf_dir}")
    
    if not local_pdf_dir.exists():
        raise ValueError(f"Specified local directory does not exist: {local_pdf_dir}")
    
    if not local_pdf_dir.is_dir():
        raise ValueError(f"Specified path is not a directory: {local_pdf_dir}")
    
    # Find all PDF files in the local directory
    local_pdfs = list(local_pdf_dir.glob("*.pdf"))
    
    if not local_pdfs:
        logger.warning(f"No PDF files found in {local_pdf_dir}")
        return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}
    
    logger.info(f"Found {len(local_pdfs)} PDF file(s) in local directory")
    
    # Copy PDFs to the corpus directory
    copied_count = 0
    skipped_count = 0
    failed_count = 0
    
    for pdf_file in local_pdfs:
        dest_pdf = pdf_path / pdf_file.name
        
        if dest_pdf.exists():
            logger.debug(f"File {pdf_file.name} already exists, skipping")
            skipped_count += 1
        else:
            try:
                shutil.copy2(pdf_file, dest_pdf)
                logger.debug(f"Copied {pdf_file.name} to {dest_pdf}")
                copied_count += 1
            except Exception as e:
                logger.error(f"Error copying {pdf_file.name}: {e}")
                failed_count += 1
    
    logger.info(
        f"Process completed. Copied: {copied_count}, "
        f"Failed: {failed_count}, "
        f"Skipped: {skipped_count}"
    )
    
    return {
        "downloaded_count": copied_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count
    }

def download_papers(config: IngestionConfig, pdf_path: Path, metadata_path: Path, skip_download: bool = False) -> dict:
    """Download papers from ArXiv or use local PDFs."""
    if config.download_from_arxiv:
        if skip_download:
            logger.info("Skipping ArXiv paper download")
            return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}
        
        return _download_from_arxiv(config, pdf_path, metadata_path)
    else:
        if not config.pdf_dir:
            raise ValueError(
                "--pdf_dir is required by default. "
                "Use --download_from_arxiv to download from ArXiv."
            )
        return use_local_pdfs(config.pdf_dir, pdf_path, metadata_path)


def _download_from_arxiv(config: IngestionConfig, pdf_path: Path, metadata_path: Path) -> dict:
    """Download papers from ArXiv."""
    query = config.effective_query
    logger.info(f"Searching ArXiv with query: '{query}'")
    
    try:
        search_results = search_arxiv_papers(
            query=query,
            max_results=config.max_results,
            sort_by=config.sort_by,
            sort_order=config.sort_order
        )

        if not search_results:
            logger.info("No papers found on ArXiv for the given query.")
            return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}

        logger.info(f"Found {len(search_results)} paper(s) on ArXiv. Starting download process...")
        
        download_stats = execute_arxiv_downloads(
            results=search_results,
            pdf_dir=pdf_path,
            metadata_dir=metadata_path
        )
        
        logger.info(
            f"ArXiv download completed. "
            f"Downloaded: {download_stats['successful']}, "
            f"Failed: {download_stats['failed']}, "
            f"Skipped: {download_stats['skipped']}"
        )
        
        return {
            "downloaded_count": download_stats['successful'], 
            "failed_count": download_stats['failed'],
            "skipped_count": download_stats['skipped']
        }
    except Exception as e:
        logger.error(f"ArXiv download failed: {e}", exc_info=True)
        raise

def process_documents(pdf_path: Path, metadata_path: Path) -> list:
    """Parse and preprocess documents."""
    logger.info("Parsing documents...")
    documents = parse_document_collection(pdf_path, metadata_path)
    if not documents:
        logger.warning("No documents parsed")
        return []

    logger.info(f"Parsed {len(documents)} documents")
    logger.info("Preprocessing documents...")
    
    chunks = preprocess_parsed_documents(documents)
    if not chunks:
        logger.warning("No chunks generated")
        return []

    logger.info(f"Generated {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list) -> list:
    """Generate embeddings for document chunks."""
    if not chunks:
        logger.info("No chunks to embed")
        return []

    logger.info("Generating embeddings...")
    embedded_chunks = generate_embeddings_for_chunks(chunks)
    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
    return embedded_chunks

def setup_mongodb(config: IngestionConfig, chunks: list) -> None:
    """Set up MongoDB with chunks and indexes."""
    if not chunks:
        logger.info("No chunks to store in MongoDB")
        return

    logger.info("Setting up MongoDB...")
    mongo_mgr = MongoDBManager(
        mongo_uri=settings.MONGODB_URI,
        db_name=settings.MONGO_DATABASE_NAME
    )
    
    try:
        mongo_mgr.connect()

        # Insert chunks
        logger.info(f"Inserting {len(chunks)} chunks into {config.collection_name}")
        insert_summary = mongo_mgr.insert_chunks_with_embeddings(
            chunks,
            collection_name=config.collection_name
        )
        logger.info(f"Insertion summary: {insert_summary}")

        # Create indexes
        _create_indexes(mongo_mgr, config)
        logger.info("MongoDB setup complete")

    except ConnectionFailure as e:
        logger.error("MongoDB connection failed", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"MongoDB setup failed: {e}", exc_info=True)
        raise
    finally:
        mongo_mgr.close()


def _create_indexes(mongo_mgr: MongoDBManager, config: IngestionConfig) -> None:
    """Create vector and text indexes."""
    # Vector index
    mongo_mgr.create_vector_search_index(
        collection_name=config.collection_name,
        index_name=config.vector_index_name,
        embedding_field="embedding",
        filter_fields=[
            "metadata.arxiv_id",
            "metadata.original_document_title",
            "metadata.primary_category"
        ]
    )

    # Text index
    mongo_mgr.create_text_search_index(
        collection_name=config.collection_name,
        index_name=config.text_index_name,
        text_field="text_chunk",
        additional_text_fields={
            "metadata.original_document_title": "string",
            "metadata.summary": "string"
        }
    )

def main():
    parser = argparse.ArgumentParser(description="MAKERS: Data Ingestion Pipeline")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        help="Path to a directory containing PDF files to use (required by default, except with --download_from_arxiv)"
    )
    parser.add_argument(
        "--download_from_arxiv",
        action="store_true",
        help="Download PDFs from ArXiv instead of using a local directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural language query for corpus naming and ArXiv search (required with --download_from_arxiv)"
    )
    parser.add_argument(
        "--arxiv_keywords",
        type=str,
        help="Optimized keywords for ArXiv search (e.g., English, using AND/OR) (required with --download_from_arxiv)"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=settings.ARXIV_MAX_RESULTS,
        help=f"Maximum papers to download (default: {settings.ARXIV_MAX_RESULTS})"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default=settings.ARXIV_SORT_BY,
        choices=["Relevance", "LastUpdatedDate", "SubmittedDate"],
        help=f"Sort criterion (default: {settings.ARXIV_SORT_BY})"
    )
    parser.add_argument(
        "--sort_order",
        type=str,
        default=settings.ARXIV_SORT_ORDER,
        choices=["Ascending", "Descending"],
        help=f"Sort order (default: {settings.ARXIV_SORT_ORDER})"
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        help="Specific corpus name (default: derived from query)"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME,
        help=f"MongoDB collection name (default: {MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--vector_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_VECTOR_INDEX_NAME,
        help=f"Vector index name (default: {MongoDBManager.DEFAULT_VECTOR_INDEX_NAME})"
    )
    parser.add_argument(
        "--text_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_TEXT_INDEX_NAME,
        help=f"Text index name (default: {MongoDBManager.DEFAULT_TEXT_INDEX_NAME})"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip ArXiv download and use existing PDFs (only with --download_from_arxiv)"
    )

    args = parser.parse_args()
    
    # Validate and prepare configuration
    config = _validate_and_create_config(args, parser)
    
    setup_logging(level=args.log_level.upper())
    logger.info("Starting ingestion pipeline")

    try:
        # Setup directories
        corpus_name = config.effective_corpus_name
        pdf_path, metadata_path = setup_corpus_directories(corpus_name)
        logger.info(f"Using corpus: {corpus_name}")

        # Execute pipeline
        download_papers(config, pdf_path, metadata_path, skip_download=args.skip_download)
        chunks = process_documents(pdf_path, metadata_path)
        embedded_chunks = generate_embeddings(chunks)
        setup_mongodb(config, embedded_chunks)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _validate_and_create_config(args, parser: argparse.ArgumentParser) -> IngestionConfig:
    """Validate arguments and create configuration."""
    # Validate pdf_dir
    pdf_dir = None
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir).resolve()
        if not pdf_dir.exists():
            parser.error(f"Directory does not exist: {pdf_dir}")
        if not pdf_dir.is_dir():
            parser.error(f"Path is not a directory: {pdf_dir}")
    
    # Validate requirements
    if not args.download_from_arxiv:
        if not pdf_dir:
            parser.error("--pdf_dir is required by default. Use --download_from_arxiv to download from ArXiv.")
    else:
        # When downloading from ArXiv, at least one of query or arxiv_keywords must be explicitly provided
        if not args.arxiv_keywords and not args.query:
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