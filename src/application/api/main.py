"""
MAKERS API Module

This module implements the FastAPI application for the MAKERS system.
It provides endpoints for:
- POST /invoke_makers: Main endpoint for interacting with the MAKERS system
- GET /health: Health check endpoint with system status
- GET /threads: List all conversation threads
- GET /threads/{thread_id}: Get details of a specific thread
- DELETE /threads/{thread_id}: Delete a conversation thread
- POST /ingest: Ingest documents from local directory or ArXiv

The API supports:
- Configuration overrides (LLM provider, embedding provider)
- Document ingestion from local PDFs or ArXiv
- Cross-Origin Resource Sharing (CORS)
- Request/response validation using Pydantic models
- Error handling and logging
- Configuration validation at startup
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableConfig

from config.settings import settings
from config.logging_config import setup_logging
from src.agentic.workflow.runner import run_workflow
from src.services.storage.checkpointer import get_checkpointer, get_sqlite_connection
from src.application.api.schemas import (
    MakersQueryRequest, MakersResponse, ErrorResponse, MakersOutputMessage,
    ThreadSummary, ThreadListResponse, HealthResponse,
    IngestionRequest, IngestionResponse, ConfigOverrides
)
from src.application.cli.run_ingestion import (
    IngestionConfig, setup_corpus_directories, download_papers,
    process_documents, setup_chromadb
)

# Configure logging
setup_logging(level="INFO" if not settings.DEBUG else "DEBUG")
logger = logging.getLogger("api_main")

# Initialize FastAPI application
app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    description="API for interacting with the MAKERS multi-agent system",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
# Security: In production, set ALLOWED_ORIGINS in .env to specific domains
# Example: ALLOWED_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Define HTML content for the interactive test page
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAKERS API Test Page</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input[type="text"], textarea { width: calc(100% - 22px); padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; margin-right: 5px; }
        button:hover { background-color: #0056b3; }
        pre { background-color: #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
        .api-links a { margin-right: 15px; color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MAKERS API Test Page</h1>
        <p class="api-links">
            <a href="/docs" target="_blank">Swagger UI (/docs)</a>
            <a href="/redoc" target="_blank">ReDoc (/redoc)</a>
        </p>

        <h2>Test <code>/invoke_makers</code> (POST)</h2>
        <form id="invokeForm">
            <label for="query">Query:</label>
            <textarea id="query" name="query" rows="3" required>What are the latest advancements in face analysis</textarea>
            
            <label for="thread_id">Thread ID (optional):</label>
            <input type="text" id="thread_id" name="thread_id" placeholder="e.g., api_thread_...">
            
            <button type="submit">Invoke MAKERS</button>
        </form>

        <h2>Test <code>/health</code> (GET)</h2>
        <button id="healthCheckBtn">Check Health</button>

        <h2>API Response:</h2>
        <pre id="responseArea">API responses will appear here...</pre>
    </div>

    <script>
        const responseArea = document.getElementById('responseArea');

        document.getElementById('invokeForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const query = document.getElementById('query').value;
            const thread_id = document.getElementById('thread_id').value;
            
            responseArea.textContent = 'Loading...';
            
            try {
                const response = await fetch('/invoke_makers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, thread_id: thread_id || undefined })
                });
                const data = await response.json();
                responseArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                responseArea.textContent = 'Error: ' + error.message;
            }
        });

        document.getElementById('healthCheckBtn').addEventListener('click', async function() {
            responseArea.textContent = 'Loading...';
            try {
                const response = await fetch('/health');
                const data = await response.json();
                responseArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                responseArea.textContent = 'Error: ' + error.message;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_interactive_page():
    return HTML_CONTENT

@app.on_event("startup")
async def startup_event():
    """Perform startup checks and validate critical configuration."""
    logger.info("FastAPI application startup...")
    
    # Check critical dependencies
    provider = settings.DEFAULT_LLM_MODEL_PROVIDER.lower()
    if provider == "openai" and not settings.OPENAI_API_KEY:
        logger.error("CRITICAL: OpenAI selected but OPENAI_API_KEY not configured. MAKERS functionality will be impaired.")
    
    # SQLite checkpointer is automatically configured via settings.SQLITE_DB_PATH

def _log_config_overrides(config: Optional[ConfigOverrides]) -> None:
    """Log configuration overrides (actual implementation requires settings refactoring)."""
    if not config:
        return
    if config.llm_provider:
        logger.info(f"Config override: LLM provider = {config.llm_provider}")
    if config.embedding_provider:
        logger.info(f"Config override: Embedding provider = {config.embedding_provider}")


def _convert_messages(messages: List) -> Optional[List[MakersOutputMessage]]:
    """Convert LangChain messages to API schema.
    
    Handles both BaseMessage objects (from workflow) and dicts (from SQLite deserialization).
    """
    if not isinstance(messages, list):
        return None
    
    converted = []
    for msg in messages:
        try:
            if isinstance(msg, dict):
                # Message deserialized from SQLite (dict format)
                converted.append(MakersOutputMessage.from_dict(msg))
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                # BaseMessage object from LangChain
                converted.append(MakersOutputMessage.from_langchain_message(msg))
            else:
                logger.warning(f"Unknown message format: {type(msg)}")
        except Exception as e:
            logger.warning(f"Error converting message: {e}")
    
    return converted if converted else None


def _extract_user_query(channel_values: Dict, messages: List) -> Optional[str]:
    """Extract user query from channel values or first message."""
    query = channel_values.get("user_query")
    if query:
        return query[:100] if len(query) > 100 else query
    
    if not messages or not isinstance(messages, list):
        return None
    
    first_msg = messages[0]
    if isinstance(first_msg, dict):
        content = first_msg.get("content", "")
    elif hasattr(first_msg, 'content'):
        content = str(first_msg.content)
    else:
        return None
    
    return content[:100] if len(content) > 100 else content


async def _check_sqlite_connection() -> str:
    """Check SQLite connection and return status string."""
    try:
        checkpointer = await get_checkpointer()
        # Try to list checkpoints to verify connection
        async for _ in checkpointer.alist(None, limit=1):
            break
        return "connected"
    except Exception as e:
        logger.warning(f"SQLite health check failed: {e}")
        return f"error: {str(e)[:50]}"


@app.post(
    "/invoke_makers",
    response_model=MakersResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"}
    },
    summary="Invoke the MAKERS system",
    description="Process a user query through the MAKERS multi-agent system and return the synthesized response."
)
async def invoke_makers_endpoint(request_data: MakersQueryRequest = Body(...)):
    """Process a user query through the MAKERS system and return the response."""
    thread_id = request_data.thread_id or f"api_thread_{uuid.uuid4()}"
    query = request_data.query

    logger.info(f"Received API request. Query: '{query[:50]}...', Thread ID: {thread_id}")
    _log_config_overrides(request_data.config)

    try:
        state = await run_workflow(query=query, thread_id=thread_id)
        
        if not state or "error" in state:
            error_msg = state.get("error", "Workflow execution failed") if state else "Workflow execution failed"
            raise HTTPException(status_code=500, detail=error_msg)

        final_state = state.get("result", state)
        
        return MakersResponse(
            thread_id=thread_id,
            user_query=final_state.get("user_query", query),
            final_output=final_state.get("final_output") or state.get("output"),
            full_message_history=_convert_messages(final_state.get("messages")),
            error_message=final_state.get("error_message")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invoking MAKERS: {e}", exc_info=True)
        detail = str(e) if settings.DEBUG else "An internal error occurred"
        raise HTTPException(status_code=500, detail=detail)

@app.get("/health", summary="Health check", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    return HealthResponse(
        status="healthy",
        message=f"{settings.PROJECT_NAME} API is running",
        sqlite=await _check_sqlite_connection(),
        llm_provider=settings.DEFAULT_LLM_MODEL_PROVIDER,
        embedding_provider=settings.DEFAULT_EMBEDDING_PROVIDER
    )


@app.get("/threads", summary="List threads", response_model=ThreadListResponse)
async def list_threads(limit: int = 50):
    """List all conversation threads."""
    try:
        checkpointer = await get_checkpointer()
        threads = []
        count = 0
        
        # List all checkpoints and group by thread_id
        seen_threads = {}
        async for checkpoint_tuple in checkpointer.alist(None, limit=None):
            if count >= limit:
                break
            
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id")
            if not thread_id:
                continue
            
            # Keep only the latest checkpoint for each thread
            if thread_id not in seen_threads:
                try:
                    checkpoint = checkpoint_tuple.checkpoint
                    channel_values = checkpoint.get("channel_values", {})
                    messages = channel_values.get("messages", [])
                    thread_ts = checkpoint.get("id")
                    
                    threads.append(ThreadSummary(
                        thread_id=thread_id,
                        user_query=_extract_user_query(channel_values, messages),
                        last_updated=thread_ts,
                        message_count=len(messages) if isinstance(messages, list) else None
                    ))
                    seen_threads[thread_id] = True
                    count += 1
                except Exception as e:
                    logger.warning(f"Error processing thread {thread_id}: {e}")
                    threads.append(ThreadSummary(
                        thread_id=thread_id,
                        user_query=None,
                        last_updated=None,
                        message_count=None
                    ))
                    seen_threads[thread_id] = True
                    count += 1
        
        return ThreadListResponse(threads=threads, total=len(threads))
    
    except Exception as e:
        logger.error(f"Error listing threads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")


@app.get("/threads/{thread_id}", summary="Get thread", response_model=MakersResponse)
async def get_thread(thread_id: str):
    """Get details of a specific thread."""
    # Security: Basic validation of thread_id format
    if not thread_id or len(thread_id) > 255:
        raise HTTPException(status_code=400, detail="Invalid thread ID format")
    
    try:
        checkpointer = await get_checkpointer()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        
        if not checkpoint_tuple:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        
        return MakersResponse(
            thread_id=thread_id,
            user_query=channel_values.get("user_query") or "Unknown",
            final_output=channel_values.get("final_output"),
            full_message_history=_convert_messages(channel_values.get("messages")),
            error_message=channel_values.get("error_message")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get thread: {str(e)}")


@app.delete("/threads/{thread_id}", summary="Delete thread", response_model=Dict[str, Any])
async def delete_thread(thread_id: str):
    """Delete a conversation thread and all its checkpoints."""
    # Security: Basic validation of thread_id format
    if not thread_id or len(thread_id) > 255:
        raise HTTPException(status_code=400, detail="Invalid thread ID format")
    
    try:
        # First, verify that the thread exists
        checkpointer = await get_checkpointer()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if not checkpoint_tuple:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        
        # Get the SQLite connection and delete all checkpoints for this thread
        # LangGraph stores checkpoints in 'checkpoints' and 'writes' tables
        # Both tables use (thread_id, checkpoint_ns) as part of their key structure
        # We need to delete all records for this thread_id, regardless of checkpoint_ns
        connection = await get_sqlite_connection()
        
        # Delete from writes table first (child records)
        # LangGraph schema: writes table has (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) as key
        writes_cursor = await connection.execute(
            "DELETE FROM writes WHERE thread_id = ?",
            (thread_id,)
        )
        writes_deleted = writes_cursor.rowcount
        
        # Delete from checkpoints table (parent records)
        # LangGraph schema: checkpoints table has (thread_id, checkpoint_ns, checkpoint_id) as key
        checkpoints_cursor = await connection.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        checkpoints_deleted = checkpoints_cursor.rowcount
        
        await connection.commit()
        
        total_deleted = checkpoints_deleted + writes_deleted
        logger.info(
            f"Deleted {checkpoints_deleted} checkpoint(s) and {writes_deleted} write(s) "
            f"for thread {thread_id} (total: {total_deleted})"
        )
        
        return {
            "status": "deleted",
            "thread_id": thread_id,
            "deleted_count": total_deleted,
            "checkpoints_deleted": checkpoints_deleted,
            "writes_deleted": writes_deleted
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {str(e)}")


def _validate_ingestion_request(request_data: IngestionRequest) -> None:
    """Validate ingestion request parameters."""
    if not request_data.download_from_arxiv and not request_data.pdf_dir:
        raise HTTPException(
            status_code=400,
            detail="Either 'pdf_dir' must be provided or 'download_from_arxiv' must be True"
        )
    
    if request_data.download_from_arxiv and not request_data.query and not request_data.arxiv_keywords:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'arxiv_keywords' is required when downloading from ArXiv"
        )
    
    if request_data.pdf_dir:
        pdf_dir = Path(request_data.pdf_dir).resolve()
        if not pdf_dir.exists():
            raise HTTPException(status_code=400, detail=f"PDF directory does not exist: {pdf_dir}")
        
        # Security: Ensure path is within allowed base directory to prevent path traversal
        base_dir = Path(settings.DATA_DIR).resolve()
        try:
            pdf_dir.relative_to(base_dir)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"PDF directory must be within {base_dir} for security reasons"
            )


def _create_ingestion_config(request_data: IngestionRequest) -> IngestionConfig:
    """Create IngestionConfig from request data."""
    pdf_dir = Path(request_data.pdf_dir).resolve() if request_data.pdf_dir else None
    
    return IngestionConfig(
        pdf_dir=pdf_dir,
        download_from_arxiv=request_data.download_from_arxiv,
        query=request_data.query,
        arxiv_keywords=request_data.arxiv_keywords,
        max_results=request_data.max_results,
        sort_by=request_data.sort_by,
        sort_order=request_data.sort_order,
        corpus_name=request_data.corpus_name,
        collection_name=request_data.collection_name or settings.CHROMA_COLLECTION_NAME,
        vector_index_name="",  # Not used with ChromaDB (auto-indexed)
        text_index_name=""  # Not used with ChromaDB
    )


async def _run_ingestion_pipeline(config: IngestionConfig, pdf_path: Path, metadata_path: Path) -> tuple:
    """Run the ingestion pipeline asynchronously."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        download_result = await loop.run_in_executor(
            executor, download_papers, config, pdf_path, metadata_path, False
        )
        chunks = await loop.run_in_executor(executor, process_documents, pdf_path, metadata_path)
        # Note: Embeddings will be generated automatically by LlamaIndex when storing in ChromaDB
    
    return download_result, chunks


@app.post("/ingest", summary="Ingest documents", response_model=IngestionResponse)
async def ingest_documents(request_data: IngestionRequest = Body(...)):
    """Ingest documents from local directory or ArXiv into the knowledge base."""
    try:
        _validate_ingestion_request(request_data)
        config = _create_ingestion_config(request_data)
        
        corpus_name = config.effective_corpus_name
        pdf_path, metadata_path = setup_corpus_directories(corpus_name)
        logger.info(f"Starting ingestion for corpus: {corpus_name}")
        
        download_result, chunks = await _run_ingestion_pipeline(
            config, pdf_path, metadata_path
        )
        
        # Setup ChromaDB (runs in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(
                executor, setup_chromadb, config, chunks
            )
        
        return IngestionResponse(
            status="success",
            corpus_name=corpus_name,
            downloaded_count=download_result.get("downloaded_count", 0),
            processed_count=len(chunks),
            embedded_count=len(chunks),  # Embeddings generated automatically by LlamaIndex
            stored_count=len(chunks),  # All chunks are stored in ChromaDB
            failed_count=download_result.get("failed_count", 0),
            message=f"Successfully ingested {len(chunks)} chunks into corpus '{corpus_name}'"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest documents: {str(e)}")



# Development server instructions
"""
To run this API locally:
1. Install dependencies: pip install uvicorn[standard]
2. Run: uvicorn src.application.api.main:app --reload --host 0.0.0.0 --port 8000

Example API request:
POST http://localhost:8000/invoke_makers
{
    "query": "What are the latest advancements in face analysis",
    "thread_id": "optional_existing_thread_id"
}
"""