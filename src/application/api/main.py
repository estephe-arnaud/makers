"""
MAKERS API Module

This module implements the FastAPI application for the MAKERS system.
It provides endpoints for:
- POST /invoke_makers: Main endpoint for interacting with the MAKERS system
- GET /health: Health check endpoint with system status

The API supports:
- Cross-Origin Resource Sharing (CORS)
- Request/response validation using Pydantic models
- Error handling and logging
- Configuration validation at startup

Security: This simplified API only allows queries and does not expose:
- Thread management endpoints
- Document ingestion endpoints
- File upload capabilities
"""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from config.settings import settings
from config.logging_config import setup_logging
from src.core.workflow.runner import run_workflow
from src.services.storage.checkpointer import get_checkpointer
from src.application.api.schemas import (
    MakersQueryRequest, MakersResponse, ErrorResponse, MakersOutputMessage,
    HealthResponse, ConfigOverrides
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

        <h2>Query MAKERS</h2>
        <form id="invokeForm">
            <label for="query">Your Query:</label>
            <textarea id="query" name="query" rows="5" required placeholder="Enter your research question here...">What are the latest advancements in face analysis</textarea>
            
            <button type="submit">Submit Query</button>
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
            
            responseArea.textContent = 'Loading...';
            
            try {
                const response = await fetch('/invoke_makers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
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
    """Log configuration overrides."""
    if not config:
        return
    if config.llm_provider:
        logger.info(f"Config override: LLM provider = {config.llm_provider}")
    if config.embedding_provider:
        logger.info(f"Config override: Embedding provider = {config.embedding_provider}")


def _convert_messages(messages) -> Optional[list]:
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
    """Process a user query through the MAKERS system and return the response.
    
    Each request creates a new isolated session (no thread persistence exposed).
    """
    # Generate a new thread_id for each request (not exposed to user)
    thread_id = f"api_thread_{uuid.uuid4()}"
    query = request_data.query

    logger.info(f"Received API request. Query: '{query[:50]}...'")
    _log_config_overrides(request_data.config)

    try:
        state = await run_workflow(query=query, thread_id=thread_id)
        
        if not state or "error" in state:
            error_msg = state.get("error", "Workflow execution failed") if state else "Workflow execution failed"
            raise HTTPException(status_code=500, detail=error_msg)

        final_state = state.get("result", state)
        
        return MakersResponse(
            thread_id=None,  # Don't expose thread_id to users
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



# Development server instructions
"""
To run this API locally:
1. Install dependencies: pip install uvicorn[standard]
2. Run: uvicorn src.application.api.main:app --reload --host 0.0.0.0 --port 8000

Example API request:
POST http://localhost:8000/invoke_makers
{
    "query": "What are the latest advancements in face analysis"
}
"""