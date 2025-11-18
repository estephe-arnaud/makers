"""
API Schemas Module

This module defines the Pydantic models used for request/response validation in the MAKERS API.
It includes schemas for:
- MakersQueryRequest: Input request for invoking the MAKERS system
- MakersOutputMessage: Message format for agent outputs
- MakersResponse: Complete response from the MAKERS system
- ErrorResponse: Standard error response format
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage  # For typing message outputs

class ConfigOverrides(BaseModel):
    """Optional configuration overrides for API requests."""
    llm_provider: Optional[str] = Field(
        None,
        description="Override LLM provider (e.g., 'openai', 'groq', 'ollama')"
    )
    embedding_provider: Optional[str] = Field(
        None,
        description="Override embedding provider (e.g., 'huggingface', 'openai', 'ollama')"
    )


class MakersQueryRequest(BaseModel):
    """Request schema for invoking the MAKERS system."""
    query: str = Field(
        ...,
        description="The user query/question for the MAKERS system",
        min_length=1,
        max_length=1000
    )
    config: Optional[ConfigOverrides] = Field(
        None,
        description="Optional configuration overrides for this request"
    )

class MakersOutputMessage(BaseModel):
    """Message format for agent outputs in the MAKERS system."""
    type: str = Field(..., description="Type of message (e.g., HUMAN, AI, SYSTEM)")
    name: Optional[str] = Field(None, description="Name of the agent or system component")
    content: Any = Field(..., description="Message content (string or structured data)")

    model_config = {"from_attributes": True}

    @classmethod
    def from_langchain_message(cls, msg: BaseMessage) -> "MakersOutputMessage":
        """Create a MakersOutputMessage from a LangChain BaseMessage."""
        return cls(
            type=msg.type.upper(), 
            name=getattr(msg, 'name', None), 
            content=msg.content
        )
    
    @classmethod
    def from_dict(cls, msg_dict: Dict[str, Any]) -> "MakersOutputMessage":
        """Create a MakersOutputMessage from a dictionary (deserialized from SQLite).
        
        Handles both simple dict format and LangChain serialization format.
        """
        # Try simple format first (type, content, name)
        if "type" in msg_dict and "content" in msg_dict:
            return cls(
                type=str(msg_dict.get("type", "unknown")).upper(),
                name=msg_dict.get("name"),
                content=msg_dict.get("content", "")
            )
        
        # Try LangChain serialization format (lc_id, lc_kwargs)
        if "lc_id" in msg_dict and "lc_kwargs" in msg_dict:
            kwargs = msg_dict.get("lc_kwargs", {})
            # Extract type from lc_id (e.g., ["langchain", "schema", "messages", "HumanMessage"])
            lc_id = msg_dict.get("lc_id", [])
            msg_type = lc_id[-1] if isinstance(lc_id, list) and lc_id else "unknown"
            # Remove "Message" suffix if present
            msg_type = msg_type.replace("Message", "").lower() if msg_type else "unknown"
            
            return cls(
                type=msg_type.upper(),
                name=kwargs.get("name"),
                content=kwargs.get("content", "")
            )
        
        # Fallback: try to extract from any available keys
        lc_id = msg_dict.get("lc_id", [])
        lc_kwargs = msg_dict.get("lc_kwargs", {})
        if isinstance(lc_id, list) and lc_id:
            msg_type = lc_id[-1].replace("Message", "").lower()
        else:
            msg_type = "unknown"
        
        return cls(
            type=str(msg_dict.get("type", msg_type)).upper(),
            name=msg_dict.get("name") or (lc_kwargs.get("name") if isinstance(lc_kwargs, dict) else None),
            content=msg_dict.get("content", "") or (lc_kwargs.get("content", "") if isinstance(lc_kwargs, dict) else "")
        )

class MakersResponse(BaseModel):
    """Complete response from the MAKERS system."""
    thread_id: Optional[str] = Field(None, description="Thread identifier (not exposed to users)")
    user_query: str = Field(..., description="Original user query")
    final_output: Optional[str] = Field(None, description="Final response from the agent")
    full_message_history: Optional[List[MakersOutputMessage]] = Field(
        None,
        description="Complete conversation history (optional, for debugging)"
    )
    error_message: Optional[str] = Field(None, description="Any error that occurred")


class ErrorResponse(BaseModel):
    """Standard error response format for the API."""
    detail: str = Field(..., description="Detailed error message")


class HealthResponse(BaseModel):
    """Detailed health check response."""
    status: str = Field(..., description="Overall health status")
    message: str = Field(..., description="Health check message")
    sqlite: str = Field(..., description="SQLite connection status")
    llm_provider: str = Field(..., description="Configured LLM provider")
    embedding_provider: str = Field(..., description="Configured embedding provider")