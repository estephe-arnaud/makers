"""
Tools Module

Exports all available tools for the agent.
"""

from src.core.tools.arxiv import arxiv_search_tool
from src.core.tools.knowledge_base import knowledge_base_retrieval_tool
# Note: document_deep_dive_analysis_tool is kept in the codebase but not registered
# for security reasons (prevents automatic PDF downloads from URLs)
# from src.core.tools.document_analysis import document_deep_dive_analysis_tool
from src.core.tools.registry import ToolRegistry

# Auto-register all tools in the registry
ToolRegistry.register("arxiv_search_tool", arxiv_search_tool)
ToolRegistry.register("knowledge_base_retrieval_tool", knowledge_base_retrieval_tool)
# document_deep_dive_analysis_tool is not registered for security reasons

__all__ = [
    "arxiv_search_tool",
    "knowledge_base_retrieval_tool",
    # "document_deep_dive_analysis_tool",  # Not available for security reasons
    "get_all_tools",
    "ToolRegistry",
]


def get_all_tools():
    """Returns a list of all available tools."""
    return ToolRegistry.get_all()

