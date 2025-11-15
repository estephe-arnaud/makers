"""
Tools Module

Exports all available tools for the agent.
"""

from src.agentic.tools.arxiv import arxiv_search_tool
from src.agentic.tools.knowledge_base import knowledge_base_retrieval_tool
from src.agentic.tools.document_analysis import document_deep_dive_analysis_tool
from src.agentic.tools.registry import ToolRegistry

# Auto-register all tools in the registry
ToolRegistry.register("arxiv_search_tool", arxiv_search_tool)
ToolRegistry.register("knowledge_base_retrieval_tool", knowledge_base_retrieval_tool)
ToolRegistry.register("document_deep_dive_analysis_tool", document_deep_dive_analysis_tool)

__all__ = [
    "arxiv_search_tool",
    "knowledge_base_retrieval_tool",
    "document_deep_dive_analysis_tool",
    "get_all_tools",
    "ToolRegistry",
]


def get_all_tools():
    """Returns a list of all available tools."""
    return ToolRegistry.get_all()

