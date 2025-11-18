"""
Agents Module

Agents and prompts for the agentic system.
"""

from src.agentic.agents.agent import get_agent, get_summary_llm
from src.agentic.agents.utils import get_agent_output, get_tool_calls
from src.agentic.agents.constants import (
    AGENT_TEMPERATURE,
    SUMMARY_LLM_TEMPERATURE,
    DOCUMENT_ANALYST_TEMPERATURE,
    DOCUMENT_SYNTHESIZER_TEMPERATURE,
)

__all__ = [
    "get_agent",
    "get_summary_llm",
    "get_agent_output",
    "get_tool_calls",
    "AGENT_TEMPERATURE",
    "SUMMARY_LLM_TEMPERATURE",
    "DOCUMENT_ANALYST_TEMPERATURE",
    "DOCUMENT_SYNTHESIZER_TEMPERATURE",
]
