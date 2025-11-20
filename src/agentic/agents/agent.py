"""
Agent Configuration

Initializes and configures the autonomous agent for the workflow.
"""

from langchain_core.language_models import BaseLanguageModel

from src.agentic.tools import get_all_tools
from src.services.llm import get_llm
from config.settings import settings


def get_agent():
    """
    Creates and returns a configured LLM with tools bound for the workflow.
    
    Instead of using create_openai_tools_agent (which requires intermediate_steps),
    we simply bind tools to the LLM. The LLM will automatically generate tool_calls
    when needed, and we can handle them manually in the workflow nodes.
    
    Returns:
        A configured LLM instance with tools bound, ready for tool call generation.
    """
    agent_llm = get_llm(temperature=settings.AGENT_TEMPERATURE)
    agent_tools = get_all_tools()
    agent = agent_llm.bind_tools(agent_tools)
    return agent


def get_summary_llm() -> BaseLanguageModel:
    """
    Creates and returns the LLM for conversation summarization.
    
    Uses a lower temperature for factual, consistent summarization.
    
    Returns:
        A configured LLM instance for summarization.
    """
    return get_llm(temperature=settings.SUMMARY_LLM_TEMPERATURE)

