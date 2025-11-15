"""
Agent Configuration

Initializes and configures the autonomous agent for the workflow.
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent

from src.agentic.agents.prompts import AGENT_SYSTEM_PROMPT
from src.agentic.tools import get_all_tools
from src.services.llm import get_llm


def get_agent():
    """
    Creates and returns the configured agent for the workflow.
    
    The agent is initialized without an executor to allow manual tool call handling
    in the workflow nodes.
    
    Returns:
        A configured agent instance ready for tool call generation.
    """
    agent_llm = get_llm(temperature=0.3)
    agent_tools = get_all_tools()
    
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    return create_openai_tools_agent(agent_llm, tools=agent_tools, prompt=agent_prompt)


def get_summary_llm() -> BaseLanguageModel:
    """
    Creates and returns the LLM for conversation summarization.
    
    Uses a lower temperature for factual, consistent summarization.
    
    Returns:
        A configured LLM instance for summarization.
    """
    return get_llm(temperature=0.1)

