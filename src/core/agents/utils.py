"""
Agent Utilities

Helper functions for agent operations.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)


def get_agent_output(
    agent_response: Dict[str, Any], agent_name: str
) -> Optional[str]:
    """
    Extracts the primary content from an agent's response.

    Args:
        agent_response: The dictionary returned by the agent executor
        agent_name: The name of the agent, for logging purposes

    Returns:
        The extracted output string, or None if no output is found
    """
    if agent_response.get("messages"):
        for msg in reversed(agent_response.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                logger.debug(f"Extracted output from AIMessage for {agent_name}.")
                return msg.content

    if "output" in agent_response and agent_response["output"]:
        logger.debug(f"Extracted output from 'output' key for {agent_name}.")
        return str(agent_response["output"])

    logger.warning(f"No output found for {agent_name} in messages or 'output' key.")
    return None


def get_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from the last AIMessage.
    
    Returns a list of tool call dictionaries with keys: 'name', 'args', 'id'.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # tool_calls can be a list of dicts or ToolCall objects
            # Convert to list of dicts for consistent handling
            tool_calls = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append(tc)
                else:
                    # ToolCall object - extract attributes
                    tool_calls.append({
                        "name": getattr(tc, "name", ""),
                        "args": getattr(tc, "args", {}),
                        "id": getattr(tc, "id", ""),
                    })
            return tool_calls
    return []

