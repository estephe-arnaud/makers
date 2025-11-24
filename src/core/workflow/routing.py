"""
Routing Functions

Determines the next step in the workflow based on the current state.
"""

import logging
from typing import Dict, Any

from src.core.workflow.state import GraphState
from config.settings import settings

logger = logging.getLogger(__name__)


def should_summarize(state: GraphState) -> bool:
    """Determine if conversation should be summarized based on message count."""
    message_count = len(state.get("messages", []))
    return message_count >= settings.SUMMARY_THRESHOLD


def route_after_agent(state: GraphState) -> str:
    """Route after agent node: determine next step based on agent's decision."""
    next_action = state.get("next_action")
    
    if next_action == "tool_call":
        return "tool"
    elif next_action == "final_answer":
        # Final output is already stored in state by agent_node
        if state.get("final_output"):
            return "end"
        return "continue"
    else:
        return "continue"


def route_after_tool(state: GraphState) -> str:
    """Route after tool node: decide whether to summarize or continue to agent."""
    if should_summarize(state):
        logger.info(f"Message count ({len(state.get('messages', []))}) >= threshold ({settings.SUMMARY_THRESHOLD}), summarizing...")
        return "summarize"
    return "agent"


def route_after_summary(state: GraphState) -> str:
    """Route after summary node: continue to agent."""
    return "agent"

