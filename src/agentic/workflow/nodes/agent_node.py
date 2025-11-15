"""
Agent Node

The agent analyzes the context and decides on the next action (tool calls or final answer).
"""

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from src.core.state import GraphState
from src.core.constants import MAX_ITERATIONS
from src.agentic.agents.agent import get_agent

logger = logging.getLogger(__name__)

# Initialize agent once at module level
agent = get_agent()


async def agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Agent node: The agent analyzes the context and decides on the next action.
    
    This node uses the agent to:
    - Analyze query context and conversation history
    - Decide whether to call tools or provide a final answer
    - Generate tool calls or final synthesis
    
    The agent receives:
    - conversation_summary (if exists) for long-term memory
    - Recent messages for immediate context
    
    Note: This node uses the agent directly (without AgentExecutor) to get
    tool calls without executing them. Tool execution happens in tool_node.
    """
    logger.info("--- AGENT NODE ---")
    
    # Check iteration limit to prevent infinite loops
    iteration_count = state.get("iteration_count", 0) + 1
    if iteration_count > MAX_ITERATIONS:
        logger.error(f"Maximum iterations ({MAX_ITERATIONS}) reached. Terminating workflow.")
        return {
            "messages": [AIMessage(content="Maximum iterations reached. The workflow has been terminated to prevent infinite loops.")],
            "final_output": "Maximum iterations reached. Please try a more specific query or contact support.",
            "next_action": "final_answer",
            "error_message": f"Maximum iterations ({MAX_ITERATIONS}) exceeded",
            "iteration_count": iteration_count,
        }
    
    try:
        # Build context for the agent
        agent_messages = []
        
        # Add conversation summary if it exists (long-term memory)
        if state.get("conversation_summary"):
            agent_messages.append(
                HumanMessage(
                    content=f"Previous conversation summary:\n{state['conversation_summary']}\n\n"
                    "Use this summary as context for the ongoing conversation."
                )
            )
        
        # Add recent messages (immediate context)
        agent_messages.extend(state["messages"])
        
        # Invoke the agent directly (without executor) to get tool calls
        # The agent will return an AIMessage with tool_calls or a final answer
        # Note: agent_scratchpad is built automatically by LangChain from messages
        response = await agent.ainvoke({"messages": agent_messages})
        
        # Extract the agent's response message
        # create_openai_tools_agent returns an AIMessage directly
        if isinstance(response, AIMessage):
            agent_message = response
        elif isinstance(response, dict):
            # If it's a dict, try to extract messages
            messages = response.get("messages", [])
            if messages:
                # Get the last AIMessage from the list
                agent_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        agent_message = msg
                        break
                if not agent_message:
                    agent_message = AIMessage(content=str(response))
            else:
                agent_message = AIMessage(content=str(response))
        elif isinstance(response, list):
            # If it's a list, get the last AIMessage
            agent_message = None
            for msg in reversed(response):
                if isinstance(msg, AIMessage):
                    agent_message = msg
                    break
            if not agent_message:
                agent_message = AIMessage(content=str(response))
        else:
            agent_message = AIMessage(content=str(response))
        
        # Determine next action based on agent response
        has_tools = bool(agent_message.tool_calls)
        # Check if content exists and is not just whitespace
        has_content = bool(agent_message.content and agent_message.content.strip())
        
        if has_tools:
            next_action = "tool_call"
            logger.info(f"Agent decided to call {len(agent_message.tool_calls)} tool(s)")
        elif has_content and not has_tools:
            next_action = "final_answer"
            logger.info("Agent provided final answer")
            # Store final output
            return {
                "messages": [agent_message],
                "final_output": agent_message.content,
                "next_action": next_action,
                "error_message": None,
                "iteration_count": iteration_count,
            }
        else:
            next_action = "continue"
            logger.warning("Agent response unclear (no tools and no meaningful content), continuing")

        return {
            "messages": [agent_message],
            "next_action": next_action,
            "error_message": None,
            "iteration_count": iteration_count,
        }
    except Exception as e:
        logger.error(f"Error in Agent Node execution: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error during agent reasoning: {e}")],
            "next_action": "continue",
            "error_message": str(e),
            "iteration_count": iteration_count,
        }

