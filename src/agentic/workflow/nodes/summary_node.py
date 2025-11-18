"""
Summary Node

Summarizes conversation history for long-term memory management.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agentic.workflow.state import GraphState
from src.agentic.agents.agent import get_summary_llm

logger = logging.getLogger(__name__)

# Initialize summary LLM once at module level
summary_llm = get_summary_llm()


async def summary_node(state: GraphState) -> Dict[str, Any]:
    """
    Summary node: Summarizes conversation history for long-term memory.
    
    This node:
    - Takes recent messages and existing summary
    - Creates a condensed summary using LLM
    - Clears old messages to prevent prompt explosion
    - Maintains essential context in conversation_summary
    """
    logger.info("--- SUMMARY NODE ---")
    
    try:
        # Prepare messages to summarize
        messages_to_summarize = state.get("messages", [])
        existing_summary = state.get("conversation_summary", "")
        
        if not messages_to_summarize:
            logger.warning("No messages to summarize")
            return {
                "conversation_summary": existing_summary,
                "messages": [],
            }
        
        # Create summary prompt
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a conversation summarizer. Your task is to create a concise, "
                    "factual summary of the research conversation that preserves all important "
                    "information, findings, and context needed for future interactions.\n\n"
                    "Focus on:\n"
                    "- Key research questions and queries\n"
                    "- Important findings from tool executions (ArXiv searches, KB retrievals, PDF analyses)\n"
                    "- Papers and documents mentioned or analyzed\n"
                    "- Conclusions and insights generated\n"
                    "- Any important context for continuing the research\n\n"
                    "Be concise but comprehensive. Preserve all factual information.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Add existing summary context if available
        if existing_summary:
            context_message = HumanMessage(
                content=f"Previous conversation summary:\n{existing_summary}\n\n"
                "Update and extend this summary with the new messages below."
            )
            summary_messages = [context_message] + messages_to_summarize
        else:
            summary_messages = messages_to_summarize
        
        # Generate summary
        chain = summary_prompt | summary_llm
        summary_response = await chain.ainvoke({"messages": summary_messages})
        
        new_summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
        logger.info(f"Generated summary (length: {len(new_summary)} chars)")
        logger.debug(f"Summary content: {new_summary[:200]}...")
        
        # Keep only the last few messages (for immediate context)
        # This prevents complete loss of recent context
        recent_messages = messages_to_summarize[-3:] if len(messages_to_summarize) > 3 else messages_to_summarize
        
        return {
            "conversation_summary": new_summary,
            "messages": recent_messages,  # Keep last 3 messages for immediate context
            "next_action": "continue",
        }
    except Exception as e:
        logger.error(f"Error in Summary Node execution: {e}", exc_info=True)
        # On error, keep existing summary and messages
        return {
            "conversation_summary": state.get("conversation_summary", ""),
            "messages": state.get("messages", []),
            "error_message": str(e),
        }

