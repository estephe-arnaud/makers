"""
Workflow Runner

Executes the research workflow for user queries.
"""

import logging
import uuid
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage

from src.agentic.workflow.graph import get_app
from src.agentic.agents.utils import get_agent_output

logger = logging.getLogger(__name__)


async def run_workflow(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the research workflow for a given query.

    Args:
        query: The user's research query
        thread_id: An optional ID to resume a previous workflow. If not provided,
                   a new one is generated.

    Returns:
        A dictionary containing the final results of the workflow.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
    logger.info(f"Starting workflow for query: '{query}' with thread_id: {thread_id}")

    try:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "conversation_summary": None,
            "user_query": query,
            "final_output": None,
            "next_action": None,
            "error_message": None,
            "iteration_count": 0,
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Get the graph with async checkpointer to avoid concurrent database connections
        app = await get_app()
        final_state = await app.ainvoke(initial_state, config=config)

        # Extract final output from last AIMessage
        final_output = final_state.get("final_output")
        if not final_output:
            final_output = get_agent_output(
                {"messages": final_state.get("messages", [])}, "Agent"
            )
        
        if not final_output:
            final_output = "No output available."
        
        logger.info("Workflow completed successfully.")
        return {
            "thread_id": thread_id,
            "result": final_state,
            "output": final_output,
        }
    except Exception as e:
        logger.error(f"Workflow execution failed for thread_id {thread_id}: {e}", exc_info=True)
        return {"thread_id": thread_id, "error": str(e)}

