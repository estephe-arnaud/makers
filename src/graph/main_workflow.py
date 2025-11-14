"""
Main Workflow Module

This module implements an autonomous research workflow using LangGraph with a single
RAG ReAct agent that handles the entire research process end-to-end through strategic
tool orchestration.

Key Components:
- GraphState: Manages the shared state across the workflow with checkpointing support.
- RAG Agent Node: An autonomous ReAct agent implementing strategic tool selection and
  end-to-end research workflows (search, retrieval, analysis, synthesis).
- Autonomous Flow: Direct execution with agent-driven decision-making, eliminating
  the need for explicit routing logic.
"""

import logging
import uuid
from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from config.settings import settings
from src.agents.agent_architectures import (
    create_rag_react_agent,
)
from src.graph.checkpointer import MongoDBSaver

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    Represents the state of the autonomous research workflow graph.

    This state structure is designed for efficient checkpointing and resumability,
    containing only essential information needed for the autonomous agent workflow.

    Attributes:
        messages: The history of messages in the conversation, enabling context
            preservation across workflow steps.
        user_query: The original query submitted by the user.
        final_output: The final response from the RAG agent after autonomous execution.
        error_message: A message describing an error, if one occurs during execution.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    final_output: Optional[str] = None
    error_message: Optional[str] = None


# Initialize the autonomous RAG ReAct agent
rag_agent_executor = create_rag_react_agent()


def _get_agent_output(
    agent_response: Dict[str, Any], agent_name: str
) -> Optional[str]:
    """
    Extracts the primary content from an agent's response.

    This helper function standardizes the process of getting the output string
    from an agent's execution result. It prioritizes the content of the last
    AIMessage, falls back to the 'output' key, and logs warnings if no
    output can be found.

    Args:
        agent_response: The dictionary returned by the agent executor.
        agent_name: The name of the agent, for logging purposes.

    Returns:
        The extracted output string, or None if no output is found.
    """
    if agent_response.get("messages"):
        for msg in reversed(agent_response.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                logger.debug(f"Extracted output from AIMessage for {agent_name}.")
                return msg.content

    if "output" in agent_response and agent_response["output"]:
        logger.debug(f"Extracted output from 'output' key for {agent_name}.")
        return str(agent_response["output"])

    logger.warning(f"No output found for {agent_name} in messages or 'output' key.")
    return None


async def rag_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes the autonomous RAG ReAct agent to handle the entire research process.

    This node implements an end-to-end research workflow where the agent autonomously:
    - Analyzes query context to determine optimal information retrieval strategy
    - Searches ArXiv for relevant papers using strategic query formulation
    - Queries the knowledge base when appropriate for internal document access
    - Selects and performs deep document analysis on the most promising papers
    - Synthesizes heterogeneous information sources into a comprehensive final answer

    The agent's decision-making is driven by the ReAct pattern, enabling dynamic
    adaptation to query complexity and information availability.
    """
    logger.info("--- RAG REACT AGENT NODE ---")
    
    try:
        # The agent receives the user query and handles everything autonomously
        agent_input = {"messages": state["messages"]}
        response = await rag_agent_executor.ainvoke(agent_input)
        
        # Extract the final output
        final_output = _get_agent_output(response, "RAGReActAgent")
        
        if not final_output:
            logger.error("RAG ReAct Agent failed to produce any output.")
            return {
                "messages": [AIMessage(content="Failed to generate a response. Please try again.")],
                "error_message": "No output from RAG agent",
            }
        
        logger.info(f"RAG Agent Output:\n{final_output}")
        return {
            "final_output": final_output,
            "messages": [AIMessage(content=final_output)],
            "error_message": None,
        }
    except Exception as e:
        logger.error(f"Error in RAG ReAct Agent execution: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error during research: {e}")],
            "error_message": str(e),
        }


def create_workflow_graph() -> StateGraph:
    """
    Creates and configures the autonomous research workflow graph.

    This workflow implements a single-node architecture where the RAG ReAct agent
    handles all research tasks autonomously, eliminating the need for explicit
    routing logic and intermediate state management.

    Returns:
        A compiled StateGraph instance ready for execution with MongoDB checkpointing.
    """
    workflow = StateGraph(GraphState)

    # Add the autonomous RAG agent node
    workflow.add_node("rag_agent", rag_agent_node)

    # Set the entry point
    workflow.set_entry_point("rag_agent")

    # The RAG agent node handles the complete workflow autonomously
    workflow.add_edge("rag_agent", END)
    
    # Compile the graph with the checkpointer
    checkpointer = MongoDBSaver(
        collection_name=settings.LANGGRAPH_CHECKPOINTS_COLLECTION
    )
    return workflow.compile(checkpointer=checkpointer)

# Create a single compiled instance of the graph
app = create_workflow_graph()


async def run_workflow(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the research workflow for a given query.

    Args:
        query: The user's research query.
        thread_id: An optional ID to resume a previous workflow. If not provided,
                   a new one is generated.

    Returns:
        A dictionary containing the final results of the workflow.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
    logger.info(f"Starting workflow for query: '{query}' with thread_id: {thread_id}")

    try:
        initial_state = {"messages": [HumanMessage(content=query)], "user_query": query}
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await app.ainvoke(initial_state, config=config)

        final_output = final_state.get("final_output", "No output available.")
        logger.info("Workflow completed successfully.")
        return {"thread_id": thread_id, "result": final_state, "output": final_output}

    except Exception as e:
        logger.error(f"Workflow execution failed for thread_id {thread_id}: {e}", exc_info=True)
        return {"thread_id": thread_id, "error": str(e)}


async def main():
    """
    Main function to run a test of the research workflow.
    """
    test_query = "What are the latest developments in explainable AI (XAI)?"
    result = await run_workflow(test_query)

    print("\n--- WORKFLOW EXECUTION FINISHED ---")
    if "error" in result:
        print(f"\nTest failed: {result['error']}")
    else:
        print("\nTest completed successfully!")
        print(f"Thread ID: {result['thread_id']}")
        print("\n--- FINAL OUTPUT ---")
        print(result.get("output", "Not available."))
    print("---------------------------------")


if __name__ == "__main__":
    import asyncio
    from config.logging_config import setup_logging

    # Configure logging for rich output
    setup_logging(level="INFO")
    asyncio.run(main())