"""
Workflow Graph

Creates and configures the LangGraph workflow.
"""

from langgraph.graph import StateGraph, END

from config.settings import settings
from src.core.state import GraphState
from src.services.storage.checkpointer import MongoDBSaver
from src.agentic.workflow.nodes import agent_node, tool_node, summary_node
from src.agentic.workflow.routing import route_after_agent, route_after_tool, route_after_summary


def create_workflow_graph() -> StateGraph:
    """
    Creates and configures the autonomous research workflow graph.
    
    This workflow implements a multi-node architecture:
    - Agent Node: Decides on actions (tool calls or final answer)
    - Tool Node: Executes requested tools
    - Summary Node: Maintains long-term memory through conversation summarization
    - Router: Determines next step based on state

    Returns:
        A compiled StateGraph instance ready for execution with MongoDB checkpointing.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("summarize", summary_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add routing edges
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool": "tool",
            "continue": "agent",
            "end": END,
        },
    )
    
    workflow.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "summarize": "summarize",
            "agent": "agent",
        },
    )
    
    workflow.add_conditional_edges(
        "summarize",
        route_after_summary,
        {
            "agent": "agent",
        },
    )
    
    # Compile the graph with the checkpointer
    checkpointer = MongoDBSaver(
        collection_name=settings.LANGGRAPH_CHECKPOINTS_COLLECTION
    )
    return workflow.compile(checkpointer=checkpointer)


# Create a single compiled instance of the graph
app = create_workflow_graph()

