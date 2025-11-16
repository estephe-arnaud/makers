"""
Workflow Graph

Creates and configures the LangGraph workflow.
"""

from langgraph.graph import StateGraph, END

from config.settings import settings
from src.core.state import GraphState
from src.agentic.workflow.nodes import agent_node, tool_node, summary_node
from src.agentic.workflow.routing import route_after_agent, route_after_tool, route_after_summary


def create_workflow_graph(checkpointer=None) -> StateGraph:
    """
    Creates and configures the autonomous research workflow graph.
    
    This workflow implements a multi-node architecture:
    - Agent Node: Decides on actions (tool calls or final answer)
    - Tool Node: Executes requested tools
    - Summary Node: Maintains long-term memory through conversation summarization
    - Router: Determines next step based on state

    Returns:
        A compiled StateGraph instance ready for execution with SQLite checkpointing.
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
    
    # Compile the graph with the checkpointer (if provided)
    # The checkpointer should be AsyncSqliteSaver to avoid concurrent connections
    # If no checkpointer is provided, compile without one (will be added at runtime)
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


# Global variable to store the compiled graph
_app = None


async def get_app():
    """
    Get the compiled workflow graph with async checkpointer.
    
    This function ensures that the graph uses AsyncSqliteSaver, which is compatible
    with the API layer and avoids concurrent database connections.
    
    Returns:
        Compiled StateGraph instance with async checkpointer.
    """
    global _app
    if _app is None:
        from src.services.storage.checkpointer import get_checkpointer
        checkpointer = await get_checkpointer()
        _app = create_workflow_graph(checkpointer=checkpointer)
    return _app


# For backward compatibility, create a synchronous version
# This will be used only if get_app() is not available (should not happen in async contexts)
def create_app_sync():
    """
    Create the graph synchronously (for backward compatibility only).
    
    WARNING: This creates a synchronous SqliteSaver which can conflict with
    the async checkpointer used by the API. Use get_app() instead in async contexts.
    """
    from src.services.storage.checkpointer import get_checkpointer_sync
    checkpointer = get_checkpointer_sync()
    return create_workflow_graph(checkpointer=checkpointer)


# Create a synchronous version for backward compatibility
# This should only be used in non-async contexts
app = create_app_sync()

