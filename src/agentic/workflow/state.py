"""
Graph State Definition

Defines the state structure for the LangGraph workflow.
"""

from typing import TypedDict, Annotated, List, Optional, Literal
import operator

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of the autonomous research workflow graph.
    
    This state structure supports long-term memory through conversation summarization,
    preventing prompt size explosion while maintaining context across long conversations.

    Attributes:
        messages: The recent messages in the conversation (last N messages before summarization)
        conversation_summary: Accumulated summary of older conversation history
        user_query: The original query submitted by the user
        final_output: The final response from the agent after autonomous execution
        next_action: The next action to take (tool_call, final_answer, summarize, continue)
        error_message: A message describing an error, if one occurs during execution
        iteration_count: Counter to prevent infinite loops (incremented on each agent node call)
    """
    messages: Annotated[List[BaseMessage], operator.add]
    conversation_summary: Optional[str]
    user_query: str
    final_output: Optional[str]
    next_action: Optional[Literal["tool_call", "final_answer", "summarize", "continue"]]
    error_message: Optional[str]
    iteration_count: Optional[int]

