"""
Tool Node

Executes tools requested by the agent.
"""

import logging
import json
from typing import Dict, Any

from langchain_core.messages import ToolMessage

from src.core.workflow.state import GraphState
from src.core.agents.utils import get_tool_calls
from src.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def tool_node(state: GraphState) -> Dict[str, Any]:
    """
    Tool node: Executes tools requested by the agent.
    
    This node:
    - Extracts tool calls from the agent's last message
    - Executes each tool call
    - Returns ToolMessage responses
    """
    logger.info("--- TOOL NODE ---")
    
    try:
        tool_calls = get_tool_calls(state["messages"])
        
        if not tool_calls:
            logger.warning("Tool node called but no tool calls found")
            return {
                "messages": [],
                "next_action": "continue",
            }
        
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Get tool from registry (modular approach)
            tool = ToolRegistry.get(tool_name)
            
            if tool is None:
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                    )
                )
                continue
            
            try:
                # Execute the tool (tools are synchronous by default)
                # Use invoke for synchronous tools, ainvoke if available
                if hasattr(tool, 'ainvoke'):
                    result = await tool.ainvoke(tool_args)
                else:
                    result = tool.invoke(tool_args)
                
                # Convert result to string if needed
                if isinstance(result, (dict, list)):
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                else:
                    result_str = str(result)
                
                tool_messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call_id,
                    )
                )
                logger.info(f"Tool {tool_name} executed successfully")
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                    )
                )
        
        return {
            "messages": tool_messages,
            "next_action": "continue",
        }
    except Exception as e:
        logger.error(f"Error in Tool Node execution: {e}", exc_info=True)
        return {
            "messages": [
                ToolMessage(
                    content=f"Error in tool execution: {e}",
                    tool_call_id="tool_node_error",
                )
            ],
            "next_action": "continue",
            "error_message": str(e),
        }

