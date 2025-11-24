"""
Tool Registry

Centralized registry for tools, enabling complete decoupling
between tools and the workflow.
"""

import logging
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized registry for managing all available tools.
    
    Allows registering and retrieving tools by name,
    decoupling the workflow from individual tools.
    """
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, name: str, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            name: The tool name (must match the name used by the agent)
            tool: The tool instance to register
        """
        if name in cls._tools:
            logger.warning(f"Tool '{name}' is already registered. Overwriting.")
        cls._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """
        Retrieve a tool by name.
        
        Args:
            name: The tool name
            
        Returns:
            The tool instance or None if not found
        """
        tool = cls._tools.get(name)
        if tool is None:
            logger.warning(f"Tool '{name}' not found in registry")
        return tool
    
    @classmethod
    def get_all(cls) -> List[BaseTool]:
        """
        Retrieve all registered tools.
        
        Returns:
            List of all tools
        """
        return list(cls._tools.values())
    
    @classmethod
    def get_all_names(cls) -> List[str]:
        """
        Retrieve all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(cls._tools.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: The tool name
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return name in cls._tools
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered tools (useful for tests).
        """
        cls._tools.clear()
        logger.debug("Tool registry cleared")

