"""
Tool Registry

Registry centralisé pour les tools, permettant un découplage complet
entre les tools et le workflow.
"""

import logging
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry centralisé pour gérer tous les tools disponibles.
    
    Permet d'enregistrer et de récupérer des tools par leur nom,
    découplant ainsi le workflow des tools individuels.
    """
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, name: str, tool: BaseTool) -> None:
        """
        Enregistre un tool dans le registry.
        
        Args:
            name: Le nom du tool (doit correspondre au nom utilisé par l'agent)
            tool: L'instance du tool à enregistrer
        """
        if name in cls._tools:
            logger.warning(f"Tool '{name}' is already registered. Overwriting.")
        cls._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """
        Récupère un tool par son nom.
        
        Args:
            name: Le nom du tool
            
        Returns:
            L'instance du tool ou None si non trouvé
        """
        tool = cls._tools.get(name)
        if tool is None:
            logger.warning(f"Tool '{name}' not found in registry")
        return tool
    
    @classmethod
    def get_all(cls) -> List[BaseTool]:
        """
        Récupère tous les tools enregistrés.
        
        Returns:
            Liste de tous les tools
        """
        return list(cls._tools.values())
    
    @classmethod
    def get_all_names(cls) -> List[str]:
        """
        Récupère tous les noms de tools enregistrés.
        
        Returns:
            Liste des noms de tools
        """
        return list(cls._tools.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Vérifie si un tool est enregistré.
        
        Args:
            name: Le nom du tool
            
        Returns:
            True si le tool est enregistré, False sinon
        """
        return name in cls._tools
    
    @classmethod
    def clear(cls) -> None:
        """
        Efface tous les tools enregistrés (utile pour les tests).
        """
        cls._tools.clear()
        logger.debug("Tool registry cleared")

