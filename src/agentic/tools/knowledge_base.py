"""
Knowledge Base Retrieval Tool

Tool for retrieving information from the ingested knowledge base.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def knowledge_base_retrieval_tool(
    query_text: str,
    top_k: int = 5,
    metadata_filters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Use this tool to retrieve relevant text chunks from our internal, curated
    knowledge base of already processed documents.

    This is useful for finding specific, targeted information or facts within
    documents that are already part of our system. It is faster than a full
    deep dive.

    Args:
        query_text: The specific question or topic to search for in the knowledge base.
        top_k: The number of most relevant text chunks to return.
        metadata_filters: Optional list of filters to apply to the search,
                          e.g., `[{"key": "arxiv_id", "value": "2301.12345"}]`.

    Returns:
        A list of retrieved document chunks, each with its content, source, and
        relevance score. Returns an empty list if no relevant chunks are found.
    """
    # Import here for lazy initialization
    from src.services.storage.vector_store import RetrievalEngine

    logger.info(
        f"Executing knowledge_base_retrieval_tool: query='{query_text[:50]}...'"
    )

    try:
        retrieval_engine = RetrievalEngine()
        logger.info("RetrievalEngine initialized successfully.")

        # Process metadata filters
        llama_filters = []
        if metadata_filters:
            from llama_index.core.vector_stores import ExactMatchFilter

            for filter_dict in metadata_filters:
                if "key" in filter_dict and "value" in filter_dict:
                    llama_filters.append(
                        ExactMatchFilter(
                            key=filter_dict["key"], value=filter_dict["value"]
                        )
                    )
                else:
                    logger.warning(f"Invalid metadata filter format: {filter_dict}")

        # Execute retrieval
        retrieved_nodes = retrieval_engine.retrieve_simple_vector_search(
            query_text=query_text,
            top_k=top_k,
            metadata_filters=llama_filters if llama_filters else None,
        )

        # Format results
        results = [
            {
                "chunk_id": node.metadata.get("chunk_id", "N/A"),
                "arxiv_id": node.metadata.get("arxiv_id", "N/A"),
                "original_document_title": node.metadata.get(
                    "original_document_title", "N/A"
                ),
                "text_chunk": node.text,
                "retrieval_score": node.score,
            }
            for node in retrieved_nodes
        ]

        logger.info(f"Retrieved {len(results)} chunks from knowledge base.")
        return results

    except Exception as e:
        logger.error(f"Knowledge base retrieval failed: {e}", exc_info=True)
        return [{"error": f"Knowledge base retrieval failed: {str(e)}"}]

