"""
ArXiv Search Tool

Tool for searching scientific papers on ArXiv.
"""

import logging
from typing import List, Dict, Any

from langchain_core.tools import tool
import arxiv

logger = logging.getLogger(__name__)

# Constants for ArXiv search
ARXIV_SORT_CRITERIA = {
    "relevance": arxiv.SortCriterion.Relevance,
    "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    "submitteddate": arxiv.SortCriterion.SubmittedDate,
}

ARXIV_SORT_ORDERS = {
    "ascending": arxiv.SortOrder.Ascending,
    "descending": arxiv.SortOrder.Descending,
}


@tool
def arxiv_search_tool(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    """
    Use this tool to search the ArXiv repository for scientific papers.

    It is your primary method for finding new research papers on a specific topic.
    The query should be a concise string, similar to what you would use in a search engine.

    Args:
        query: A specific and targeted search query.
               Examples: "What are the latest advancements in face analysis", "author:Geoffrey Hinton", "cat:cs.CV"
        max_results: The maximum number of papers to return. Keep this low (e.g., 3 to 5)
                     to avoid overwhelming the analysis stage.
        sort_by: The criterion for sorting results. Options: 'relevance', 'lastUpdatedDate', 'submittedDate'.
        sort_order: The order of results. Options: 'ascending', 'descending'.

    Returns:
        A list of dictionaries, where each dictionary represents a found paper
        with its title, authors, summary, PDF link, and other metadata.
        Returns an empty list if no papers are found.
    """
    logger.info(
        f"Executing arxiv_search_tool: query='{query}', max_results={max_results}"
    )

    try:
        # Validate and map sort parameters
        sort_criterion = ARXIV_SORT_CRITERIA.get(
            sort_by.lower(), arxiv.SortCriterion.Relevance
        )
        if sort_by.lower() not in ARXIV_SORT_CRITERIA:
            logger.warning(f"Invalid sort_by value '{sort_by}'. Using 'relevance'.")

        order_criterion = ARXIV_SORT_ORDERS.get(
            sort_order.lower(), arxiv.SortOrder.Descending
        )
        if sort_order.lower() not in ARXIV_SORT_ORDERS:
            logger.warning(f"Invalid sort_order value '{sort_order}'. Using 'descending'.")

        # Execute search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=order_criterion,
        )

        # Process results
        results = []
        for result in search.results():
            results.append(
                {
                    "entry_id": result.entry_id,
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "summary": result.summary.replace("\n", " "),
                    "published_date": result.published.isoformat()
                    if result.published
                    else None,
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category,
                }
            )

        logger.info(f"Found {len(results)} papers for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"ArXiv search failed for query '{query}': {e}", exc_info=True)
        return [{"error": f"ArXiv search failed: {str(e)}"}]

