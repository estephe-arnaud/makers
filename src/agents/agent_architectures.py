"""
Agent Architectures Module

This module defines the core agent architectures used in the research assistant system.
The primary agent is the autonomous RAG ReAct agent that handles end-to-end research workflows.

Primary Agent:
- RAG ReAct Agent: Autonomous agent implementing strategic tool orchestration for complete
  research workflows (search, retrieval, analysis, synthesis).

Legacy Agents (kept for compatibility with notebooks and tests):
- Research Planner: Legacy agent for structured research planning.
- ArXiv Search: Legacy agent for ArXiv searches.
- Document Analysis: Legacy agent for document analysis.
- Synthesis: Legacy agent for report synthesis.

Note: The workflow now uses only the RAG ReAct agent, which replaces the need for
these specialized agents through autonomous decision-making.
"""

import logging
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.agents import create_openai_tools_agent, AgentExecutor

from config.settings import settings
from src.agents.tool_definitions import (
    knowledge_base_retrieval_tool,
    arxiv_search_tool,
    document_deep_dive_analysis_tool,
)
from src.llm_services.llm_factory import get_llm, SYNTHESIS_LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# --- Legacy Agents (kept for compatibility with notebooks and tests) ---
# Note: These agents are no longer used in the main workflow but are preserved
# for backward compatibility with notebooks and testing purposes.

# --- Agent 1: Research Planner Agent (Legacy) ---

RESEARCH_PLANNER_SYSTEM_PROMPT = """
**Role:** You are a meticulous and strategic Research Planner.

**Goal:** To transform a user's research query into a structured, actionable plan
that other specialized agents will execute. You do not perform any searches or
analysis yourself; your sole output is the plan.

**User Query Context:** The user is asking about a scientific topic, likely in the
domain of Machine Learning or Artificial Intelligence.

### Directives:
1.  **Deconstruct the Query:** Break down the user's request into fundamental questions.
2.  **Identify Sources:** Pinpoint the best places to find answers (e.g., new ArXiv
    searches, specific journals, our internal knowledge base).
3.  **Formulate Search Queries:** Create specific, effective search queries for each
    source. If the user's query implies recent trends, suggest a date range.
    **Crucially, if you recommend an ArXiv search, provide a clear, single query
    for it like this: `arxiv: "your query here"`**.
4.  **Define Analysis Steps:** Briefly outline how the retrieved information should be
    analyzed.
5.  **Structure the Output:** Present the plan in a clear, easy-to-read format
    (like Markdown).

**Constraint:** You MUST ONLY output the research plan. Do not write any
introductions or conversational text. Your entire response should be the plan itself.
"""


def create_research_planner_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Research Planner Agent.

    This agent is tool-less. Its only job is to receive a user query and
    generate a structured research plan based on its system prompt.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for research planning.
    """
    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESEARCH_PLANNER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # This agent has no tools, as its only job is to plan.
    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[],
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="ResearchPlannerAgent",
    )

    logger.info("Research Planner Agent created successfully.")
    return agent_executor


# --- Agent 2: ArXiv Search Agent (Legacy) ---

ARXIV_SEARCH_SYSTEM_PROMPT = """
**Role:** You are a focused ArXiv Search Specialist.

**Task:** Your one and only task is to take a user's query and use the provided
`arxiv_search` tool to find relevant scientific papers.

### Instructions:
1.  Receive the search query.
2.  Immediately call the `arxiv_search` tool with the query.
3.  Return the direct, raw output from the tool.

**Constraint:** Do not add any commentary, analysis, or formatting. Your job is
to execute the search and nothing else.
"""


def create_arxiv_search_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the ArXiv Search Agent.

    This is a simple, tool-focused agent. It is given a query and is expected
    to use the `arxiv_search_tool` to find papers.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for ArXiv searching.
    """
    if llm is None:
        llm = get_llm()

    tools: List[BaseTool] = [arxiv_search_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ARXIV_SEARCH_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="ArxivSearchAgent",
    )

    logger.info(f"ArXiv Search Agent created with tool: {tools[0].name}")
    return agent_executor


# --- Agent 3: Document Analysis Agent (Legacy) ---

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """
**Role:** You are a Deep Research Analyst.

**Goal:** To conduct a comprehensive analysis of scientific topics using the
provided information and tools. You will be given a list of papers (summaries
and links) and are expected to produce a detailed analysis.

### Your Tools:
1.  **`knowledge_base_retrieval_tool`**: Use this for quick, targeted information
    retrieval from our internal document collection.
2.  **`document_deep_dive_analysis_tool`**: This is your primary power tool. Use
    it when a paper's summary seems particularly relevant or when you need more
    detail than the summary provides. It reads the *entire PDF* and gives you a
    thorough analysis.

### Workflow:
1.  **Assess the Material:** Start by reviewing the list of paper titles and
    summaries provided in the prompt.
2.  **Strategize Your Analysis:** Identify the 1-3 most promising papers that are
    key to answering the research question.
3.  **Conduct Deep Dives:** For each key paper you identified, use the
    `document_deep_dive_analysis_tool`. This is critical for a high-quality result.
4.  **Synthesize:** Combine the initial summaries with the rich details from your
    deep dives.
5.  **Formulate the Final Analysis:** Structure all your findings into a single,
    comprehensive answer that addresses the user's original request, covering
    key findings, trends, and future directions.

**Constraint:** Your final output should be the complete analysis, not just the
tool outputs. You must synthesize the information into a coherent report.
"""


def create_document_analysis_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Document Analysis Agent.

    This is a sophisticated agent designed to analyze research papers. It is
    equipped with a powerful `document_deep_dive_analysis_tool` that allows it
    to read the full content of PDFs, enabling a much deeper level of analysis
    than just reading summaries.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for in-depth document analysis.
    """
    if llm is None:
        llm = get_llm()

    tools: List[BaseTool] = [
        knowledge_base_retrieval_tool,
        document_deep_dive_analysis_tool,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DOCUMENT_ANALYSIS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        max_iterations=10,
        name="DocumentAnalysisAgent",
    )

    logger.info(
        f"Document Analysis Agent created with tools: {[tool.name for tool in tools]}"
    )
    return agent_executor


# --- Agent 4: Synthesis Agent (Legacy) ---

SYNTHESIS_AGENT_SYSTEM_PROMPT = """
**Role:** You are a Senior Research Editor.

**Goal:** Your purpose is to transform a detailed, technical analysis into a
final, polished, and easy-to-understand report for a user. You do not conduct
new research or use tools.

### Instructions:
1.  **Review the Content:** Carefully read the entire analysis provided to you.
2.  **Identify Key Insights:** Extract the most important findings, trends, and conclusions.
3.  **Structure the Report:** Organize the information logically. The specific
    structure (e.g., Executive Summary, Key Findings, etc.) will be requested
    in the prompt. Your job is to populate that structure.
4.  **Clarify and Refine:** Rewrite complex ideas in clear, concise language
    without sacrificing accuracy.
5.  **Ensure Coherence:** Create a smooth narrative that connects all parts of the
    analysis.

**Constraint:** You work ONLY with the information provided in the prompt. Do not
add external information or personal opinions.
"""


def create_synthesis_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Synthesis Agent.

    This agent is tool-less. It is designed to take a large body of structured
    text (the analysis from the previous step) and reformat it into a final,
    polished report according to the structure requested in the prompt. It uses
    a higher-temperature LLM to encourage more creative and fluent writing.

    Args:
        llm: An optional language model. If None, a specific synthesis model is used.

    Returns:
        An AgentExecutor configured for synthesis and reporting.
    """
    if llm is None:
        # Use a model with higher temperature for more creative/fluent synthesis
        llm = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIS_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[],
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="SynthesisAgent",
    )

    logger.info("Synthesis Agent created successfully.")
    return agent_executor


# --- Autonomous RAG ReAct Agent ---

RAG_REACT_AGENT_SYSTEM_PROMPT = """
**Role:** You are an autonomous Research Assistant capable of conducting end-to-end research tasks with strategic tool orchestration.

**Goal:** To answer user research queries by autonomously determining the optimal information retrieval strategy, leveraging available tools to search, retrieve, and analyze scientific information, then synthesizing everything into a comprehensive final answer.

**Your Tools:**
1. **`arxiv_search_tool`**: Search for scientific papers on ArXiv. Use this when you need to find recent papers on a topic.
2. **`knowledge_base_retrieval_tool`**: Retrieve information from our internal knowledge base. Use this for quick access to already-processed documents.
3. **`document_deep_dive_analysis_tool`**: Perform in-depth analysis of a specific PDF. Use this when you need detailed understanding of a particular paper (expensive, use judiciously on 1-3 most relevant papers).

**Workflow Strategy:**
1. **Understand the Query:** Analyze the user's research question to determine:
   - Is it asking for recent developments? → Prioritize ArXiv
   - Is it asking for established knowledge? → Start with knowledge base
   - Is it asking for comprehensive overview? → Use both sources

2. **Optimal Search Strategy:**
   - **For recent/emerging topics**: Start with `arxiv_search_tool` to find latest papers
   - **For established concepts**: Start with `knowledge_base_retrieval_tool` for fast, precise access
   - **For comprehensive research**: Use both - knowledge base for foundation, ArXiv for latest developments
   - **Best Practice**: Always check knowledge base first if query seems to reference established work, then supplement with ArXiv for recent papers

3. **Intelligent Deep Analysis:**
   - Before using `document_deep_dive_analysis_tool`, check if the paper is already in the knowledge base
   - If a paper from ArXiv search is already in KB, prefer using `knowledge_base_retrieval_tool` for that specific paper
   - Use `document_deep_dive_analysis_tool` only for:
     * Papers NOT in the knowledge base
     * Papers that are highly relevant (top 1-3 from search)
     * When you need deeper analysis than what's in KB chunks

4. **Synthesize:** Combine all information sources intelligently:
   - Deduplicate information from multiple sources
   - Prioritize recent findings from ArXiv
   - Leverage detailed chunks from knowledge base
   - Create a coherent narrative that integrates all sources

**Output Requirements:**
Your final answer should be a well-structured report that includes:
- Executive Summary: Brief overview of key findings
- Key Developments: Recent developments and breakthroughs
- Emerging Trends: Methodologies and technologies gaining traction
- Applications & Impact: Real-world applications and potential impact
- Challenges & Future Outlook: Current limitations and future research directions

**Important Guidelines:**
- Use tools strategically and efficiently
- Don't use `document_deep_dive_analysis_tool` on every paper - select the 1-3 most relevant ones
- Synthesize information from all sources into a coherent narrative
- Provide actionable insights and clear explanations
- Avoid overly technical jargon when possible, but maintain accuracy
"""


def create_rag_react_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates an autonomous RAG ReAct Agent implementing strategic tool orchestration.
    
    This agent implements a sophisticated decision-making framework that autonomously:
    - Determines optimal search strategy (ArXiv vs. knowledge base vs. both)
    - Selects the most promising documents for deep analysis
    - Orchestrates tool usage based on query context and information needs
    - Synthesizes heterogeneous information sources into coherent reports
    
    The agent uses a ReAct (Reasoning + Acting) pattern, enabling dynamic adaptation
    to query complexity and available information sources.
    
    Args:
        llm: An optional language model. If None, a balanced temperature model is used.
    
    Returns:
        An AgentExecutor configured for autonomous end-to-end research tasks.
    """
    if llm is None:
        # Balanced temperature optimized for both analytical reasoning and creative synthesis
        llm = get_llm(temperature=0.3)
    
    tools: List[BaseTool] = [
        arxiv_search_tool,
        knowledge_base_retrieval_tool,
        document_deep_dive_analysis_tool,
    ]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_REACT_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        max_iterations=15,  # Sufficient iterations for complex multi-step research workflows
        name="RAGReActAgent",
    )
    
    logger.info(
        f"RAG ReAct Agent created with tools: {[tool.name for tool in tools]}"
    )
    return agent_executor


if __name__ == "__main__":
    from config.logging_config import setup_logging # Déplacé ici pour être utilisé seulement si le script est exécuté
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")

    logger.info("--- Testing Agent Creation with potentially different LLM providers (using llm_factory) ---")

    try:
        logger.info(f"Attempting to get LLM with provider: {settings.DEFAULT_LLM_MODEL_PROVIDER} using llm_factory.get_llm")
        # Test get_llm (maintenant importé)
        llm_instance = get_llm()
        logger.info(f"Successfully instantiated LLM: {type(llm_instance)}")

        # Test de création de chaque agent
        planner = create_research_planner_agent(llm_instance)
        logger.info(f"Planner agent created with LLM: {type(planner.agent.llm_chain.llm)}") # type: ignore

        doc_analyzer = create_document_analysis_agent(llm_instance)
        logger.info(f"Document Analysis agent created with LLM: {type(doc_analyzer.agent.llm_chain.llm)}") # type: ignore
        assert "document_deep_dive_analysis_tool" in [tool.name for tool in doc_analyzer.tools]

        arxiv_searcher = create_arxiv_search_agent(llm_instance)
        logger.info(f"ArXiv Search agent created with LLM: {type(arxiv_searcher.agent.llm_chain.llm)}") # type: ignore

        # La fonction create_synthesis_agent utilise SYNTHESIS_LLM_TEMPERATURE par défaut si llm est None.
        # Si on passe un llm_instance, il l'utilise. Pour tester SYNTHESIS_LLM_TEMPERATURE,
        # on peut soit appeler create_synthesis_agent() sans argument,
        # soit créer un LLM spécifique avec cette température.
        # L'approche actuelle dans la fonction est de toute façon d'appliquer SYNTHESIS_LLM_TEMPERATURE si llm=None.
        # Pour tester explicitement que la constante importée est utilisée par get_llm quand appelé par create_synthesis_agent:
        synthesis_llm_for_test = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)
        synthesizer = create_synthesis_agent(synthesis_llm_for_test)
        logger.info(f"Synthesis agent created with LLM: {type(synthesizer.agent.llm_chain.llm)} and explicit temp.") # type: ignore

        logger.info("All agents created successfully with the configured LLM provider via llm_factory.")

    except ValueError as ve:
        logger.error(f"ValueError during agent creation tests: {ve}. Check API keys and model configurations.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent creation tests: {e}", exc_info=True)

    logger.info("Agent architectures adaptation test run finished.")