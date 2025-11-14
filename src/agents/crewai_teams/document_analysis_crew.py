"""
Document Analysis Crew Module

This module implements a CrewAI-based document analysis system that uses a complementary
two-agent architecture to perform comprehensive analysis of scientific documents. The crew consists of:
- Deep Document Analyst: Extracts structured information and performs critical analysis
- Research Report Synthesizer: Synthesizes the analysis into a comprehensive final report

The module implements a sequential pipeline with clear separation of concerns, where each agent
is optimized for its specific role (analytical precision vs. creative synthesis).
"""

import os
import logging
from typing import List, Dict, Any, Optional

from crewai import Agent, Task, Crew, Process
# MODIFICATION: Suppression de l'import de CrewOutput car le chemin semble incorrect ou non nécessaire
# from crewai.outputs import CrewOutput 

from src.llm_services.llm_factory import get_llm 
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentAnalysisCrew:
    """
    A CrewAI-based document analysis system implementing a complementary two-agent architecture
    for comprehensive analysis of scientific documents.
    
    This design leverages separation of concerns: analytical extraction and critical evaluation
    are handled by one agent, while synthesis and report generation are handled by another,
    each optimized with appropriate LLM temperature settings for their respective tasks.
    """
    
    def __init__(self, document_id: str, document_content: str, research_focus: str):
        """
        Initialize the document analysis crew.
        
        Args:
            document_id: Unique identifier for the document
            document_content: Full text content to analyze
            research_focus: Specific focus or questions for the analysis
        """
        self.document_id = document_id
        self.document_content = document_content
        self.research_focus = research_focus
        
        try:
            # Deep Document Analyst: Lower temperature for analytical precision and factual accuracy
            self.llm_analyst = get_llm(temperature=0.2)
            # Research Report Synthesizer: Higher temperature for creative synthesis and narrative flow
            self.llm_synthesizer = get_llm(temperature=0.4)
        except ValueError as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning("Falling back to CrewAI's default LLM if OPENAI_API_KEY is set")
            self.llm_analyst = None
            self.llm_synthesizer = None

    def _create_agents(self) -> List[Agent]:
        """
        Create the complementary two-agent team for document analysis.
        
        The architecture implements clear separation of concerns:
        - Analytical agent focuses on extraction and critical evaluation
        - Synthesizer agent focuses on report compilation and narrative structure
        
        Returns:
            List of configured CrewAI agents with specialized roles
        """
        # Agent 1: Deep Document Analyst
        # Combines extraction and critical analysis for technical depth
        deep_analyst = Agent(
            role='Deep Document Analyst',
            goal=(
                f"Extract structured information and perform critical analysis of the document "
                f"focusing on '{self.research_focus}'. Your analysis should cover methodology, "
                f"datasets, results, limitations, strengths, weaknesses, and key contributions."
            ),
            backstory=(
                "You are an experienced researcher with deep expertise in scientific literature analysis. "
                "You excel at extracting structured information from complex documents and performing "
                "rigorous critical evaluation. Your analytical skills help identify key innovations, "
                "methodological strengths, and potential limitations in scientific research."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm_analyst
        )
        
        # Agent 2: Research Report Synthesizer
        # Transforms technical analysis into a well-structured, accessible report
        report_synthesizer = Agent(
            role='Research Report Synthesizer',
            goal=(
                f"Synthesize the technical analysis into a comprehensive, well-structured report "
                f"for document {self.document_id} focusing on '{self.research_focus}'. "
                f"Organize the information clearly and make it accessible."
            ),
            backstory=(
                "You are a senior scientific writer and editor with expertise in synthesizing "
                "complex technical analyses into clear, comprehensive reports. You excel at "
                "structuring information logically, creating coherent narratives, and making "
                "sophisticated research accessible to diverse audiences."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm_synthesizer
        )
        
        return [deep_analyst, report_synthesizer]

    def _create_tasks(self, agents: List[Agent]) -> List[Task]:
        """
        Create the sequence of tasks for document analysis.
        
        Args:
            agents: List of agents to assign tasks to
            
        Returns:
            List of configured CrewAI tasks
        """
        deep_analyst, report_synthesizer = agents
        
        # Task 1: Deep Analysis (Extraction + Critical Analysis)
        task_deep_analysis = Task(
            description=(
                f"Perform a comprehensive analysis of document {self.document_id} focusing on '{self.research_focus}'.\n\n"
                f"**Document Content:**\n---\n{self.document_content}\n---\n\n"
                f"**Your tasks:**\n"
                f"1. Extract structured information:\n"
                f"   - Methodology and experimental design\n"
                f"   - Datasets and data sources\n"
                f"   - Key results and findings\n"
                f"   - Limitations and constraints\n"
                f"2. Perform critical analysis:\n"
                f"   - Identify strengths and innovations\n"
                f"   - Evaluate weaknesses and potential issues\n"
                f"   - Assess contributions and significance\n"
                f"   - Highlight key concepts and techniques\n\n"
                f"Focus specifically on aspects related to: {self.research_focus}"
            ),
            expected_output=(
                "A comprehensive technical analysis containing:\n"
                "- Structured extraction of methodology, datasets, results, and limitations\n"
                "- Critical evaluation of strengths, weaknesses, and contributions\n"
                "- Identification of key innovations and significant findings"
            ),
            agent=deep_analyst
        )
        
        # Task 2: Report Synthesis (Compilation)
        task_synthesize_report = Task(
            description=(
                f"Synthesize the technical analysis into a comprehensive, well-structured report "
                f"for document {self.document_id} focusing on '{self.research_focus}'.\n\n"
                f"**Instructions:**\n"
                f"1. Review the detailed technical analysis from the previous task\n"
                f"2. Organize the information into a clear, logical structure\n"
                f"3. Create a comprehensive report that includes:\n"
                f"   - Executive Summary: Brief overview of key findings\n"
                f"   - Key Information: Methodology, datasets, results, limitations\n"
                f"   - Critical Analysis: Strengths, weaknesses, contributions\n"
                f"   - Key Insights: Innovations, significant findings, and implications\n"
                f"4. Ensure the report is well-written, accessible, and addresses the research focus: {self.research_focus}"
            ),
            expected_output=(
                "A comprehensive, well-structured analytical report covering:\n"
                "1. Executive Summary\n"
                "2. Key Information (Methodology, Results, Limitations)\n"
                "3. Critical Analysis (Strengths, Weaknesses, Contributions)\n"
                "4. Key Insights and Implications"
            ),
            agent=report_synthesizer,
            context=[task_deep_analysis]
        )
        
        return [task_deep_analysis, task_synthesize_report]

    def run(self) -> str:
        """
        Execute the document analysis process.
        
        Returns:
            Final analytical report or error message
        """
        if self.llm_analyst is None and not os.environ.get("OPENAI_API_KEY"):
            error_msg = "LLM not configured. Set API keys via .env or OPENAI_API_KEY"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        logger.info(f"Starting analysis for document {self.document_id}")
        
        try:
            # Initialize and run the crew
            crew = Crew(
                agents=self._create_agents(),
                tasks=self._create_tasks(self._create_agents()),
                process=Process.sequential,
                verbose=False
            )
            
            # Execute analysis and process results
            crew_output = crew.kickoff()
            final_report = self._process_crew_output(crew_output)
            
            logger.info(f"Analysis completed for document {self.document_id}")
            return final_report
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    def _process_crew_output(self, output: Any) -> str:
        """
        Process the crew's output into a final report.
        
        Args:
            output: Raw output from the crew
            
        Returns:
            Processed report text
        """
        if not output:
            logger.warning("Crew returned no output")
            return "Error: No analysis results available"
            
        # Try different output formats
        if hasattr(output, 'raw') and output.raw:
            return str(output.raw)
            
        if hasattr(output, 'tasks_output') and output.tasks_output:
            last_task = output.tasks_output[-1]
            if hasattr(last_task, 'exported_output') and last_task.exported_output:
                return str(last_task.exported_output)
            if hasattr(last_task, 'raw_output') and last_task.raw_output:
                return str(last_task.raw_output)
            return str(last_task)
            
        if hasattr(output, 'data') and hasattr(output.data, 'output'):
            return str(output.data.output)
            
        return str(output)

def run_document_deep_dive_crew(
    document_id: str,
    document_content: str,
    research_focus: str
) -> str:
    """
    Run a deep dive analysis on a document using the DocumentAnalysisCrew.
    
    Args:
        document_id: Unique identifier for the document
        document_content: Full text content to analyze
        research_focus: Specific focus or questions for the analysis
        
    Returns:
        Analysis report or error message
    """
    if not document_content or not document_content.strip():
        logger.warning(f"Empty document content for {document_id}")
        return "Error: Document content is empty"
        
    crew = DocumentAnalysisCrew(
        document_id=document_id,
        document_content=document_content,
        research_focus=research_focus
    )
    return crew.run()

def _can_run_crew_test() -> bool:
    """
    Check if the crew test can be run with current configuration.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        provider = settings.DEFAULT_LLM_MODEL_PROVIDER.lower()
        
        if provider == "openai" and settings.OPENAI_API_KEY:
            return True
        if provider == "huggingface_api" and settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_REPO_ID:
            return True
        if provider == "ollama" and settings.OLLAMA_BASE_URL and settings.OLLAMA_GENERATIVE_MODEL_NAME:
            return True
        if os.environ.get("OPENAI_API_KEY"):
            logger.warning("Using OPENAI_API_KEY from environment")
            return True
            
    except Exception as e:
        logger.warning(f"Configuration check failed: {e}")
        return bool(os.environ.get("OPENAI_API_KEY"))
        
    return False

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")
    
    logger.info("Testing Document Analysis Crew")
    
    if not _can_run_crew_test():
        logger.error("Cannot run test: LLM configuration missing")
    else:
        test_doc = {
            "id": "test_001",
            "content": "Sample document about methodology, results, and limitations.",
            "focus": "Extract methodology, results, and limitations"
        }
        
        report = run_document_deep_dive_crew(
            document_id=test_doc["id"],
            document_content=test_doc["content"],
            research_focus=test_doc["focus"]
        )
        
        print("\nAnalysis Report:")
        print(report)
        print("-" * 50)
    
    logger.info("Test completed")