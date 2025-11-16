"""
Document Deep Dive Analysis Tool

Tool for performing in-depth analysis of scientific documents using CrewAI.
Includes both the LangChain tool and the CrewAI service implementation.
"""

import logging
import io
import os
from typing import Dict, Any, List

from langchain_core.tools import tool
import requests
from pypdf import PdfReader
# Note: crewai is not in dependencies - install separately if you want to use this tool
# This tool is currently disabled for security reasons (see tools/__init__.py)
try:
    from crewai import Agent, Task, Crew, Process
except ImportError:
    Agent = Task = Crew = Process = None  # Type stubs for when crewai is not installed

from src.services.llm import get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


def _fetch_pdf_content(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    logger.info(f"Fetching PDF content from: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        with io.BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text_content = " ".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        logger.info(f"Successfully extracted text from {pdf_url}")
        return text_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF from {pdf_url}: {e}")
        return f"Error: Failed to download PDF. {e}"
    except Exception as e:
        logger.error(f"Failed to parse PDF from {pdf_url}: {e}", exc_info=True)
        return f"Error: Failed to parse PDF content. {e}"


class DocumentAnalysisService:
    """
    A CrewAI-based document analysis service implementing a complementary two-agent architecture
    for comprehensive analysis of scientific documents.
    
    This design leverages separation of concerns: analytical extraction and critical evaluation
    are handled by one agent, while synthesis and report generation are handled by another,
    each optimized with appropriate LLM temperature settings for their respective tasks.
    """
    
    def __init__(self, document_id: str, document_content: str, research_focus: str):
        """
        Initialize the document analysis service.
        
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
        Create the two complementary agents for document analysis.
        
        Returns:
            List of Agent instances
        """
        agents = []
        
        # Agent 1: Deep Document Analyst
        analyst_role = (
            "You are a Deep Research Analyst specializing in extracting structured information "
            "from scientific documents. Your role is to identify key concepts, methodologies, "
            "findings, and critical details with precision and accuracy."
        )
        
        analyst_goal = (
            "Extract and structure all relevant information from the document, including:\n"
            "- Core contributions and innovations\n"
            "- Methodology and experimental design\n"
            "- Key findings and results\n"
            "- Limitations and future work\n"
            "- Technical details and specifications"
        )
        
        analyst_backstory = (
            "You have extensive experience in scientific document analysis and are known for "
            "your meticulous attention to detail and ability to extract structured information "
            "from complex technical documents."
        )
        
        analyst = Agent(
            role=analyst_role,
            goal=analyst_goal,
            backstory=analyst_backstory,
            llm=self.llm_analyst,
            verbose=False,
            allow_delegation=False,
        )
        agents.append(analyst)
        
        # Agent 2: Research Report Synthesizer
        synthesizer_role = (
            "You are a Research Report Synthesizer specializing in transforming technical "
            "analysis into comprehensive, well-structured reports. Your role is to synthesize "
            "detailed analysis into coherent narratives that are both accurate and accessible."
        )
        
        synthesizer_goal = (
            "Synthesize the detailed analysis into a comprehensive report that:\n"
            "- Presents findings in a clear, logical structure\n"
            "- Integrates all extracted information coherently\n"
            "- Maintains technical accuracy while ensuring readability\n"
            "- Addresses the research focus comprehensively"
        )
        
        synthesizer_backstory = (
            "You are an expert at transforming complex technical information into clear, "
            "comprehensive reports. You excel at creating narratives that weave together "
            "detailed analysis while maintaining accuracy and clarity."
        )
        
        synthesizer = Agent(
            role=synthesizer_role,
            goal=synthesizer_goal,
            backstory=synthesizer_backstory,
            llm=self.llm_synthesizer,
            verbose=False,
            allow_delegation=False,
        )
        agents.append(synthesizer)
        
        return agents

    def _create_tasks(self, agents: List[Agent]) -> List[Task]:
        """
        Create tasks for the document analysis pipeline.
        
        Args:
            agents: List of Agent instances
            
        Returns:
            List of Task instances
        """
        deep_analyst = agents[0]
        report_synthesizer = agents[1]
        
        # Task 1: Deep Analysis
        task_deep_analysis = Task(
            description=(
                f"Analyze the following document with focus on: {self.research_focus}\n\n"
                f"Document ID: {self.document_id}\n\n"
                f"Document Content:\n{self.document_content}\n\n"
                "Extract and structure all relevant information, including:\n"
                "- Core contributions and innovations\n"
                "- Methodology and experimental design\n"
                "- Key findings and results\n"
                "- Limitations and future work\n"
                "- Technical details and specifications\n\n"
                "Provide a detailed, structured analysis that captures all important aspects "
                "of the document."
            ),
            agent=deep_analyst,
            expected_output=(
                "A structured analysis containing:\n"
                "- Executive summary of key points\n"
                "- Detailed methodology analysis\n"
                "- Comprehensive findings extraction\n"
                "- Critical evaluation of contributions\n"
                "- Limitations and future directions"
            ),
        )
        
        # Task 2: Report Synthesis
        task_synthesize_report = Task(
            description=(
                f"Based on the detailed analysis provided, create a comprehensive research report "
                f"that addresses the research focus: {self.research_focus}\n\n"
                "The report should:\n"
                "- Synthesize all extracted information into a coherent narrative\n"
                "- Maintain technical accuracy while ensuring clarity\n"
                "- Structure the content logically\n"
                "- Address the research focus comprehensively\n"
                "- Include all relevant findings, methodologies, and insights"
            ),
            agent=report_synthesizer,
            expected_output=(
                "A comprehensive research report that:\n"
                "- Presents findings in a clear, logical structure\n"
                "- Integrates all analysis components coherently\n"
                "- Addresses the research focus thoroughly\n"
                "- Maintains technical accuracy and clarity"
            ),
            context=[task_deep_analysis],
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
        Process the crew output to extract the final report.
        
        Args:
            output: Raw output from CrewAI
            
        Returns:
            Processed report string
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
            
        if isinstance(output, dict):
            # Try to extract meaningful content from dict
            if 'output' in output:
                return str(output['output'])
            elif 'tasks_output' in output:
                tasks = output['tasks_output']
                if tasks:
                    return str(tasks[-1])
            
        return str(output)


def analyze_document(
    document_id: str,
    document_content: str,
    research_focus: str
) -> str:
    """
    Analyze a document using the CrewAI-based document analysis service.
    
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
        
    service = DocumentAnalysisService(
        document_id=document_id,
        document_content=document_content,
        research_focus=research_focus
    )
    return service.run()


@tool
def document_deep_dive_analysis_tool(
    pdf_url: str, research_focus: str
) -> str:
    """
    Use this powerful tool to perform a comprehensive, deep analysis of a *single*
    scientific paper by providing its PDF URL.

    This tool is "expensive" as it reads and analyzes the entire document, so
    use it judiciously on the most promising papers identified by an ArXiv search.
    It is your best method for understanding the core contributions, methodology,
    and conclusions of a paper.

    Args:
        pdf_url: The direct URL to the PDF of the paper to be analyzed.
        research_focus: A clear, specific question or a set of themes to guide the
                      analysis. For example: "What is the main contribution of this
                      paper?" or "Analyze the methodology and experimental results."

    Returns:
        A structured, in-depth analytical report of the document, or a detailed
        error message if the analysis fails.
    """
    logger.info(
        f"Executing document_deep_dive_analysis_tool: url='{pdf_url}', focus='{research_focus}'"
    )

    document_content = _fetch_pdf_content(pdf_url)
    if document_content.startswith("Error:"):
        return document_content  # Propagate the error message

    if not document_content or not document_content.strip():
        logger.error(f"Could not extract any text content from PDF at {pdf_url}")
        return "Error: Document content is empty or could not be extracted."

    if not research_focus or not research_focus.strip():
        logger.warning("Research focus is empty; analysis may be too generic.")

    try:
        # Extract ArXiv ID from URL for logging and identification
        arxiv_id = pdf_url.split("/")[-1]
        report = analyze_document(
            document_id=arxiv_id,
            document_content=document_content,
            research_focus=research_focus,
        )

        logger.info(f"Analysis completed successfully for doc_id='{arxiv_id}'")
        return report

    except ImportError as e:
        logger.error(f"CrewAI import error: {e}", exc_info=True)
        return f"Error: CrewAI components are not available. {e}"
    except Exception as e:
        logger.error(f"Analysis failed for url='{pdf_url}': {e}", exc_info=True)
        return f"Error during deep dive analysis: {e}"
