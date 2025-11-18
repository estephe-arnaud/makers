"""
Synthesis Evaluator Module

This module provides tools for evaluating the quality of synthesized research content.
It implements a LLM-as-a-Judge approach to assess different aspects of synthesis quality:
- Relevance: How well the synthesis addresses the original query
- Faithfulness: How accurately the synthesis reflects the source context
- Coherence: (Planned future feature) How well-structured and logical the synthesis is

The evaluator uses LangChain's framework and supports multiple LLM providers.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from src.services.llm import get_llm, DEFAULT_LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# --- Data Structures for Evaluation (unchanged) ---
class EvaluationAspectScore(TypedDict):
    """Represents the score and reasoning for a specific evaluation aspect."""
    score: float
    reasoning: str

class SynthesisEvaluationResult(TypedDict):
    """Contains evaluation scores for different aspects of a synthesis."""
    relevance: Optional[EvaluationAspectScore]
    faithfulness: Optional[EvaluationAspectScore]
    coherence: Optional[EvaluationAspectScore]  # Future feature

# --- Prompts for LLM-as-a-Judge (unchanged) ---
RELEVANCE_EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator tasked with assessing the relevance of a generated synthesis to an original user query.
Relevance measures how well the synthesis directly and appropriately addresses the query.

**Original User Query:**
{query}

**Generated Synthesis:**
{synthesis}

**Evaluation Instructions for Relevance:**
1. Carefully read the user query and generated synthesis.
2. Evaluate how well the synthesis addresses the key points of the query.
3. Ignore factual accuracy or writing quality for now, focus only on relevance to the query.
4. Provide a relevance score between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
5. Provide a brief justification (1-2 sentences) for your score.

**Expected Output Format (JSON):**
{{
    "score": <float, ex: 0.8>,
    "reasoning": "<string, your justification>"
}}
"""

FAITHFULNESS_EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator tasked with assessing the faithfulness (factual accuracy) of a generated synthesis against provided source context.
Faithfulness measures whether claims in the synthesis are properly supported by the source context and don't contain information contradicting or hallucinated beyond this context.

**Original User Query (for reference, but evaluation is based on context):**
{query}

**Source Context Provided to Synthesis Agent:**
{context}

**Generated Synthesis (to evaluate for faithfulness TO SOURCE CONTEXT):**
{synthesis}

**Evaluation Instructions for Faithfulness:**
1. Carefully read the source context and generated synthesis.
2. Verify each major claim in the synthesis. Is it directly supported by information in the source context?
3. Identify any information in the synthesis that contradicts the source context or appears to be additional information not present in the context.
4. Provide a faithfulness score between 0.0 (completely unfaithful, many hallucinations or contradictions) and 1.0 (perfectly faithful, all claims supported by context).
5. Provide a brief justification (1-2 sentences) for your score, citing examples if possible.

**Expected Output Format (JSON):**
{{
    "score": <float, ex: 0.9>,
    "reasoning": "<string, your justification with examples if possible>"
}}
"""

class SynthesisEvaluator:
    """
    Evaluates the quality of research content synthesis using LLM-as-a-Judge approach.
    
    This class provides methods to evaluate different aspects of synthesis quality:
    - Relevance to original query
    - Faithfulness to source context
    - (Future) Coherence of the synthesis
    """
    
    def __init__(
        self,
        judge_llm_provider: Optional[str] = None,
        judge_llm_model_name: Optional[str] = None,
        # Uses DEFAULT_LLM_TEMPERATURE imported from llm factory
        judge_llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    ):
        """
        Initialize the synthesis evaluator.
        
        Args:
            judge_llm_provider: Optional override for the LLM provider
            judge_llm_model_name: Optional override for the LLM model name
            judge_llm_temperature: Temperature setting for the judge LLM
        """
        self.judge_llm_provider_init = judge_llm_provider
        self.judge_llm_model_name_init = judge_llm_model_name
        self.judge_llm_temperature = judge_llm_temperature
        self.judge_llm: BaseLanguageModel = self._get_judge_llm()
        logger.info(f"SynthesisEvaluator initialized with judge LLM type: {type(self.judge_llm)}")

    def _get_judge_llm(self) -> BaseLanguageModel:
        """
        Get the judge LLM instance using the centralized get_llm function.
        
        Returns:
            BaseLanguageModel: The initialized LLM instance for evaluation
            
        Raises:
            ValueError: If LLM initialization fails
        """
        try:
            return get_llm(
                temperature=self.judge_llm_temperature,
                model_provider_override=self.judge_llm_provider_init,
                model_name_override=self.judge_llm_model_name_init
            )
        except ValueError as e:
            logger.error(f"Failed to initialize Judge LLM for SynthesisEvaluator: {e}")
            raise

    async def _evaluate_aspect(
        self,
        prompt_template_str: str,
        query: str,
        synthesis: str,
        context: Optional[str] = None
    ) -> Optional[EvaluationAspectScore]:
        """
        Evaluate a specific aspect of the synthesis using the provided prompt template.
        
        Args:
            prompt_template_str: The prompt template for the evaluation
            query: The original user query
            synthesis: The generated synthesis to evaluate
            context: Optional source context for faithfulness evaluation
            
        Returns:
            Optional[EvaluationAspectScore]: The evaluation score and reasoning, or None if evaluation fails
        """
        if not self.judge_llm:
            logger.error("Judge LLM not initialized for aspect evaluation.")
            return None

        prompt_inputs = {"query": query, "synthesis": synthesis}
        if "{context}" in prompt_template_str:
            if context is None:
                logger.warning("Context required for evaluation aspect but not provided. Skipping this aspect.")
                return None
            prompt_inputs["context"] = context

        eval_prompt = ChatPromptTemplate.from_template(prompt_template_str)
        
        current_provider = self.judge_llm_provider_init or settings.DEFAULT_LLM_MODEL_PROVIDER
        supports_json_mode = current_provider.lower() in ["openai", "ollama"]

        if hasattr(self.judge_llm, 'bind') and supports_json_mode:
            try:
                judge_llm_with_json_mode = self.judge_llm.bind(
                    response_format={"type": "json_object"}
                )
                chain_with_json_mode = eval_prompt | judge_llm_with_json_mode | JsonOutputParser()
                response_dict = await chain_with_json_mode.ainvoke(prompt_inputs)
                logger.debug(f"JSON mode response for aspect: {response_dict}")
            except Exception as e_json_bind:
                logger.warning(f"Failed to use JSON mode with LLM {type(self.judge_llm)} (provider: {current_provider}), possibly not supported or model error: {e_json_bind}. Falling back to standard parsing.")
                chain = eval_prompt | self.judge_llm | JsonOutputParser()
                response_dict = await chain.ainvoke(prompt_inputs)
        else:
            logger.info(f"LLM {type(self.judge_llm)} (provider: {current_provider}) may not support native JSON mode binding or not attempted. Relying on prompt for JSON output.")
            chain = eval_prompt | self.judge_llm | JsonOutputParser()
            response_dict = await chain.ainvoke(prompt_inputs)

        try:
            if isinstance(response_dict, dict) and "score" in response_dict and "reasoning" in response_dict:
                return EvaluationAspectScore(score=float(response_dict["score"]), reasoning=str(response_dict["reasoning"]))
            else:
                logger.error(f"Failed to parse valid score/reasoning from Judge LLM response: {response_dict}")
                return None
        except Exception as e:
            logger.error(f"Error processing LLM-as-a-Judge response for an aspect: {e}", exc_info=True)
            return None

    async def evaluate_relevance(self, query: str, synthesis: str) -> Optional[EvaluationAspectScore]:
        """
        Evaluate the relevance of the synthesis to the original query.
        
        Args:
            query: The original user query
            synthesis: The generated synthesis to evaluate
            
        Returns:
            Optional[EvaluationAspectScore]: The relevance score and reasoning
        """
        logger.info(f"Evaluating relevance for query: '{query[:50]}...'")
        return await self._evaluate_aspect(RELEVANCE_EVAL_PROMPT_TEMPLATE, query, synthesis)

    async def evaluate_faithfulness(self, query: str, synthesis: str, context: str) -> Optional[EvaluationAspectScore]:
        """
        Evaluate the faithfulness of the synthesis to the source context.
        
        Args:
            query: The original user query (for reference)
            synthesis: The generated synthesis to evaluate
            context: The source context to evaluate against
            
        Returns:
            Optional[EvaluationAspectScore]: The faithfulness score and reasoning
        """
        logger.info(f"Evaluating faithfulness for query: '{query[:50]}...' against context (len: {len(context)}).")
        if not context or not context.strip():
            logger.warning("Context is empty or whitespace-only. Faithfulness evaluation will be skipped or unreliable.")
            return EvaluationAspectScore(score=0.0, reasoning="Context was not provided or was empty.")
        return await self._evaluate_aspect(FAITHFULNESS_EVAL_PROMPT_TEMPLATE, query, synthesis, context=context)

    async def evaluate_synthesis(
        self,
        query: str,
        synthesis: str,
        context: str
    ) -> SynthesisEvaluationResult:
        """
        Perform a complete evaluation of the synthesis.
        
        Args:
            query: The original user query
            synthesis: The generated synthesis to evaluate
            context: The source context for faithfulness evaluation
            
        Returns:
            SynthesisEvaluationResult: Complete evaluation results including relevance and faithfulness scores
        """
        logger.info(f"Starting synthesis evaluation for query: '{query[:50]}...'")

        relevance_score = await self.evaluate_relevance(query, synthesis)
        faithfulness_score = await self.evaluate_faithfulness(query, synthesis, context)

        result: SynthesisEvaluationResult = {
            "relevance": relevance_score,
            "faithfulness": faithfulness_score,
            "coherence": None,  # Future feature
        }
        logger.info(f"Synthesis evaluation completed. Relevance: {relevance_score}, Faithfulness: {faithfulness_score}")
        return result

    def print_results(self, results: SynthesisEvaluationResult, query: Optional[str] = None):
        """
        Print the evaluation results in a human-readable format.
        
        Args:
            results: The evaluation results to print
            query: Optional original query for context
        """
        print("\n--- Synthesis Evaluation Results ---")
        if query:
            print(f"For Query: {query}")

        if results["relevance"]:
            print(f"  Relevance Score: {results['relevance']['score']:.2f}")
            print(f"    Reasoning: {results['relevance']['reasoning']}")
        else:
            print("  Relevance: Not Evaluated / Error")

        if results["faithfulness"]:
            print(f"  Faithfulness Score: {results['faithfulness']['score']:.2f}")
            print(f"    Reasoning: {results['faithfulness']['reasoning']}")
        else:
            print("  Faithfulness: Not Evaluated / Error")
        print("------------------------------------")