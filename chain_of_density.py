"""
Chain of Density Module

This module implements the Chain of Density method for refining summaries.
It uses the Gemini API to iteratively improve summaries by adding more entities
while maintaining the same length.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any

from llm_client import get_llm_client
from models import InitialSummary, EntityList, DenseSummary, SummaryEvaluation
from prompts import (
    INITIAL_SUMMARY_PROMPT,
    ENTITY_IDENTIFICATION_PROMPT,
    SUMMARY_REWRITE_PROMPT,
    ENTITY_DENSITY_EVALUATION_PROMPT,  # Keep for backward compatibility
    SUMMARY_QUALITY_EVALUATION_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChainOfDensity:
    """Class to implement the Chain of Density summarization method."""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the Chain of Density processor.
        
        Args:
            api_key: Google API key for Gemini
            model_id: Gemini model ID to use
        """
        self.model_id = model_id
        self.llm_client = get_llm_client(api_key=api_key, model_name=model_id)
        logger.info(f"Initialized ChainOfDensity with model: {model_id}")
    
    def generate_initial_summary(self, content: str) -> str:
        """
        Generate an initial summary of the content.
        
        Args:
            content: The content to summarize
            
        Returns:
            Initial summary
        """
        logger.info("Generating initial summary")
        
        prompt = INITIAL_SUMMARY_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                initial_summary_model = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=InitialSummary
                )
                initial_summary = initial_summary_model.summary
            except Exception as e:
                logger.warning(f"Error generating structured initial summary: {str(e)}. Falling back to unstructured output.")
                # Fall back to unstructured output
                initial_summary = self.llm_client.generate_content(prompt=prompt)
            
            logger.info(f"Generated initial summary of {len(initial_summary.split())} words")
            return initial_summary
        except Exception as e:
            logger.error(f"Error generating initial summary: {str(e)}")
            raise
    
    def identify_missing_entities(self, content: str, current_summary: str) -> List[str]:
        """
        Identify entities from the content that are missing from the current summary.
        
        Args:
            content: The original content
            current_summary: The current summary
            
        Returns:
            List of missing entities
        """
        logger.info("Identifying missing entities")
        
        prompt = ENTITY_IDENTIFICATION_PROMPT.format(
            content=content,
            current_summary=current_summary
        )
        
        try:
            # Try to use structured output
            try:
                entity_list = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=EntityList
                )
                entities = entity_list.entities
            except Exception as e:
                logger.warning(f"Error generating structured entity list: {str(e)}. Falling back to unstructured output.")
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                # Parse the response to get the list of entities
                entities = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            logger.info(f"Identified {len(entities)} missing entities")
            return entities
        except Exception as e:
            logger.error(f"Error identifying missing entities: {str(e)}")
            return []
    
    def rewrite_summary(self, content: str, current_summary: str, missing_entities: List[str]) -> str:
        """
        Rewrite the summary to include the missing entities.
        
        Args:
            content: The original content
            current_summary: The current summary
            missing_entities: List of entities to include
            
        Returns:
            Rewritten summary
        """
        logger.info(f"Rewriting summary to include {len(missing_entities)} new entities")
        
        entities_text = "\n".join([f"- {entity}" for entity in missing_entities])
        
        prompt = SUMMARY_REWRITE_PROMPT.format(
            content=content,
            current_summary=current_summary,
            entities_text=entities_text,
            len_current_summary=len(current_summary.split())
        )
        
        try:
            # Try to use structured output
            try:
                dense_summary = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=DenseSummary
                )
                new_summary = dense_summary.summary
            except Exception as e:
                logger.warning(f"Error generating structured dense summary: {str(e)}. Falling back to unstructured output.")
                # Fall back to unstructured output
                new_summary = self.llm_client.generate_content(prompt=prompt)
            
            logger.info(f"Generated new summary of {len(new_summary.split())} words")
            return new_summary
        except Exception as e:
            logger.error(f"Error rewriting summary: {str(e)}")
            return current_summary
    
    def evaluate_summary_quality(self, summary: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the technical density and specificity of the summary.
        
        Args:
            summary: The summary to evaluate
            content: (Optional) Original content for context if needed for evaluation
            
        Returns:
            Dictionary with qualitative evaluation
        """
        logger.info("Evaluating summary quality")
        
        # Construct prompt, including content if provided
        prompt_parts = [
            "Evaluate the technical density and specificity of the following summary regarding writing, running, profiling, or understanding CUDA/Triton kernels and GPU concepts.\n",
            "Does it contain actionable, concrete technical information (specific function names, tool names, code patterns, architectural details, performance concepts)?\n",
            "Identify 1-2 areas where the summary could be *more specific* or add *more technical detail* based on common knowledge about this topic or the provided original content (if available).\n\n"
        ]
        if content:
            prompt_parts.extend(["Original Content Context:\n---\n", content, "\n---\n\n"])
        prompt_parts.extend(["Summary to Evaluate:\n---\n", summary, "\n---\n\n", "Evaluation:"])
        prompt = "".join(prompt_parts)
        
        try:
            # Try to use structured output
            try:
                evaluation = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=SummaryEvaluation
                )
                
                return {
                    "word_count": len(summary.split()),
                    "entity_coverage_score": evaluation.entity_coverage_score,
                    "accuracy_score": evaluation.accuracy_score,
                    "conciseness_score": evaluation.conciseness_score,
                    "overall_quality_score": evaluation.overall_quality_score,
                    "missing_critical_entities": evaluation.missing_critical_entities,
                    "irrelevant_or_vague_parts": evaluation.irrelevant_or_vague_parts,
                    "improvement_suggestions": evaluation.improvement_suggestions
                }
            except Exception as e:
                logger.warning(f"Error generating structured evaluation: {str(e)}. Falling back to unstructured output.")
                # Fall back to unstructured output
                quality_assessment = self.llm_client.generate_content(prompt=prompt)
                
                return {
                    "word_count": len(summary.split()),
                    "quality_assessment": quality_assessment
                }
        except Exception as e:
            logger.error(f"Error evaluating summary quality: {str(e)}")
            return {"word_count": len(summary.split()), "error": str(e)}
    
    def apply_chain_of_density(self, content: str, iterations: int = 3) -> List[str]:
        """
        Apply the Chain of Density method to generate increasingly dense summaries.
        
        Args:
            content: The content to summarize
            iterations: Number of refinement iterations
            
        Returns:
            List of summaries, with the last one being the most dense
        """
        logger.info(f"Applying Chain of Density with {iterations} iterations")
        
        if not content:
            logger.error("Content cannot be empty.")
            return []
            
        summaries = []
        
        try:
            # Generate initial summary
            current_summary = self.generate_initial_summary(content)
            if not current_summary:  # Handle potential empty initial summary
                logger.error("Initial summary generation failed.")
                return []
            summaries.append(current_summary)
            
            # Refine the summary through iterations
            for i in range(iterations):
                iteration_num = i + 1
                logger.info(f"--- Starting Iteration {iteration_num}/{iterations} ---")
                
                # Identify missing entities
                missing_entities = self.identify_missing_entities(content, current_summary)
                if not missing_entities:
                    logger.info(f"No missing entities identified in iteration {iteration_num}. Stopping refinement.")
                    break  # Stop if no more entities are found
                
                # Rewrite the summary
                current_summary = self.rewrite_summary(content, current_summary, missing_entities)
                summaries.append(current_summary)
                
                # Log summary word count
                logger.info(f"Iteration {iteration_num} summary word count: {len(current_summary.split())}")
                
                # Optional: Evaluate quality at each step (can be slow/costly)
                # evaluation = self.evaluate_summary_quality(current_summary, content)
                # logger.info(f"Iteration {iteration_num} evaluation: {evaluation}")
                
        except Exception as e:
            logger.error(f"Chain of Density process failed during execution: {str(e)}")
            # Return whatever summaries were generated before the error
            return summaries
            
        logger.info("Chain of Density process completed.")
        return summaries
    
    def judge_entity_extraction(self, content: str, summary: str) -> Dict[str, Any]:
        """
        Use the LLM as an expert judge for final evaluation against the original content.
        
        Args:
            content: The original content
            summary: The final summary to evaluate
            
        Returns:
            Dictionary with judgment results
        """
        logger.info("Judging final entity extraction quality")
        
        prompt = SUMMARY_QUALITY_EVALUATION_PROMPT.format(
            content=content,
            summary=summary
        )
        
        # Specify JSON output if using a model version that reliably supports it
        generation_config = {
            "response_mime_type": "application/json"
        }
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                generation_config=generation_config
            )
            
            # Attempt to parse the JSON response
            try:
                judgment_text = response.text
                judgment_json = json.loads(judgment_text)
                return judgment_json
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON judgment: {response.text}")
                return {"judgment_text": response.text, "parsing_error": True}  # Return raw text if JSON fails
        except Exception as e:
            logger.error(f"Error judging entity extraction: {str(e)}")
            return {"error": str(e)}