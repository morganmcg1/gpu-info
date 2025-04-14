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
from google import genai
from prompts import (
    INITIAL_SUMMARY_PROMPT,
    ENTITY_IDENTIFICATION_PROMPT,
    SUMMARY_REWRITE_PROMPT,
    ENTITY_DENSITY_EVALUATION_PROMPT,
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
        self.client = genai.Client(api_key=api_key)
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            initial_summary = response.text
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # Parse the response to get the list of entities
            entities = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            new_summary = response.text
            logger.info(f"Generated new summary of {len(new_summary.split())} words")
            return new_summary
        except Exception as e:
            logger.error(f"Error rewriting summary: {str(e)}")
            return current_summary
    
    def evaluate_entity_density(self, summary: str) -> Dict[str, Any]:
        """
        Evaluate the entity density of a summary.
        
        Args:
            summary: The summary to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating entity density")
        
        prompt = ENTITY_DENSITY_EVALUATION_PROMPT.format(summary=summary)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the JSON response
            # For simplicity, we'll just return a basic evaluation
            word_count = len(summary.split())
            return {
                "word_count": word_count,
                "raw_evaluation": response.text
            }
        except Exception as e:
            logger.error(f"Error evaluating entity density: {str(e)}")
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
        
        summaries = []
        
        # Generate initial summary
        current_summary = self.generate_initial_summary(content)
        summaries.append(current_summary)
        
        # Refine the summary through iterations
        for i in range(iterations):
            logger.info(f"Starting iteration {i+1}/{iterations}")
            
            # Identify missing entities
            missing_entities = self.identify_missing_entities(content, current_summary)
            
            # Rewrite the summary
            current_summary = self.rewrite_summary(content, current_summary, missing_entities)
            summaries.append(current_summary)
            
            # Evaluate the new summary
            evaluation = self.evaluate_entity_density(current_summary)
            logger.info(f"Iteration {i+1} summary: {evaluation['word_count']} words")
        
        return summaries
    
    def judge_entity_extraction(self, content: str, summary: str) -> Dict[str, Any]:
        """
        Use Gemini 2.5 Pro as a judge to evaluate entity extraction quality.
        
        Args:
            content: The original content
            summary: The summary to evaluate
            
        Returns:
            Dictionary with judgment results
        """
        logger.info("Judging entity extraction quality")
        
        prompt = SUMMARY_QUALITY_EVALUATION_PROMPT.format(
            content=content,
            summary=summary
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return {"judgment": response.text}
        except Exception as e:
            logger.error(f"Error judging entity extraction: {str(e)}")
            return {"error": str(e)}