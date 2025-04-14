"""
Information Extractor Module

This module extracts specific types of information from video content,
such as code examples, equations, key steps, and gotchas.
"""

import logging
from typing import Dict, List, Any

from llm_client import get_llm_client
from models import CodeExample, Equation, ImplementationStep, Gotcha, PerformanceTip, ExtractedInformation
from prompts import (
    CODE_EXAMPLES_EXTRACTION_PROMPT,
    EQUATIONS_EXTRACTION_PROMPT,
    STEPS_EXTRACTION_PROMPT,
    GOTCHAS_EXTRACTION_PROMPT,
    PERFORMANCE_TIPS_EXTRACTION_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InformationExtractor:
    """Class to extract specific types of information from content."""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the Information Extractor.
        
        Args:
            api_key: Google API key for Gemini
            model_id: Gemini model ID to use
        """
        self.model_id = model_id
        self.llm_client = get_llm_client(api_key=api_key, model_name=model_id)
        logger.info(f"Initialized InformationExtractor with model: {model_id}")
    
    def extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """
        Extract code examples from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing code examples with descriptions
        """
        logger.info("Extracting code examples")
        
        prompt = CODE_EXAMPLES_EXTRACTION_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                extracted_info = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=ExtractedInformation
                )
                
                # Convert Pydantic models to dictionaries
                examples = []
                for example in extracted_info.code_examples:
                    examples.append({
                        "title": example.title,
                        "description": example.description,
                        "code": example.code,
                        "language": example.language
                    })
                
                logger.info(f"Extracted {len(examples)} code examples using structured output")
                return examples
            except Exception as e:
                logger.warning(f"Error generating structured code examples: {str(e)}. Falling back to unstructured output.")
                
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                
                # For simplicity, we'll just return the raw text
                return [{"raw_examples": response}]
        except Exception as e:
            logger.error(f"Error extracting code examples: {str(e)}")
            return []
    
    def extract_equations(self, content: str) -> List[Dict[str, str]]:
        """
        Extract mathematical equations from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing equations with explanations
        """
        logger.info("Extracting equations")
        
        prompt = EQUATIONS_EXTRACTION_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                extracted_info = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=ExtractedInformation
                )
                
                # Convert Pydantic models to dictionaries
                equations = []
                for equation in extracted_info.equations:
                    equations.append({
                        "equation": equation.equation,
                        "latex": equation.latex,
                        "explanation": equation.explanation,
                        "context": equation.context
                    })
                
                logger.info(f"Extracted {len(equations)} equations using structured output")
                return equations
            except Exception as e:
                logger.warning(f"Error generating structured equations: {str(e)}. Falling back to unstructured output.")
                
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                
                # For simplicity, we'll just return the raw text
                return [{"raw_equations": response}]
        except Exception as e:
            logger.error(f"Error extracting equations: {str(e)}")
            return []
    
    def extract_key_steps(self, content: str) -> List[Dict[str, str]]:
        """
        Extract key steps or processes from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing steps with explanations
        """
        logger.info("Extracting key steps")
        
        prompt = STEPS_EXTRACTION_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                extracted_info = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=ExtractedInformation
                )
                
                # Convert Pydantic models to dictionaries
                steps = []
                for step in extracted_info.implementation_steps:
                    steps.append({
                        "step_number": step.step_number,
                        "title": step.title,
                        "description": step.description,
                        "code": step.code if step.code else "",
                        "importance": step.importance
                    })
                
                logger.info(f"Extracted {len(steps)} key steps using structured output")
                return steps
            except Exception as e:
                logger.warning(f"Error generating structured key steps: {str(e)}. Falling back to unstructured output.")
                
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                
                # For simplicity, we'll just return the raw text
                return [{"raw_steps": response}]
        except Exception as e:
            logger.error(f"Error extracting key steps: {str(e)}")
            return []
    
    def extract_gotchas(self, content: str) -> List[Dict[str, str]]:
        """
        Extract gotchas and warnings from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing gotchas with explanations
        """
        logger.info("Extracting gotchas")
        
        prompt = GOTCHAS_EXTRACTION_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                extracted_info = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=ExtractedInformation
                )
                
                # Convert Pydantic models to dictionaries
                gotchas = []
                for gotcha in extracted_info.gotchas:
                    gotchas.append({
                        "title": gotcha.title,
                        "description": gotcha.description,
                        "impact": gotcha.impact,
                        "solution": gotcha.solution
                    })
                
                logger.info(f"Extracted {len(gotchas)} gotchas using structured output")
                return gotchas
            except Exception as e:
                logger.warning(f"Error generating structured gotchas: {str(e)}. Falling back to unstructured output.")
                
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                
                # For simplicity, we'll just return the raw text
                return [{"raw_gotchas": response}]
        except Exception as e:
            logger.error(f"Error extracting gotchas: {str(e)}")
            return []
    
    def extract_performance_tips(self, content: str) -> List[Dict[str, str]]:
        """
        Extract performance optimization tips from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing performance tips with explanations
        """
        logger.info("Extracting performance tips")
        
        prompt = PERFORMANCE_TIPS_EXTRACTION_PROMPT.format(content=content)
        
        try:
            # Try to use structured output
            try:
                extracted_info = self.llm_client.generate_structured_content(
                    prompt=prompt,
                    response_model=ExtractedInformation
                )
                
                # Convert Pydantic models to dictionaries
                tips = []
                for tip in extracted_info.performance_tips:
                    tips.append({
                        "title": tip.title,
                        "description": tip.description,
                        "impact": tip.impact,
                        "implementation": tip.implementation,
                        "code_example": tip.code_example if tip.code_example else ""
                    })
                
                logger.info(f"Extracted {len(tips)} performance tips using structured output")
                return tips
            except Exception as e:
                logger.warning(f"Error generating structured performance tips: {str(e)}. Falling back to unstructured output.")
                
                # Fall back to unstructured output
                response = self.llm_client.generate_content(prompt=prompt)
                
                # For simplicity, we'll just return the raw text
                return [{"raw_tips": response}]
        except Exception as e:
            logger.error(f"Error extracting performance tips: {str(e)}")
            return []
    
    def extract_all_information(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract all types of information from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            Dictionary containing all extracted information
        """
        logger.info("Extracting all information")
        
        return {
            "code_examples": self.extract_code_examples(content),
            "equations": self.extract_equations(content),
            "key_steps": self.extract_key_steps(content),
            "gotchas": self.extract_gotchas(content),
            "performance_tips": self.extract_performance_tips(content)
        }