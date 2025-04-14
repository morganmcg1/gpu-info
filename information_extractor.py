"""
Information Extractor Module

This module extracts specific types of information from video content,
such as code examples, equations, key steps, and gotchas.
"""

import logging
from typing import Dict, List, Any
from google import genai
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
        self.client = genai.Client(api_key=api_key)
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll just return the raw text
            return [{"raw_examples": response.text}]
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll just return the raw text
            return [{"raw_equations": response.text}]
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll just return the raw text
            return [{"raw_steps": response.text}]
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll just return the raw text
            return [{"raw_gotchas": response.text}]
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll just return the raw text
            return [{"raw_tips": response.text}]
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