"""
Information Extractor Module

This module extracts specific types of information from video content,
such as code examples, equations, key steps, and gotchas.
"""

import logging
from typing import Dict, List, Any
from google import genai

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
        
        prompt = f"""
        Extract all code examples related to CUDA or Triton kernels from the following content.
        Include the full code and a brief description of what each example demonstrates.
        
        Content:
        {content}
        
        For each code example, provide:
        1. The complete code snippet
        2. A description of what the code demonstrates
        3. Any key points or optimizations shown in the code
        
        Format your response as a list of examples, with each example clearly separated.
        Ensure that code formatting is preserved with proper indentation.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # In a real implementation, we would parse the response into structured data
            # For simplicity, we'll return the raw response for now
            return [{"raw_extraction": response.text}]
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
        
        prompt = f"""
        Extract all mathematical equations or formulas related to CUDA or Triton kernels from the following content.
        Include a clear explanation of each equation and its relevance.
        
        Content:
        {content}
        
        For each equation, provide:
        1. The equation itself (in a clear format)
        2. An explanation of what the equation represents
        3. How this equation is relevant to CUDA or Triton kernel programming
        
        Format your response as a list of equations, with each equation clearly separated.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return [{"raw_extraction": response.text}]
        except Exception as e:
            logger.error(f"Error extracting equations: {str(e)}")
            return []
    
    def extract_key_steps(self, content: str) -> List[Dict[str, str]]:
        """
        Extract key steps or processes from the content.
        
        Args:
            content: The content to extract from
            
        Returns:
            List of dictionaries containing key steps with explanations
        """
        logger.info("Extracting key steps")
        
        prompt = f"""
        Extract the key steps or processes for writing effective CUDA or Triton kernels from the following content.
        Focus on practical, actionable steps that developers should follow.
        
        Content:
        {content}
        
        For each key step or process, provide:
        1. A clear title or name for the step
        2. A detailed explanation of what this step involves
        3. Why this step is important
        4. Any tips or best practices related to this step
        
        Format your response as a numbered list of steps, with each step clearly separated.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return [{"raw_extraction": response.text}]
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
        logger.info("Extracting gotchas and warnings")
        
        prompt = f"""
        Extract all gotchas, warnings, and common pitfalls related to CUDA or Triton kernel programming from the following content.
        Focus on issues that developers commonly encounter and how to avoid them.
        
        Content:
        {content}
        
        For each gotcha or warning, provide:
        1. A clear title that describes the issue
        2. A detailed explanation of the problem
        3. Why this is a common issue or mistake
        4. How to avoid or solve this problem
        5. Any specific examples mentioned
        
        Format your response as a list of gotchas, with each gotcha clearly separated.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return [{"raw_extraction": response.text}]
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
        logger.info("Extracting performance optimization tips")
        
        prompt = f"""
        Extract all performance optimization tips and techniques for CUDA or Triton kernels from the following content.
        Focus on specific, actionable advice that can improve kernel performance.
        
        Content:
        {content}
        
        For each performance tip, provide:
        1. A clear title for the optimization technique
        2. A detailed explanation of the technique
        3. When and why this technique is effective
        4. Any benchmarks or performance improvements mentioned
        5. Any trade-offs or considerations to keep in mind
        
        Format your response as a list of performance tips, with each tip clearly separated.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return [{"raw_extraction": response.text}]
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
        logger.info("Extracting all information types")
        
        return {
            "code_examples": self.extract_code_examples(content),
            "equations": self.extract_equations(content),
            "key_steps": self.extract_key_steps(content),
            "gotchas": self.extract_gotchas(content),
            "performance_tips": self.extract_performance_tips(content)
        }