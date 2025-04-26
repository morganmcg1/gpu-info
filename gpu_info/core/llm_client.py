"""
This module provides a centralized client for interacting with the Gemini API.
It handles all LLM calls and provides a consistent interface for the rest of the application.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import google.generativeai as genai
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

class LLMClient:
    """A client for interacting with the Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the LLM client.
        
        Args:
            api_key: The API key for the Gemini API. If None, it will be read from the environment.
            model_name: The name of the model to use.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and GOOGLE_API_KEY environment variable not set")
        
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)
        logger.info(f"Initialized LLMClient with model: {model_name}")
    
    def generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content using the Gemini API.
        
        Args:
            prompt: The prompt to send to the model.
            max_retries: The maximum number of retries in case of failure.
            
        Returns:
            The generated content as a string.
        """
        logger.debug(f"Generating content with prompt: {prompt[:100]}...")
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate content after {max_retries} attempts")
                    raise
        
        # This should never be reached due to the raise in the exception handler
        return ""
    
    def generate_structured_content(self, prompt: str, response_model: Type[T], max_retries: int = 3) -> T:
        """
        Generate structured content using the Gemini API.
        
        Args:
            prompt: The prompt to send to the model.
            response_model: The Pydantic model to parse the response into.
            max_retries: The maximum number of retries in case of failure.
            
        Returns:
            The generated content parsed into the specified Pydantic model.
        """
        logger.debug(f"Generating structured content with prompt: {prompt[:100]}...")
        
        # Add instructions to return JSON in the prompt
        structured_prompt = f"{prompt}\n\nPlease provide your response as a valid JSON object that matches the following structure: {response_model.schema_json()}"
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(structured_prompt)
                
                # Extract JSON from the response
                json_str = self._extract_json_from_text(response.text)
                
                # Parse the JSON into the Pydantic model
                return response_model.parse_raw(json_str)
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate structured content after {max_retries} attempts")
                    raise
        
        # This should never be reached due to the raise in the exception handler
        raise RuntimeError("Failed to generate structured content")
    
    def generate_list_content(self, prompt: str, response_model: Type[T], max_retries: int = 3) -> List[T]:
        """
        Generate a list of structured content using the Gemini API.
        
        Args:
            prompt: The prompt to send to the model.
            response_model: The Pydantic model to parse each item in the response into.
            max_retries: The maximum number of retries in case of failure.
            
        Returns:
            The generated content parsed into a list of the specified Pydantic model.
        """
        logger.debug(f"Generating list content with prompt: {prompt[:100]}...")
        
        # Add instructions to return JSON array in the prompt
        structured_prompt = f"{prompt}\n\nPlease provide your response as a valid JSON array where each item matches the following structure: {response_model.schema_json()}"
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(structured_prompt)
                
                # Extract JSON from the response
                json_str = self._extract_json_from_text(response.text)
                
                # Parse the JSON into a list of Pydantic models
                json_data = json.loads(json_str)
                if not isinstance(json_data, list):
                    json_data = [json_data]
                
                return [response_model.parse_obj(item) for item in json_data]
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate list content after {max_retries} attempts")
                    raise
        
        # This should never be reached due to the raise in the exception handler
        raise RuntimeError("Failed to generate list content")


    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from text that might contain markdown code blocks or other text.
        
        Args:
            text: The text to extract JSON from.
            
        Returns:
            The extracted JSON as a string.
        """
        # Try to extract JSON from markdown code blocks
        import re
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_block_pattern, text)
        
        if matches:
            # Use the first match that parses as valid JSON
            for match in matches:
                try:
                    # Validate that it's proper JSON by parsing it
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found in code blocks, try to find JSON-like content
        # Look for content between curly braces
        brace_pattern = r"(\{[\s\S]*\})"
        brace_matches = re.findall(brace_pattern, text)
        
        if brace_matches:
            # Use the first match that parses as valid JSON
            for match in brace_matches:
                try:
                    # Validate that it's proper JSON by parsing it
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
                    
        # If no valid JSON found in curly braces, look for arrays
        array_pattern = r"(\[[\s\S]*\])"
        array_matches = re.findall(array_pattern, text)
        
        if array_matches:
            # Use the first match that parses as valid JSON
            for match in array_matches:
                try:
                    # Validate that it's proper JSON by parsing it
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        # If we still haven't found valid JSON, return the original text
        # This will likely fail when parsing, but at least we tried
        return text


# Create a singleton instance of the LLM client
_llm_client = None

def get_llm_client(api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro-preview-03-25") -> LLMClient:
    """
    Get the singleton instance of the LLM client.
    
    Args:
        api_key: The API key for the Gemini API. If None, it will be read from the environment.
        model_name: The name of the model to use.
        
    Returns:
        The LLM client instance.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(api_key=api_key, model_name=model_name)
    return _llm_client