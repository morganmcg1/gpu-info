"""
Gemini Video Processor Module

This module uses the Gemini API to directly process YouTube videos.
It handles the direct video URL input to Gemini for processing.
"""

import logging
import time
from typing import Dict, Optional, Any
from google import genai
from prompts import (
    BASIC_VIDEO_SUMMARY_PROMPT,
    DETAILED_VIDEO_ANALYSIS_PROMPT,
    CONTENT_RELEVANCE_CHECK_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiVideoProcessor:
    """Class to process YouTube videos directly with Gemini."""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the Gemini Video Processor.
        
        Args:
            api_key: Google API key for Gemini
            model_id: Gemini model ID to use
        """
        self.model_id = model_id
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized GeminiVideoProcessor with model: {model_id}")
    
    def check_content_relevance(self, video_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Check if the video content is relevant to CUDA or Triton kernel programming.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            
        Returns:
            Dictionary with relevance information
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
        """
        if not video_url or not video_url.startswith("https://"):
            raise ValueError(f"Invalid video URL: {video_url}")
            
        logger.info(f"Checking content relevance for video: {video_url}")
        
        model = self.client.get_model(self.model_id)
        
        for attempt in range(max_retries):
            try:
                prompt = CONTENT_RELEVANCE_CHECK_PROMPT.format(content=video_url)
                response = model.generate_content(prompt)
                
                # Try to parse the JSON response
                try:
                    import json
                    import re
                    
                    # Extract JSON from the response
                    text = response.text
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        relevance_data = json.loads(json_str)
                        logger.info(f"Content relevance check completed for {video_url}")
                        return relevance_data
                    else:
                        logger.warning(f"Could not extract JSON from relevance check response")
                        return {
                            "is_relevant": False,
                            "relevance_score": 0,
                            "contains_cuda_content": False,
                            "contains_triton_content": False,
                            "primary_topic": "Unknown",
                            "key_technical_concepts": [],
                            "explanation": "Failed to parse relevance check response"
                        }
                except Exception as e:
                    logger.warning(f"Error parsing relevance check response: {e}")
                    return {
                        "is_relevant": False,
                        "relevance_score": 0,
                        "contains_cuda_content": False,
                        "contains_triton_content": False,
                        "primary_topic": "Unknown",
                        "key_technical_concepts": [],
                        "explanation": f"Error parsing response: {str(e)}"
                    }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to check content relevance after {max_retries} attempts")
                    raise ConnectionError(f"Failed to check content relevance: {e}")
        
        # This should not be reached due to the raise in the loop
        return {
            "is_relevant": False,
            "relevance_score": 0,
            "contains_cuda_content": False,
            "contains_triton_content": False,
            "primary_topic": "Unknown",
            "key_technical_concepts": [],
            "explanation": "Failed to complete relevance check"
        }
    
    def process_video(self, video_url: str, max_retries: int = 3, timeout: int = 300) -> Dict[str, Any]:
        """
        Process a YouTube video directly with Gemini.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            timeout: Timeout in seconds for API calls
            
        Returns:
            Dictionary with the processed content
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
            TimeoutError: If the API call times out
            Exception: For other unexpected errors
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        logger.info(f"Processing video: {video_url}")
        
        # First, check if the content is relevant to CUDA/Triton kernel programming
        try:
            relevance_data = self.check_content_relevance(video_url, max_retries)
            
            # If the content is not relevant, return early with the relevance data
            if not relevance_data.get("is_relevant", False) or relevance_data.get("relevance_score", 0) < 3:
                logger.info(f"Video content not relevant to CUDA/Triton kernel programming: {video_url}")
                return {
                    "video_url": video_url,
                    "is_relevant": False,
                    "relevance_data": relevance_data,
                    "summary": f"This video does not contain relevant CUDA/Triton kernel programming content. " +
                              f"It appears to be about: {relevance_data.get('primary_topic', 'unknown topic')}. " +
                              f"{relevance_data.get('explanation', '')}"
                }
            
            logger.info(f"Video content is relevant to CUDA/Triton kernel programming: {video_url}")
        except Exception as e:
            logger.warning(f"Error checking content relevance: {e}. Proceeding with video processing anyway.")
            # If relevance check fails, continue with normal processing
        
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the basic prompt to summarize the video
                prompt = BASIC_VIDEO_SUMMARY_PROMPT
                
                # Call the Gemini API with the video URL
                # Note: timeout is handled at the client level, not in the API call
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        {
                            "parts": [
                                {"text": prompt},
                                {"file_data": {"file_uri": video_url}}
                            ]
                        }
                    ]
                )
                
                if not response or not hasattr(response, 'text'):
                    logger.warning(f"Empty or invalid response from Gemini API on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise ValueError("Received empty or invalid response from Gemini API")
                
                # Return the response
                result = {
                    "video_url": video_url,
                    "summary": response.text
                }
                
                # Add relevance data if available
                if 'relevance_data' in locals():
                    result["is_relevant"] = relevance_data.get("is_relevant", True)
                    result["relevance_data"] = relevance_data
                
                return result
                
            except ConnectionError as e:
                logger.warning(f"Network error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Could not connect to Gemini API.")
                    raise ConnectionError(f"Failed to connect to Gemini API after {max_retries} attempts: {str(e)}")
            except TimeoutError as e:
                logger.warning(f"Timeout error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with increased timeout...")
                    timeout *= 1.5  # Increase timeout for next attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. API call timed out.")
                    raise TimeoutError(f"API call timed out after {max_retries} attempts with timeout {timeout}s: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing video with Gemini: {str(e)}")
                raise
    
    def extract_detailed_content(self, video_url: str, max_retries: int = 3, timeout: int = 300, 
                                relevance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract detailed content from a YouTube video using Gemini.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            timeout: Timeout in seconds for API calls
            relevance_data: Optional pre-checked relevance data
            
        Returns:
            Dictionary with detailed content
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
            TimeoutError: If the API call times out
            Exception: For other unexpected errors
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        logger.info(f"Extracting detailed content from video: {video_url}")
        
        # Check content relevance if not already provided
        if relevance_data is None:
            try:
                relevance_data = self.check_content_relevance(video_url, max_retries)
                
                # If the content is not relevant, return early with the relevance data
                if not relevance_data.get("is_relevant", False) or relevance_data.get("relevance_score", 0) < 3:
                    logger.info(f"Video content not relevant to CUDA/Triton kernel programming: {video_url}")
                    return {
                        "video_url": video_url,
                        "is_relevant": False,
                        "relevance_data": relevance_data,
                        "detailed_content": f"This video does not contain relevant CUDA/Triton kernel programming content. " +
                                          f"It appears to be about: {relevance_data.get('primary_topic', 'unknown topic')}. " +
                                          f"{relevance_data.get('explanation', '')}"
                    }
                
                logger.info(f"Video content is relevant to CUDA/Triton kernel programming: {video_url}")
            except Exception as e:
                logger.warning(f"Error checking content relevance: {e}. Proceeding with detailed content extraction anyway.")
                # If relevance check fails, continue with normal processing
        elif not relevance_data.get("is_relevant", False) or relevance_data.get("relevance_score", 0) < 3:
            logger.info(f"Video content not relevant to CUDA/Triton kernel programming (from provided data): {video_url}")
            return {
                "video_url": video_url,
                "is_relevant": False,
                "relevance_data": relevance_data,
                "detailed_content": f"This video does not contain relevant CUDA/Triton kernel programming content. " +
                                  f"It appears to be about: {relevance_data.get('primary_topic', 'unknown topic')}. " +
                                  f"{relevance_data.get('explanation', '')}"
            }
        
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the detailed prompt to extract specific information
                prompt = DETAILED_VIDEO_ANALYSIS_PROMPT
                
                # Call the Gemini API with the video URL
                # Note: timeout is handled at the client level, not in the API call
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        {
                            "parts": [
                                {"text": prompt},
                                {"file_data": {"file_uri": video_url}}
                            ]
                        }
                    ]
                )
                
                if not response or not hasattr(response, 'text'):
                    logger.warning(f"Empty or invalid response from Gemini API on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise ValueError("Received empty or invalid response from Gemini API")
                
                # Return the response
                result = {
                    "video_url": video_url,
                    "detailed_content": response.text
                }
                
                # Add relevance data if available
                if relevance_data:
                    result["is_relevant"] = relevance_data.get("is_relevant", True)
                    result["relevance_data"] = relevance_data
                
                return result
            except ConnectionError as e:
                logger.warning(f"Network error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Could not connect to Gemini API.")
                    raise ConnectionError(f"Failed to connect to Gemini API after {max_retries} attempts: {str(e)}")
            except TimeoutError as e:
                logger.warning(f"Timeout error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with increased timeout...")
                    timeout *= 1.5  # Increase timeout for next attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. API call timed out.")
                    raise TimeoutError(f"API call timed out after {max_retries} attempts with timeout {timeout}s: {str(e)}")
            except Exception as e:
                logger.error(f"Error extracting detailed content with Gemini: {str(e)}")
                raise