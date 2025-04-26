"""
Gemini Video Processor Module

This module uses the Gemini API to directly process YouTube videos.
It handles the direct video URL input to Gemini for processing.
"""

import logging
import time
from typing import Dict, Optional, Any

from gpu_info.core.llm_client import get_llm_client
from gpu_info.models.models import VideoAnalysis
from gpu_info.utils.prompts import (
    BASIC_VIDEO_SUMMARY_PROMPT,
    DETAILED_VIDEO_ANALYSIS_PROMPT
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
        self.llm_client = get_llm_client(api_key=api_key, model_name=model_id)
        logger.info(f"Initialized GeminiVideoProcessor with model: {model_id}")
    
    def process_video(self, video_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Process a YouTube video directly with Gemini.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            
        Returns:
            Dictionary with the processed content
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
            Exception: For other unexpected errors
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        logger.info(f"Processing video: {video_url}")
        
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the basic prompt to summarize the video
                prompt = BASIC_VIDEO_SUMMARY_PROMPT
                
                # Call the Gemini API with the video URL
                response = self.llm_client.generate_content(
                    prompt=f"{prompt}\n\nVideo URL: {video_url}"
                )
                
                if not response:
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
                    "summary": response
                }
                
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
            except Exception as e:
                logger.error(f"Error processing video with Gemini: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def extract_detailed_content(self, video_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract detailed content from a YouTube video using Gemini.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            
        Returns:
            Dictionary with detailed content
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
            Exception: For other unexpected errors
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        logger.info(f"Extracting detailed content from video: {video_url}")
        
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the detailed prompt to extract specific information
                prompt = DETAILED_VIDEO_ANALYSIS_PROMPT
                
                # Call the Gemini API with the video URL using structured output
                try:
                    video_analysis = self.llm_client.generate_structured_content(
                        prompt=f"{prompt}\n\nVideo URL: {video_url}",
                        response_model=VideoAnalysis
                    )
                    
                    # Return the response as a dictionary
                    result = {
                        "video_url": video_url,
                        "detailed_content": video_analysis.dict()
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error generating structured content: {e}. Falling back to unstructured output.")
                    # Fall back to unstructured output
                    response = self.llm_client.generate_content(
                        prompt=f"{prompt}\n\nVideo URL: {video_url}"
                    )
                    
                    if not response:
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
                        "detailed_content": response
                    }
                    
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
            except Exception as e:
                logger.error(f"Error extracting detailed content with Gemini: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise