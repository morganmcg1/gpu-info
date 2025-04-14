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
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized GeminiVideoProcessor with model: {model_id}")
    
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
                return {
                    "video_url": video_url,
                    "summary": response.text
                }
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
    
    def extract_detailed_content(self, video_url: str, max_retries: int = 3, timeout: int = 300) -> Dict[str, Any]:
        """
        Extract detailed content from a YouTube video using Gemini.
        
        Args:
            video_url: URL of the YouTube video
            max_retries: Maximum number of retries for API calls
            timeout: Timeout in seconds for API calls
            
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
                return {
                    "video_url": video_url,
                    "detailed_content": response.text
                }
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