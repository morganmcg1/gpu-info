"""
Gemini Video Processor Module

This module uses the Gemini API to directly process YouTube videos.
It handles the direct video URL input to Gemini for processing.
"""

import logging
from typing import Dict, Optional, Any
from google import genai

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
    
    def process_video(self, video_url: str) -> Dict[str, Any]:
        """
        Process a YouTube video directly with Gemini.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Dictionary with the processed content
        """
        logger.info(f"Processing video: {video_url}")
        
        try:
            # Create a basic prompt to summarize the video
            prompt = "Can you summarize this video?"
            
            # Call the Gemini API with the video URL
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
            
            # Return the response
            return {
                "video_url": video_url,
                "summary": response.text
            }
        except Exception as e:
            logger.error(f"Error processing video with Gemini: {str(e)}")
            raise
    
    def extract_detailed_content(self, video_url: str) -> Dict[str, Any]:
        """
        Extract detailed content from a YouTube video using Gemini.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Dictionary with detailed content
        """
        logger.info(f"Extracting detailed content from video: {video_url}")
        
        try:
            # Create a detailed prompt to extract specific information
            prompt = """
            Please analyze this YouTube video about CUDA or Triton kernels and extract the following information:
            
            1. A comprehensive summary of the key points
            2. Any code examples shown or discussed
            3. Mathematical equations or formulas presented
            4. Step-by-step processes explained
            5. Common pitfalls or gotchas mentioned
            6. Performance optimization techniques discussed
            
            For each section, provide detailed information with specific examples from the video.
            Focus on technical details that would be valuable for someone learning to write CUDA or Triton kernels.
            """
            
            # Call the Gemini API with the video URL
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
            
            # Return the response
            return {
                "video_url": video_url,
                "detailed_content": response.text
            }
        except Exception as e:
            logger.error(f"Error extracting detailed content with Gemini: {str(e)}")
            raise