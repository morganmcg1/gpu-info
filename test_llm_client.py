"""
Test script for the LLMClient.
"""

import os
import sys
import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the LLMClient
from llm_client import LLMClient, get_llm_client
from models import CodeExample, Equation, Gotcha, PerformanceTip, VideoAnalysis

def get_test_video_url():
    """Get the test video URL from the config file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'test_videos.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('default_test_video')
    except Exception as e:
        logger.error(f"Error loading test video config: {str(e)}")
        # Fallback to the hardcoded URL
        return "https://www.youtube.com/watch?v=pPStdjuYzSI"

def test_generate_content():
    """Test the generate_content method."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    client = LLMClient(api_key=api_key)
    
    prompt = "What is CUDA and how is it used in GPU programming? Keep it brief."
    
    logger.info("Testing generate_content...")
    response = client.generate_content(prompt)
    
    logger.info(f"Response: {response}")
    
    return response

def test_generate_structured_content():
    """Test the generate_structured_content method."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    client = LLMClient(api_key=api_key)
    
    # Define a simple model for testing
    class TestModel(BaseModel):
        name: str
        description: str
        score: int = Field(..., ge=0, le=100)
    
    prompt = "Generate information about CUDA. Return a JSON object with name, description, and score (0-100)."
    
    logger.info("Testing generate_structured_content...")
    try:
        response = client.generate_structured_content(prompt, TestModel)
        logger.info(f"Response: {response}")
        logger.info(f"Name: {response.name}")
        logger.info(f"Description: {response.description}")
        logger.info(f"Score: {response.score}")
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

def test_generate_list_content():
    """Test the generate_list_content method."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    client = LLMClient(api_key=api_key)
    
    # Define a simple model for testing
    class TestItem(BaseModel):
        name: str
        value: int
    
    prompt = "Generate a list of 3 programming languages with their popularity scores (1-10). Return as a JSON array."
    
    logger.info("Testing generate_list_content...")
    try:
        response = client.generate_list_content(prompt, TestItem)
        logger.info(f"Response: {response}")
        for item in response:
            logger.info(f"Name: {item.name}, Value: {item.value}")
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

def test_video_analysis_model():
    """Test the VideoAnalysis model."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    client = LLMClient(api_key=api_key)
    
    # Get the test video URL
    video_url = get_test_video_url()
    logger.info(f"Using test video: {video_url}")
    
    prompt = f"""
    Analyze the following CUDA programming video: {video_url}
    
    This is a video about CUDA programming for GPU acceleration. Please provide a structured analysis of this video, 
    including core technical concepts, code examples, equations, implementation techniques, and common pitfalls.
    
    Focus on extracting detailed technical information about CUDA programming, kernel optimization, 
    memory management, and performance considerations.
    """
    
    logger.info("Testing VideoAnalysis model...")
    try:
        response = client.generate_structured_content(prompt, VideoAnalysis)
        logger.info(f"Response: {response}")
        logger.info(f"Core Technical Concepts: {len(response.core_technical_concepts)}")
        logger.info(f"Code Examples: {len(response.code_examples)}")
        logger.info(f"Mathematical Formulas: {len(response.mathematical_formulas)}")
        logger.info(f"Implementation Techniques: {len(response.implementation_techniques)}")
        logger.info(f"Common Pitfalls: {len(response.common_pitfalls)}")
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the generate_content method
    test_generate_content()
    
    # Test the generate_structured_content method
    test_generate_structured_content()
    
    # Test the generate_list_content method
    test_generate_list_content()
    
    # Test the VideoAnalysis model
    test_video_analysis_model()