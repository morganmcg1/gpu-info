"""
Test script for directly processing a YouTube video with the Gemini API.
"""

import os
import sys
import json
import logging
import google.generativeai as genai

# Add the parent directory to the path so we can import the gpu_info package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gpu_info.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_test_video_url():
    """Get the test video URL from the config file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../gpu_info/config/test_videos.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('default_test_video')
    except Exception as e:
        logger.error(f"Error loading test video config: {str(e)}")
        # Fallback to the hardcoded URL
        return "https://www.youtube.com/watch?v=pPStdjuYzSI"

def main():
    """Main function."""
    # Get the API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Get the test video URL
    video_url = get_test_video_url()
    logger.info(f"Using test video: {video_url}")
    
    # Create the model
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Create the prompt
    prompt = f"""
    Please analyze this CUDA programming video: {video_url}
    
    Provide a detailed summary of the key technical concepts, code examples, and best practices 
    for CUDA programming shown in the video. Focus on extracting specific, actionable information 
    that would be useful for someone learning to write CUDA kernels.
    """
    
    # Generate the content
    try:
        logger.info("Generating content from video...")
        response = model.generate_content(prompt)
        
        # Print the response
        logger.info("Response received:")
        print(response.text)
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return None

if __name__ == "__main__":
    main()