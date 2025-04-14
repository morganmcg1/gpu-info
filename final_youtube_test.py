#!/usr/bin/env python3
"""
Test script for directly processing YouTube videos with Gemini API
using the exact format provided by the user.
"""

import os
import time
import logging
import google.generativeai as genai
from datetime import datetime

# Check genai version
import pkg_resources
genai_version = pkg_resources.get_distribution("google-generativeai").version
print(f"Using google-generativeai version: {genai_version}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    # Initialize Gemini client
    logger.info("Initializing Gemini client")
    genai.configure(api_key=api_key)
    
    # Test video URL - CUDA lecture
    youtube_url = "https://www.youtube.com/watch?v=LuhJEEJQgUM"
    logger.info(f"Processing video: {youtube_url}")
    
    try:
        # Using the exact API format provided by the user
        logger.info("Sending request to Gemini API with exact format")
        start_time = time.time()
        logger.info(f"Request started at: {datetime.now().strftime('%H:%M:%S')}")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            [
                "Can you summarize this video?",
                {"file_data": {"file_uri": youtube_url}}
            ]
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Request completed in {elapsed_time:.2f} seconds")
        logger.info(f"Response received at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Save response to file
        output_file = "final_youtube_response.md"
        with open(output_file, "w") as f:
            f.write(response.text)
        
        logger.info(f"Response saved to {output_file}")
        
        # Print first 500 characters of response
        preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
        logger.info(f"Response preview:\n{preview}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()