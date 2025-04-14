"""
Test script to process a single video.
"""

import os
import logging
from main import process_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_video.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Process a test video."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Test video URL
    video_url = "https://www.youtube.com/watch?v=D7_ipDqhtwk"
    
    # Output directory
    output_dir = "./output"
    
    # Process the video
    try:
        result = process_video(api_key, video_url, output_dir)
        logger.info(f"Output saved to: {result}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()