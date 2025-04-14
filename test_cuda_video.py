"""
Test script for processing a CUDA video.
"""

import os
import sys
import json
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process a CUDA video.')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='Directory to save the output files.')
    args = parser.parse_args()
    
    # Get the test video URL
    video_url = get_test_video_url()
    logger.info(f"Using test video: {video_url}")
    
    # Import the main module
    from main import process_video
    
    # Get the API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Process the video
    try:
        result = process_video(api_key, video_url, args.output_dir)
        logger.info(f"Video processing completed: {result}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()