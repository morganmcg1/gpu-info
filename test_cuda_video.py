"""
Test script for processing a CUDA video using our full pipeline.
"""

import os
import sys
import json
import logging
import argparse
import google.generativeai as genai
from gpu_info.models.models import VideoAnalysis

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

def process_video_with_gemini(api_key, video_url):
    """Process a video directly with the Gemini API."""
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Create the model
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Create the prompt
    prompt = f"""
    Please analyze this CUDA programming video: {video_url}
    
    Provide a detailed technical analysis of the CUDA programming concepts, code examples, 
    and best practices shown in the video. Focus on extracting specific, actionable information 
    that would be useful for someone learning to write CUDA kernels.
    
    Format your response as a structured analysis with the following sections:
    1. Core Technical Concepts
    2. Code Examples (with complete code)
    3. Mathematical Formulas or Algorithms
    4. Implementation Techniques
    5. Common Pitfalls and Gotchas
    6. Performance Optimization Tips
    
    Be as detailed and technical as possible, including exact code syntax, parameter values,
    and specific performance implications where mentioned.
    """
    
    # Generate the content
    try:
        logger.info("Generating content from video with Gemini API...")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return None

def process_video_with_pipeline(api_key, video_url, output_dir):
    """Process a video using our full pipeline."""
    # Import the main module
    from main import process_video
    
    # Process the video
    try:
        logger.info("Processing video with our pipeline...")
        result = process_video(api_key, video_url, output_dir)
        return result
    except Exception as e:
        logger.error(f"Error processing video with pipeline: {str(e)}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process a CUDA video.')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='Directory to save the output files.')
    parser.add_argument('--method', type=str, choices=['gemini', 'pipeline', 'both'], default='both',
                        help='Method to use for processing the video.')
    args = parser.parse_args()
    
    # Get the test video URL
    video_url = get_test_video_url()
    logger.info(f"Using test video: {video_url}")
    
    # Get the API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process the video with the selected method
    if args.method in ['gemini', 'both']:
        gemini_result = process_video_with_gemini(api_key, video_url)
        if gemini_result:
            # Save the result to a file
            gemini_output_path = os.path.join(args.output_dir, 'gemini_result.md')
            with open(gemini_output_path, 'w') as f:
                f.write(gemini_result)
            logger.info(f"Gemini result saved to {gemini_output_path}")
    
    if args.method in ['pipeline', 'both']:
        pipeline_result = process_video_with_pipeline(api_key, video_url, args.output_dir)
        logger.info(f"Pipeline processing completed: {pipeline_result}")

if __name__ == "__main__":
    main()