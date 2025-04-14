#!/usr/bin/env python3
"""
YouTube Summary Extractor with Chain of Density

This script processes YouTube videos using the Gemini API and applies the Chain of Density
method to extract high-quality, detailed information about CUDA and Triton kernels.
"""

import os
import time
import logging
import argparse
import sys
from datetime import datetime
import google.generativeai as genai

# Add the parent directory to the path so we can import the gpu_info package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu_info.utils.prompts import (
    CHAIN_OF_DENSITY_YOUTUBE_PROMPT,
    INFORMATION_EXTRACTION_YOUTUBE_PROMPT
)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_info", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "youtube_summary.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_youtube_video(youtube_url, api_key):
    """
    Process a YouTube video using the Gemini API.
    
    Args:
        youtube_url (str): URL of the YouTube video to process
        api_key (str): Google API key
        
    Returns:
        str: The initial summary of the video
    """
    logger.info(f"Processing video: {youtube_url}")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    try:
        # Step 1: Initial summary
        logger.info("Generating initial summary...")
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
        logger.info(f"Initial summary completed in {elapsed_time:.2f} seconds")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def apply_chain_of_density(initial_summary, api_key, youtube_url):
    """
    Apply the Chain of Density method to refine the summary.
    
    Args:
        initial_summary (str): Initial summary of the video
        api_key (str): Google API key
        youtube_url (str): URL of the YouTube video
        
    Returns:
        str: Refined summary with high-density information
    """
    logger.info("Applying Chain of Density method...")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Chain of Density prompt
    cod_prompt = CHAIN_OF_DENSITY_YOUTUBE_PROMPT.format(
        initial_summary=initial_summary,
        youtube_url=youtube_url
    )
    
    try:
        logger.info("Generating refined summary...")
        start_time = time.time()
        
        response = model.generate_content(cod_prompt)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Refined summary completed in {elapsed_time:.2f} seconds")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error applying Chain of Density: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def extract_specific_information(refined_summary, api_key):
    """
    Extract specific types of information from the refined summary.
    
    Args:
        refined_summary (str): Refined summary from Chain of Density
        api_key (str): Google API key
        
    Returns:
        str: Final structured report with specific information categories
    """
    logger.info("Extracting specific information...")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    extraction_prompt = INFORMATION_EXTRACTION_YOUTUBE_PROMPT.format(
        refined_summary=refined_summary
    )
    
    try:
        logger.info("Generating specific information extraction...")
        start_time = time.time()
        
        response = model.generate_content(extraction_prompt)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Specific information extraction completed in {elapsed_time:.2f} seconds")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error extracting specific information: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='YouTube Summary Extractor with Chain of Density')
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL')
    
    default_output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "gpu_info", "output", "summary.md"
    )
    parser.add_argument('--output', type=str, default=default_output, help='Output file path')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    # Process the YouTube video
    initial_summary = process_youtube_video(args.url, api_key)
    if not initial_summary:
        logger.error("Failed to generate initial summary")
        return
    
    # Apply Chain of Density
    refined_summary = apply_chain_of_density(initial_summary, api_key, args.url)
    if not refined_summary:
        logger.error("Failed to apply Chain of Density")
        # Fall back to initial summary
        refined_summary = initial_summary
    
    # Extract specific information
    final_report = extract_specific_information(refined_summary, api_key)
    if not final_report:
        logger.error("Failed to extract specific information")
        # Fall back to refined summary
        final_report = refined_summary
    
    # Save the final report
    with open(args.output, 'w') as f:
        f.write(f"# Summary of {args.url}\n\n")
        f.write(final_report)
    
    logger.info(f"Final report saved to {args.output}")
    
    # Print a preview
    preview = final_report[:500] + "..." if len(final_report) > 500 else final_report
    logger.info(f"Final report preview:\n{preview}")

if __name__ == "__main__":
    main()