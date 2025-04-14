"""
Main Module

This is the main script that ties together all the components of the system.
It processes YouTube videos about CUDA and Triton kernels to extract high-quality information.
"""

import os
import argparse
import logging
from typing import List, Dict, Any
import json
import sys

# Add the parent directory to the path so we can import the gpu_info package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu_info.core.youtube_processor import YouTubeProcessor
from gpu_info.analysis.chain_of_density import ChainOfDensity
from gpu_info.analysis.information_extractor import InformationExtractor
from gpu_info.output.output_formatter import OutputFormatter
from gpu_info.core.gemini_video_processor import GeminiVideoProcessor
from gpu_info.core.llm_client import get_llm_client

from dotenv import load_dotenv

# Determine the absolute path to the directory containing main.py
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(script_dir, '.env')
# Load the .env file using the explicit path
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_info", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "gpu_info.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_video(api_key: str, video_url: str, output_dir: str) -> Dict[str, str]:
    """
    Process a single YouTube video.
    
    Args:
        api_key: Google API key for Gemini
        video_url: URL of the YouTube video
        output_dir: Directory to save output
        
    Returns:
        Dictionary with paths to output files
    """
    logger.info(f"Processing video: {video_url}")
    
    # Initialize components
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_info", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    youtube_processor = YouTubeProcessor(cache_dir=cache_dir)
    llm_client = get_llm_client(api_key=api_key)
    chain_of_density = ChainOfDensity(api_key=api_key)
    information_extractor = InformationExtractor(api_key=api_key)
    output_formatter = OutputFormatter(output_dir=output_dir)
    gemini_processor = GeminiVideoProcessor(api_key=api_key)
    
    try:
        # Get video information
        video_info, _ = youtube_processor.process_video(video_url)
        
        # Check if we received partial information (a fallback)
        fallback_info = False
        if "Could not retrieve" in video_info.get("description", ""):
            logger.warning(f"Using fallback video information for {video_url}")
            fallback_info = True
        
        # Process all videos regardless of content
        
        # Process the video directly with Gemini
        try:
            detailed_content = gemini_processor.extract_detailed_content(video_url)
            content = detailed_content.detailed_content
        except Exception as e:
            logger.error(f"Error extracting content with Gemini: {str(e)}")
            if fallback_info:
                # If we already have fallback info and Gemini fails, we can't proceed
                logger.error(f"Both video info extraction and Gemini processing failed for {video_url}")
                return {"error": f"Failed to process video: {str(e)}"}
            else:
                # We have good video info but Gemini failed, return partial result
                return {
                    "video_info": video_info,
                    "error": f"Gemini processing failed: {str(e)}"
                }
        
        # Apply Chain of Density to refine the summary
        summaries = chain_of_density.apply_chain_of_density(content, iterations=3)
        final_summary = summaries[-1]
        
        # Extract specific information
        extracted_info = information_extractor.extract_all_information(content)
        
        # Format and save the results
        output_paths = output_formatter.format_and_save(video_info, final_summary, extracted_info)
        
        logger.info(f"Successfully processed video: {video_info['title']}")
        return output_paths
    
    except Exception as e:
        logger.error(f"Error processing video {video_url}: {str(e)}")
        raise

def process_video_list(api_key: str, video_urls: List[str], output_dir: str) -> List[Dict[str, str]]:
    """
    Process a list of YouTube videos.
    
    Args:
        api_key: Google API key for Gemini
        video_urls: List of YouTube video URLs
        output_dir: Directory to save output
        
    Returns:
        List of dictionaries with paths to output files
    """
    logger.info(f"Processing {len(video_urls)} videos")
    
    results = []
    for video_url in video_urls:
        try:
            result = process_video(api_key, video_url, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process video {video_url}: {str(e)}")
    
    return results

def read_video_urls_from_file(file_path: str) -> List[str]:
    """
    Read YouTube video URLs from a file.
    
    Args:
        file_path: Path to a file containing a list of video URLs
        
    Returns:
        List of YouTube video URLs
    """
    logger.info(f"Reading video URLs from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(video_urls)} video URLs in {file_path}")
        return video_urls
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Process YouTube videos about CUDA and Triton kernels.")
    parser.add_argument("--api_key", help="Google API key for Gemini")
    parser.add_argument("--video_url", help="URL of a YouTube video to process")
    parser.add_argument("--video_list", help="Path to a file containing a list of video URLs")
    parser.add_argument("--link_file", default="youtube_links.txt", help="Path to a file containing YouTube links (default: youtube_links.txt)")
    parser.add_argument("--output_dir", 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_info", "output"), 
                        help="Directory to save output")
    
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Use --api_key or set GOOGLE_API_KEY environment variable.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process videos
    if args.video_url:
        # Process a single video
        result = process_video(api_key, args.video_url, args.output_dir)
        logger.info(f"Output saved to: {result}")
    elif args.video_list:
        # Process videos from a specified list file
        video_urls = read_video_urls_from_file(args.video_list)
        if video_urls:
            results = process_video_list(api_key, video_urls, args.output_dir)
            logger.info(f"Processed {len(results)} videos")
        else:
            logger.error(f"No valid video URLs found in {args.video_list}")
    elif os.path.exists(args.link_file):
        # Process videos from the default link file
        video_urls = read_video_urls_from_file(args.link_file)
        if video_urls:
            results = process_video_list(api_key, video_urls, args.output_dir)
            logger.info(f"Processed {len(results)} videos from {args.link_file}")
        else:
            logger.error(f"No valid video URLs found in {args.link_file}")
    else:
        logger.error("No video source provided. Use --video_url, --video_list, or create a youtube_links.txt file.")

if __name__ == "__main__":
    main()