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
from youtube_processor import YouTubeProcessor
from chain_of_density import ChainOfDensity
from information_extractor import InformationExtractor
from output_formatter import OutputFormatter
from gemini_video_processor import GeminiVideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_info.log"),
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
    youtube_processor = YouTubeProcessor(cache_dir=os.path.join(output_dir, "cache"))
    chain_of_density = ChainOfDensity(api_key=api_key)
    information_extractor = InformationExtractor(api_key=api_key)
    output_formatter = OutputFormatter(output_dir=output_dir)
    gemini_processor = GeminiVideoProcessor(api_key=api_key)
    
    try:
        # Get video information
        video_info, _ = youtube_processor.process_video(video_url)
        
        # Process the video directly with Gemini
        detailed_content = gemini_processor.extract_detailed_content(video_url)
        
        # Apply Chain of Density to refine the summary
        content = detailed_content["detailed_content"]
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

def process_video_list(api_key: str, video_list_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    Process a list of YouTube videos.
    
    Args:
        api_key: Google API key for Gemini
        video_list_path: Path to a file containing a list of video URLs
        output_dir: Directory to save output
        
    Returns:
        List of dictionaries with paths to output files
    """
    logger.info(f"Processing videos from list: {video_list_path}")
    
    # Read the video list
    with open(video_list_path, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    results = []
    for video_url in video_urls:
        try:
            result = process_video(api_key, video_url, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process video {video_url}: {str(e)}")
    
    return results

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Process YouTube videos about CUDA and Triton kernels.")
    parser.add_argument("--api_key", help="Google API key for Gemini")
    parser.add_argument("--video_url", help="URL of a YouTube video to process")
    parser.add_argument("--video_list", help="Path to a file containing a list of video URLs")
    parser.add_argument("--output_dir", default="./output", help="Directory to save output")
    
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
        result = process_video(api_key, args.video_url, args.output_dir)
        logger.info(f"Output saved to: {result}")
    elif args.video_list:
        results = process_video_list(api_key, args.video_list, args.output_dir)
        logger.info(f"Processed {len(results)} videos")
    else:
        logger.error("No video URL or list provided. Use --video_url or --video_list.")

if __name__ == "__main__":
    main()