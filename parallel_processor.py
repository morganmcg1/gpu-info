"""
Parallel Video Processor Module

This module provides functionality for processing multiple YouTube videos in parallel.
It uses concurrent.futures to parallelize the processing of videos.
"""

import os
import argparse
import logging
import concurrent.futures
from typing import List, Dict, Any
import time

from gemini_video_processor import GeminiVideoProcessor
from output_formatter import OutputFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_video(video_url: str, api_key: str, model_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Process a single video and save the output.
    
    Args:
        video_url: URL of the YouTube video
        api_key: Google API key for Gemini
        model_id: Gemini model ID to use
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Extract video ID from URL
        video_id = video_url.split("v=")[-1].split("&")[0]
        
        # Initialize processors
        processor = GeminiVideoProcessor(api_key=api_key, model_id=model_id)
        formatter = OutputFormatter(output_dir=output_dir)
        
        # Process the video
        logger.info(f"Processing video: {video_url}")
        start_time = time.time()
        
        # Get basic summary
        summary_result = processor.process_video(video_url)
        
        # Get detailed content
        detailed_result = processor.extract_detailed_content(video_url)
        
        # Combine results
        result = {
            "video_url": video_url,
            "video_id": video_id,
            "summary": summary_result.get("summary", ""),
            "detailed_content": detailed_result.get("detailed_content", "")
        }
        
        # Format and save the output
        output_file = formatter.format_output(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing video {video_id} in {processing_time:.2f} seconds")
        
        return {
            "video_id": video_id,
            "status": "success",
            "output_file": output_file,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing video {video_url}: {str(e)}")
        return {
            "video_url": video_url,
            "status": "error",
            "error": str(e)
        }

def process_videos_parallel(video_urls: List[str], api_key: str, model_id: str, output_dir: str, max_workers: int) -> List[Dict[str, Any]]:
    """
    Process multiple videos in parallel.
    
    Args:
        video_urls: List of YouTube video URLs
        api_key: Google API key for Gemini
        model_id: Gemini model ID to use
        output_dir: Directory to save output files
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of dictionaries with processing results
    """
    results = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(process_video, url, api_key, model_id, output_dir): url 
            for url in video_urls
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    logger.info(f"Successfully processed video: {url}")
                else:
                    logger.error(f"Failed to process video: {url}")
            except Exception as e:
                logger.error(f"Exception processing video {url}: {str(e)}")
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })
    
    return results

def main():
    """Main function to run the parallel processor."""
    parser = argparse.ArgumentParser(description="Process multiple YouTube videos in parallel")
    parser.add_argument("--api-key", required=True, help="Google API key for Gemini")
    parser.add_argument("--model-id", default="gemini-2.5-pro-preview-03-25", help="Gemini model ID to use")
    parser.add_argument("--output-dir", default="./summaries", help="Directory to save output files")
    parser.add_argument("--video-list", required=True, help="File containing list of YouTube video URLs (one per line)")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of parallel workers")
    args = parser.parse_args()
    
    # Read video URLs from file
    try:
        with open(args.video_list, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading video list file: {str(e)}")
        return
    
    if not video_urls:
        logger.error("No video URLs found in the provided file")
        return
    
    logger.info(f"Processing {len(video_urls)} videos with {args.max_workers} parallel workers")
    
    # Process videos in parallel
    results = process_videos_parallel(
        video_urls=video_urls,
        api_key=args.api_key,
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Print summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(f"Processing complete: {success_count} successful, {error_count} failed")
    
    # Print details of failed videos
    if error_count > 0:
        logger.info("Failed videos:")
        for result in results:
            if result.get("status") == "error":
                logger.info(f"  - {result.get('video_url')}: {result.get('error')}")

if __name__ == "__main__":
    main()