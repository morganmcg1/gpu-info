"""
Parallel Video Processor Module

This module provides functionality for processing multiple YouTube videos in parallel.
It uses concurrent.futures to parallelize the processing of videos.
"""

import os
import argparse
import concurrent.futures
from typing import List, Dict, Any
import time

from gemini_video_processor import GeminiVideoProcessor
from output_formatter import OutputFormatter
from config import Config
from logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

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
        
        # Check if output file already exists
        output_path = os.path.join(output_dir, f"{video_id}_summary.md")
        if os.path.exists(output_path) and not Config.get("FORCE_REPROCESS", False):
            logger.info(f"Video {video_id} already processed, skipping (use FORCE_REPROCESS=true to override)")
            return {
                "video_id": video_id,
                "status": "skipped",
                "output_file": output_path,
                "processing_time": 0
            }
        
        # Initialize processors
        processor = GeminiVideoProcessor(
            api_key=api_key, 
            model_id=model_id
        )
        formatter = OutputFormatter(output_dir=output_dir)
        
        # Process the video
        logger.info(f"Processing video: {video_url}")
        start_time = time.time()
        
        # Get basic summary with retry logic
        max_retries = Config.get("MAX_RETRIES", 3)
        timeout = Config.get("TIMEOUT", 300)
        summary_result = processor.process_video(
            video_url=video_url,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Get detailed content with retry logic
        detailed_result = processor.extract_detailed_content(
            video_url=video_url,
            max_retries=max_retries,
            timeout=timeout
        )
        
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
        import traceback
        logger.debug(traceback.format_exc())
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
    
    # Use configuration values if not provided
    if not model_id:
        model_id = Config.get("GEMINI_MODEL_ID")
    if not output_dir:
        output_dir = Config.get("OUTPUT_DIR")
    if not max_workers:
        max_workers = Config.get("MAX_WORKERS")
    
    # Log processing start
    logger.info(f"Starting parallel processing of {len(video_urls)} videos with {max_workers} workers")
    start_time = time.time()
    
    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(process_video, url, api_key, model_id, output_dir): url 
            for url in video_urls
        }
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if result["status"] == "success":
                    logger.info(f"[{completed}/{len(video_urls)}] Successfully processed video: {url}")
                elif result["status"] == "skipped":
                    logger.info(f"[{completed}/{len(video_urls)}] Skipped already processed video: {url}")
                else:
                    logger.error(f"[{completed}/{len(video_urls)}] Failed to process video: {url}")
                    
                # Calculate and log progress
                progress_pct = (completed / len(video_urls)) * 100
                elapsed = time.time() - start_time
                videos_per_second = completed / elapsed if elapsed > 0 else 0
                estimated_total = (len(video_urls) / videos_per_second) if videos_per_second > 0 else 0
                estimated_remaining = estimated_total - elapsed if estimated_total > 0 else 0
                
                logger.info(f"Progress: {progress_pct:.1f}% ({completed}/{len(video_urls)}) - "
                           f"Est. remaining: {estimated_remaining/60:.1f} minutes")
                
            except Exception as e:
                logger.error(f"Exception processing video {url}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                results.append({
                    "video_url": url,
                    "status": "error",
                    "error": str(e)
                })
                completed += 1
    
    # Log final statistics
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(f"Completed processing {len(video_urls)} videos in {total_time/60:.1f} minutes")
    logger.info(f"Results: {success_count} successful, {skipped_count} skipped, {error_count} failed")
    
    return results

def main():
    """Main function to run the parallel processor."""
    parser = argparse.ArgumentParser(description="Process multiple YouTube videos in parallel")
    parser.add_argument("--api-key", help="Google API key for Gemini (defaults to GOOGLE_API_KEY env var)")
    parser.add_argument("--model-id", default=Config.get("GEMINI_MODEL_ID"), help="Gemini model ID to use")
    parser.add_argument("--output-dir", default=Config.get("OUTPUT_DIR"), help="Directory to save output files")
    parser.add_argument("--video-list", required=True, help="File containing list of YouTube video URLs (one per line)")
    parser.add_argument("--max-workers", type=int, default=Config.get("MAX_WORKERS"), help="Maximum number of parallel workers")
    parser.add_argument("--log-level", default=Config.get("LOG_LEVEL"), help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", default=Config.get("LOG_FILE"), help="Log file path")
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    if args.api_key:
        Config.set("GEMINI_API_KEY", args.api_key)
    if args.model_id:
        Config.set("GEMINI_MODEL_ID", args.model_id)
    if args.output_dir:
        Config.set("OUTPUT_DIR", args.output_dir)
    if args.max_workers:
        Config.set("MAX_WORKERS", args.max_workers)
    if args.log_level:
        Config.set("LOG_LEVEL", args.log_level)
    if args.log_file:
        Config.set("LOG_FILE", args.log_file)
    
    # Get API key from configuration
    api_key = Config.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set GOOGLE_API_KEY environment variable or use --api-key")
        return
    
    # Read video URLs from file
    try:
        with open(args.video_list, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading video list file: {str(e)}")
        return
    
    if not video_urls:
        logger.error("No video URLs found in the provided file")
        return
    
    logger.info(f"Processing {len(video_urls)} videos with {Config.get('MAX_WORKERS')} parallel workers")
    
    # Process videos in parallel
    results = process_videos_parallel(
        video_urls=video_urls,
        api_key=api_key,
        model_id=Config.get("GEMINI_MODEL_ID"),
        output_dir=Config.get("OUTPUT_DIR"),
        max_workers=Config.get("MAX_WORKERS")
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