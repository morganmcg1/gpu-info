#!/usr/bin/env python3
"""
Batch YouTube Video Processor

This script processes multiple YouTube videos from a list, applying the Chain of Density
method to extract high-quality information about CUDA and Triton kernels.

It now uses the parallel_processor module for improved performance and reliability.
"""

import os
import logging
import argparse
from parallel_processor import process_videos_parallel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Batch YouTube Video Processor')
    parser.add_argument('--video_list', type=str, required=True, help='Path to file containing YouTube URLs (one per line)')
    parser.add_argument('--output_dir', type=str, default='summaries', help='Directory to save summaries')
    parser.add_argument('--max_workers', type=int, default=3, help='Maximum number of concurrent workers')
    parser.add_argument('--model_id', type=str, default='gemini-2.5-pro-preview-03-25', help='Gemini model ID to use')
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read video URLs from file
    try:
        with open(args.video_list, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading video list: {e}")
        return
    
    logger.info(f"Found {len(video_urls)} videos to process")
    logger.info(f"Using {args.max_workers} parallel workers")
    
    # Process videos in parallel
    results = process_videos_parallel(
        video_urls=video_urls,
        api_key=api_key,
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