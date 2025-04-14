#!/usr/bin/env python3
"""
Batch YouTube Video Processor

This script processes multiple YouTube videos from a list, applying the Chain of Density
method to extract high-quality information about CUDA and Triton kernels.

It now uses the parallel_processor module for improved performance and reliability.
"""

import os
import sys
import argparse

# Add the parent directory to the path so we can import the gpu_info package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu_info.parallel.parallel_processor import process_videos_parallel
from gpu_info.utils.config import Config
from gpu_info.utils.logger import get_logger, setup_root_logger

# Set up logging
setup_root_logger()
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Batch YouTube Video Processor')
    parser.add_argument('--video_list', type=str, required=True, help='Path to file containing YouTube URLs (one per line)')
    parser.add_argument('--output_dir', type=str, default=Config.get("OUTPUT_DIR"), help='Directory to save summaries')
    parser.add_argument('--max_workers', type=int, default=Config.get("MAX_WORKERS"), help='Maximum number of concurrent workers')
    parser.add_argument('--model_id', type=str, default=Config.get("GEMINI_MODEL_ID"), help='Gemini model ID to use')
    parser.add_argument('--api_key', type=str, help='Google API key (defaults to GOOGLE_API_KEY env var)')
    parser.add_argument('--log_level', type=str, default=Config.get("LOG_LEVEL"), help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of already processed videos')
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    if args.output_dir:
        Config.set("OUTPUT_DIR", args.output_dir)
    if args.max_workers:
        Config.set("MAX_WORKERS", args.max_workers)
    if args.model_id:
        Config.set("GEMINI_MODEL_ID", args.model_id)
    if args.api_key:
        Config.set("GEMINI_API_KEY", args.api_key)
    if args.log_level:
        Config.set("LOG_LEVEL", args.log_level)
    if args.force_reprocess:
        Config.set("FORCE_REPROCESS", True)
    
    # Get API key from configuration
    api_key = Config.get("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("No API key provided. Set GOOGLE_API_KEY environment variable or use --api_key")
            return
        Config.set("GEMINI_API_KEY", api_key)
    
    # Create output directory if it doesn't exist
    output_dir = Config.get("OUTPUT_DIR")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read video URLs from file
    try:
        with open(args.video_list, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading video list: {e}")
        return
    
    if not video_urls:
        logger.error("No video URLs found in the provided file")
        return
    
    logger.info(f"Found {len(video_urls)} videos to process")
    logger.info(f"Using {Config.get('MAX_WORKERS')} parallel workers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model ID: {Config.get('GEMINI_MODEL_ID')}")
    
    # Process videos in parallel
    results = process_videos_parallel(
        video_urls=video_urls,
        api_key=api_key,
        model_id=Config.get("GEMINI_MODEL_ID"),
        output_dir=output_dir,
        max_workers=Config.get("MAX_WORKERS")
    )
    
    # Print summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(f"Processing complete: {success_count} successful, {skipped_count} skipped, {error_count} failed")
    
    # Print details of failed videos
    if error_count > 0:
        logger.info("Failed videos:")
        for result in results:
            if result.get("status") == "error":
                logger.info(f"  - {result.get('video_url')}: {result.get('error')}")

if __name__ == "__main__":
    main()