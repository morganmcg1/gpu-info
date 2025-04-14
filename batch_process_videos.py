#!/usr/bin/env python3
"""
Batch YouTube Video Processor

This script processes multiple YouTube videos from a list, applying the Chain of Density
method to extract high-quality information about CUDA and Triton kernels.
"""

import os
import time
import logging
import argparse
from datetime import datetime
import concurrent.futures
from pathlib import Path

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

def process_video(youtube_url, output_dir, api_key):
    """
    Process a single YouTube video and save the summary.
    
    Args:
        youtube_url (str): URL of the YouTube video
        output_dir (str): Directory to save the summary
        api_key (str): Google API key
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Extract video ID for the filename
        video_id = youtube_url.split("v=")[1].split("&")[0]
        output_file = os.path.join(output_dir, f"{video_id}_summary.md")
        
        # Skip if already processed
        if os.path.exists(output_file):
            logger.info(f"Video {video_id} already processed, skipping")
            return True
        
        # Import here to avoid circular imports
        from youtube_summary_extractor import process_youtube_video, apply_chain_of_density, extract_specific_information
        
        # Process the video
        logger.info(f"Processing video: {youtube_url}")
        start_time = time.time()
        
        # Step 1: Initial summary
        initial_summary = process_youtube_video(youtube_url, api_key)
        if not initial_summary:
            logger.error(f"Failed to generate initial summary for {video_id}")
            return False
        
        # Step 2: Apply Chain of Density
        refined_summary = apply_chain_of_density(initial_summary, api_key, youtube_url)
        if not refined_summary:
            logger.error(f"Failed to apply Chain of Density for {video_id}")
            # Fall back to initial summary
            refined_summary = initial_summary
        
        # Step 3: Extract specific information
        final_report = extract_specific_information(refined_summary, api_key)
        if not final_report:
            logger.error(f"Failed to extract specific information for {video_id}")
            # Fall back to refined summary
            final_report = refined_summary
        
        # Save the final report
        with open(output_file, 'w') as f:
            f.write(f"# Summary of {youtube_url}\n\n")
            f.write(final_report)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Video {video_id} processed in {elapsed_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing video {youtube_url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch YouTube Video Processor')
    parser.add_argument('--video_list', type=str, required=True, help='Path to file containing YouTube URLs (one per line)')
    parser.add_argument('--output_dir', type=str, default='summaries', help='Directory to save summaries')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of concurrent workers')
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
    
    # Process videos
    successful = 0
    failed = 0
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_url = {executor.submit(process_video, url, args.output_dir, api_key): url for url in video_urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                failed += 1
    
    logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()