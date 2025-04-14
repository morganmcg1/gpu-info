#!/usr/bin/env python3
"""
Example script demonstrating how to use the GPU Info system to process a YouTube video.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the necessary modules
from gpu_info.core.gemini_video_processor import GeminiVideoProcessor
from gpu_info.analysis.chain_of_density import ChainOfDensityRefiner
from gpu_info.analysis.information_extractor import InformationExtractor
from gpu_info.output.output_formatter import OutputFormatter
from gpu_info.utils.logger import setup_logger
from gpu_info.utils.config import Config

# Set up logging
logger = setup_logger("process_video_example")

def process_video(video_url, output_dir=None):
    """
    Process a YouTube video and extract information about CUDA/Triton kernels.
    
    Args:
        video_url (str): The URL of the YouTube video to process.
        output_dir (str, optional): The directory to save the output to.
            Defaults to None (uses the default output directory).
    
    Returns:
        dict: The extracted information.
    """
    # Set up the output directory
    if output_dir:
        Config.set("OUTPUT_DIR", output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Process the video with Gemini
    logger.info(f"Processing video: {video_url}")
    processor = GeminiVideoProcessor()
    video_info = processor.process_video(video_url)
    
    # Apply the Chain of Density method to refine the summary
    logger.info("Refining summary with Chain of Density method")
    refiner = ChainOfDensityRefiner()
    refined_summary = refiner.refine(video_info["summary"])
    video_info["refined_summary"] = refined_summary
    
    # Extract specific information
    logger.info("Extracting specific information")
    extractor = InformationExtractor()
    detailed_content = extractor.extract_information(video_info)
    
    # Format the output
    logger.info("Formatting output")
    formatter = OutputFormatter()
    output_path = formatter.format_and_save(video_info, detailed_content)
    
    logger.info(f"Output saved to: {output_path}")
    return {
        "video_info": video_info,
        "detailed_content": detailed_content,
        "output_path": output_path
    }

def main():
    """Main function."""
    # Check if a video URL was provided
    if len(sys.argv) < 2:
        print("Usage: python process_video_example.py <video_url> [output_dir]")
        sys.exit(1)
    
    # Get the video URL and output directory
    video_url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Process the video
    result = process_video(video_url, output_dir)
    
    # Print the output path
    print(f"\nOutput saved to: {result['output_path']}")

if __name__ == "__main__":
    main()