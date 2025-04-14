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
from datetime import datetime
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    cod_prompt = f"""
    I want you to help me create a more detailed and information-dense summary of a YouTube video about CUDA or Triton kernels.
    
    Here is the initial summary:
    {initial_summary}
    
    Video URL: {youtube_url}
    
    Using the Chain of Density method, please:
    
    1. Identify specific entities, concepts, techniques, code examples, and performance tips from the video
    2. Create a more detailed summary that includes:
       - Key technical concepts explained in the video
       - Specific code examples or patterns mentioned
       - Performance optimization techniques
       - Common pitfalls or "gotchas" when writing CUDA/Triton kernels
       - Mathematical equations or algorithms discussed
       - Benchmark results or performance comparisons
    
    Focus on extracting high-signal, technical information that would be valuable for someone learning to write CUDA or Triton kernels.
    
    Format the output as a well-structured report with sections for:
    - Overview
    - Key Concepts
    - Code Examples (with full code when available)
    - Performance Optimization Techniques
    - Common Pitfalls
    - Additional Resources
    
    Use markdown formatting for better readability.
    """
    
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
    
    extraction_prompt = f"""
    I have a refined summary of a YouTube video about CUDA or Triton kernels:
    
    {refined_summary}
    
    Please extract and organize the following specific types of information:
    
    1. Code Examples: Extract any complete code examples, ensuring they are properly formatted and include all necessary context
    
    2. Performance Optimization Techniques: List specific techniques mentioned for optimizing CUDA or Triton kernels
    
    3. Mathematical Equations: Extract any mathematical equations or algorithms discussed
    
    4. Common Pitfalls: Identify common mistakes or "gotchas" when writing CUDA or Triton kernels
    
    5. Benchmark Results: Extract any specific performance numbers or comparisons
    
    Format the output as a well-structured markdown report with clear sections and code blocks where appropriate.
    """
    
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
    parser.add_argument('--output', type=str, default='summary.md', help='Output file path')
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