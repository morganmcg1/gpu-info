"""
Test script to directly process a shorter YouTube video with Gemini.
Based on the example from geminibyexample.com
"""

import os
import logging
import json
import datetime
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_youtube_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Process a short test video directly with Gemini."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Initialize Gemini client
    logger.info("Initializing Gemini client")
    client = genai.Client(api_key=api_key)
    
    # Test video URL - shorter video
    video_url = "https://www.youtube.com/watch?v=Fwoyko4uuvI"
    logger.info(f"Processing video: {video_url}")
    
    # Create a prompt to summarize the video
    prompt = """
    Please analyze this YouTube video about CUDA or Triton kernels and extract the following information:
    
    1. A comprehensive summary of the key points
    2. Any code examples shown or discussed (with full code)
    3. Mathematical equations or formulas presented
    4. Step-by-step processes explained
    5. Common pitfalls or gotchas mentioned
    6. Performance optimization techniques discussed
    
    For each section, provide detailed information with specific examples from the video.
    Focus on technical details that would be valuable for someone learning to write CUDA or Triton kernels.
    
    Format your response as Markdown with clear section headers.
    """
    
    try:
        # Following the approach from geminibyexample.com but adapted for the current API
        logger.info("Sending request to Gemini API")
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"file_uri": video_url, "mime_type": "video/youtube"}}
                    ]
                }
            ]
        )
        
        logger.info("Response received from Gemini API")
        
        # Save the response to a file
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/short_video_response.md", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        logger.info(f"Response saved to {output_dir}/short_video_response.md")
        
        # Print the first 500 characters of the response
        preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
        logger.info(f"Response preview: {preview}")
        
        # Update summaries.md file
        update_summaries_file(video_url, response.text)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def update_summaries_file(video_url, content):
    """Update the summaries.md file with the new summary."""
    summaries_file = "summaries.md"
    
    # Get video title - use a placeholder since we can't reliably get it
    video_title = "CUDA Kernel Optimization Short Video"
    
    # Current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Prepare the new entry
    new_entry = f"""
## {video_title}

- **URL:** {video_url}
- **Date Processed:** {current_date}

{content}

---
"""
    
    # Append to the summaries file
    with open(summaries_file, "a", encoding="utf-8") as f:
        f.write(new_entry)
    
    logger.info(f"Updated {summaries_file} with summary for {video_title}")

if __name__ == "__main__":
    main()