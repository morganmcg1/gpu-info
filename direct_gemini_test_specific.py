"""
Test script to directly process a specific YouTube video with Gemini.
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
        logging.FileHandler("direct_gemini_test_specific.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_video_title(video_url):
    """Get the title of a YouTube video using pytube."""
    try:
        from pytube import YouTube
        yt = YouTube(video_url)
        return yt.title
    except Exception as e:
        logger.error(f"Error getting video title: {str(e)}")
        return "Unknown Title"

def process_video(video_url, api_key):
    """Process a specific YouTube video with Gemini."""
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    logger.info(f"Processing video: {video_url}")
    
    # Get video title
    video_title = get_video_title(video_url)
    logger.info(f"Video title: {video_title}")
    
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
        # Call the Gemini API with the video URL
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"file_uri": video_url}}
                    ]
                }
            ]
        )
        
        # Get the response text
        response_text = response.text
        
        # Create a summary object
        summary = {
            "title": video_title,
            "url": video_url,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "content": response_text
        }
        
        # Save the response to a file
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f"{output_dir}/{video_title.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Save as Markdown
        with open(f"{output_dir}/{video_title.replace(' ', '_')}.md", "w", encoding="utf-8") as f:
            f.write(f"# {video_title}\n\n")
            f.write(f"- **URL:** {video_url}\n")
            f.write(f"- **Date Processed:** {summary['date']}\n\n")
            f.write(response_text)
        
        logger.info(f"Response saved to {output_dir}/{video_title.replace(' ', '_')}.md")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None

def main():
    """Process a test video directly with Gemini."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Test video URL
    video_url = "https://www.youtube.com/watch?v=LuhJEEJQgUM"
    
    # Process the video
    summary = process_video(video_url, api_key)
    
    # Update summaries.md file
    if summary:
        update_summaries_file(summary)

def update_summaries_file(summary):
    """Update the summaries.md file with the new summary."""
    summaries_file = "summaries.md"
    
    # Create or append to the summaries file
    mode = "a" if os.path.exists(summaries_file) else "w"
    
    with open(summaries_file, "a" if os.path.exists(summaries_file) else "w", encoding="utf-8") as f:
        if mode == "w":
            f.write("# GPU Mode Knowledge Base Summaries\n\n")
            f.write("This file contains summaries of various YouTube videos about CUDA and Triton kernels.\n\n")
        
        f.write(f"## {summary['title']}\n\n")
        f.write(f"- **URL:** {summary['url']}\n")
        f.write(f"- **Date Processed:** {summary['date']}\n\n")
        f.write(summary['content'])
        f.write("\n\n---\n\n")  # Delimiter between summaries
    
    logger.info(f"Updated {summaries_file} with summary for {summary['title']}")

if __name__ == "__main__":
    main()