"""
Test script to directly process a YouTube video with Gemini.
"""

import os
import logging
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("direct_gemini_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Process a test video directly with Gemini."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Test video URL
    video_url = "https://www.youtube.com/watch?v=D7_ipDqhtwk"
    
    # Process the video
    try:
        logger.info(f"Processing video: {video_url}")
        
        # Create a prompt to summarize the video
        prompt = """
        Please analyze this YouTube video about CUDA or Triton kernels and extract the following information:
        
        1. A comprehensive summary of the key points
        2. Any code examples shown or discussed
        3. Mathematical equations or formulas presented
        4. Step-by-step processes explained
        5. Common pitfalls or gotchas mentioned
        6. Performance optimization techniques discussed
        
        For each section, provide detailed information with specific examples from the video.
        Focus on technical details that would be valuable for someone learning to write CUDA or Triton kernels.
        """
        
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
        
        # Print the response
        logger.info("Response received from Gemini")
        print(response.text)
        
        # Save the response to a file
        with open("output/direct_gemini_response.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info("Response saved to output/direct_gemini_response.txt")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()