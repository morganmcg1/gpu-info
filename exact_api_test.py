"""
Test script using the exact API format provided.
"""

import os
import logging
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("exact_api_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Process a YouTube video using the exact API format provided."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    # Initialize Gemini client
    logger.info("Initializing Gemini client")
    client = genai.Client(api_key=api_key)
    
    # Test video URL - shorter test video (3:52)
    youtube_url = "https://www.youtube.com/watch?v=D7_ipDqhtwk"
    logger.info(f"Processing video: {youtube_url}")
    
    try:
        # Using the exact API format provided
        logger.info("Sending request to Gemini API with exact format")
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=[
                {
                    "parts": [
                        {"text": "Can you summarize this video?"},
                        {"file_data": {"file_uri": youtube_url}},
                    ]
                }
            ],
        )
        
        logger.info("Response received from Gemini API")
        
        # Save the response to a file
        with open("exact_api_response.md", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        logger.info(f"Response saved to exact_api_response.md")
        
        # Print the first 500 characters of the response
        preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
        logger.info(f"Response preview: {preview}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()