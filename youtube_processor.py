"""
YouTube Processor Module

This module handles fetching and processing YouTube videos for summarization.
It uses the pytube library to download video information and transcripts.
"""

import os
import time
from typing import Dict, Optional, Tuple
from pytube import YouTube
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeProcessor:
    """Class to handle YouTube video processing."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the YouTube processor.
        
        Args:
            cache_dir: Directory to cache downloaded transcripts
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized YouTubeProcessor with cache directory: {cache_dir}")
    
    def get_video_info(self, video_url: str) -> Dict:
        """
        Get basic information about a YouTube video.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Dictionary containing video information
            
        Raises:
            ValueError: If the video URL is invalid
            ConnectionError: If there are network issues
            Exception: For other unexpected errors
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        # Extract video ID from URL to help with error recovery
        try:
            # Basic extraction, assumes standard YouTube URL format
            if "youtu.be" in video_url:
                video_id = video_url.split("/")[-1].split("?")[0]
            elif "youtube.com/watch" in video_url:
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(video_url)
                video_id = parse_qs(parsed_url.query).get('v', [''])[0]
            else:
                video_id = None
                
            if not video_id:
                logger.warning(f"Could not extract video ID from URL: {video_url}")
        except Exception as e:
            logger.warning(f"Error extracting video ID: {str(e)}")
            video_id = None
            
        # Retry mechanism for network issues
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                yt = YouTube(video_url)
                
                # Sometimes the first connection may succeed but getting attributes fails
                # Try to access attributes with additional error handling
                try:
                    title = yt.title
                    author = yt.author
                    description = yt.description or "No description available"
                except Exception as attr_err:
                    logger.warning(f"Error getting video attributes: {str(attr_err)}")
                    # Fallback to basic info if we couldn't get full details
                    if video_id:
                        logger.info(f"Using partial information for video ID: {video_id}")
                        return {
                            "title": f"YouTube Video {video_id}",
                            "author": "Unknown",
                            "length": 0,
                            "views": 0,
                            "publish_date": None,
                            "description": "Could not retrieve full video details",
                            "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                            "video_id": video_id
                        }
                    else:
                        # We have no fallback information, re-raise the error
                        raise
                
                video_info = {
                    "title": yt.title,
                    "author": yt.author,
                    "length": yt.length,
                    "views": yt.views,
                    "publish_date": yt.publish_date,
                    "description": yt.description or "No description available",
                    "thumbnail_url": yt.thumbnail_url,
                    "video_id": yt.video_id
                }
                logger.info(f"Successfully retrieved info for video: {yt.title}")
                return video_info
            except ConnectionError as e:
                logger.warning(f"Network error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Could not connect to YouTube.")
                    # If we have the video_id, return partial information
                    if video_id:
                        logger.info(f"Using partial information for video ID: {video_id}")
                        return {
                            "title": f"YouTube Video {video_id}",
                            "author": "Unknown",
                            "length": 0,
                            "views": 0,
                            "publish_date": None,
                            "description": "Could not retrieve video details due to connection error",
                            "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                            "video_id": video_id
                        }
                    raise ConnectionError(f"Failed to connect to YouTube after {max_retries} attempts: {str(e)}")
            except Exception as e:
                logger.error(f"Error retrieving video info: {str(e)}")
                # If we have the video_id and this is the last attempt, return partial information
                if video_id and attempt == max_retries - 1:
                    logger.info(f"Using partial information for video ID: {video_id}")
                    return {
                        "title": f"YouTube Video {video_id}",
                        "author": "Unknown",
                        "length": 0,
                        "views": 0,
                        "publish_date": None,
                        "description": f"Could not retrieve video details: {str(e)}",
                        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                        "video_id": video_id
                    }
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def get_transcript(self, video_url: str) -> Optional[str]:
        """
        Get the transcript of a YouTube video.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            String containing the transcript or None if not available
            
        Raises:
            ValueError: If the video URL is invalid
        """
        if not video_url or not isinstance(video_url, str):
            logger.error("Invalid YouTube URL provided")
            raise ValueError("Invalid YouTube URL provided")
            
        try:
            video_id = YouTube(video_url).video_id
            cache_path = os.path.join(self.cache_dir, f"{video_id}_transcript.txt")
            
            # Check if transcript is already cached
            if os.path.exists(cache_path):
                logger.info(f"Loading transcript from cache for video ID: {video_id}")
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        transcript = f.read()
                    if transcript and len(transcript) > 0:
                        return transcript
                    else:
                        logger.warning(f"Cached transcript for video ID {video_id} is empty, will try to fetch again")
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Error reading cached transcript: {str(e)}. Will try to fetch again.")
            
            # Note: pytube doesn't directly support transcript extraction
            # This is a placeholder - in a real implementation, we would use
            # a service like YouTube Data API or youtube-transcript-api
            logger.warning("Direct transcript extraction not implemented in pytube")
            logger.info("For a production system, use youtube-transcript-api or YouTube Data API")
            
            # For now, we'll return None to indicate that the transcript is not available
            # In the main application, we'll use the Gemini API to process the video directly
            return None
        except ConnectionError as e:
            logger.error(f"Network error retrieving transcript: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving transcript: {str(e)}")
            return None
    
    def process_video(self, video_url: str) -> Tuple[Dict, Optional[str]]:
        """
        Process a YouTube video to get both info and transcript.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Tuple of (video_info, transcript)
        """
        video_info = self.get_video_info(video_url)
        transcript = self.get_transcript(video_url)
        return video_info, transcript