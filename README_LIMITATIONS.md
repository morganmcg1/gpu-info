# Limitations and Troubleshooting

## Direct YouTube Video Processing with Gemini API

We encountered issues when attempting to directly process YouTube videos with the Gemini API as described in the [Gemini by Example documentation](https://geminibyexample.com/015-youtube-video-summarization/).

### Issues Encountered

1. **API Timeouts**: When attempting to call the Gemini API with YouTube URLs, the requests timed out after extended periods, both for longer videos (~56 minutes) and shorter videos (~4 minutes).

2. **YouTube Access**: There may be restrictions in the current environment that prevent the Gemini API from accessing YouTube content directly.

### Code Attempted

We tried two different approaches:

1. Using the file_data parameter:
```python
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
```

2. Using the approach from geminibyexample.com (adapted for the current API):
```python
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
```

### Workarounds

1. **Manual Summaries**: For demonstration purposes, we've created manual summaries based on video content.

2. **Alternative Approaches**: In a production environment, you could:
   - Download the videos and process them locally
   - Extract audio and transcribe it first
   - Use a different API or service for video processing

### Requirements for Direct YouTube Processing

For the direct YouTube processing to work in a production environment, you would need:

1. An API key with appropriate permissions
2. Network access that allows the Gemini API to access YouTube content
3. Proper configuration of the API request
4. Sufficient timeout settings for longer videos

## Next Steps

1. Test the API in a different environment with fewer restrictions
2. Implement alternative approaches for video processing
3. Contact Google support for guidance on direct YouTube video processing with Gemini