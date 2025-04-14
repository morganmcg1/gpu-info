# Limitations and Troubleshooting

## Direct YouTube Video Processing with Gemini API

We encountered issues when attempting to directly process YouTube videos with the Gemini API as described in the [Gemini by Example documentation](https://geminibyexample.com/015-youtube-video-summarization/).

### Issues Encountered

1. **API Timeouts**: When attempting to call the Gemini API with YouTube URLs, the requests timed out after extended periods, both for longer videos (~56 minutes) and shorter videos (~4 minutes).

2. **YouTube Access**: There may be restrictions in the current environment that prevent the Gemini API from accessing YouTube content directly.

### Code Attempted

We tried multiple approaches:

1. Using the file_data parameter with mime_type:
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

2. Using the exact API format from the documentation:
```python
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        {
            "parts": [
                {"text": "Can you summarize this video?"},
                {"file_data": {"file_uri": youtube_url}},
            ]
        }
    ],
)
```

3. Trying different model versions:
```python
response = client.models.generate_content(
    model="gemini-2.5-pro-preview-03-25",  # Also tried gemini-2.0-flash
    contents=[
        {
            "parts": [
                {"text": "Can you summarize this video?"},
                {"file_data": {"file_uri": youtube_url}},
            ]
        }
    ],
)
```

4. Testing with different video lengths:
   - Short video (3:52 minutes): https://www.youtube.com/watch?v=D7_ipDqhtwk
   - Medium video (56 minutes): https://www.youtube.com/watch?v=LuhJEEJQgUM
   - Very short video (3:52 minutes): https://www.youtube.com/watch?v=Fwoyko4uuvI

**Note**: The very short video (Fwoyko4uuvI) did work with gemini-2.0-flash model, but longer videos consistently timed out.

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