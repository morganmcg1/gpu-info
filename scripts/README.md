# GPU Info Scripts

This directory contains command-line scripts for processing YouTube videos about CUDA and Triton kernels.

## Scripts

- `main.py`: Main script for processing a single YouTube video
- `batch_process_videos.py`: Script for processing multiple YouTube videos from a list
- `youtube_summary_extractor.py`: Standalone script for generating a summary of a YouTube video using the Chain of Density method

## Test Scripts

The `test_scripts/` directory contains scripts for testing the functionality of the system:

- `test_cuda_video.py`: Test script for processing a CUDA video
- `test_llm_client.py`: Test script for the LLM client
- `test_gemini_video.py`: Test script for direct Gemini API video processing

## Usage

### As Command-Line Scripts

After installing the package, you can use the scripts as command-line tools:

```bash
# Process a single video
process-video --video_url "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir "./output"

# Process multiple videos
batch-process --video_list "youtube_links.txt" --output_dir "./output"

# Generate a summary
youtube-summary --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "summary.md"
```

### As Python Modules

You can also run the scripts as Python modules:

```bash
# Process a single video
python -m scripts.main --video_url "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir "./output"

# Process multiple videos
python -m scripts.batch_process_videos --video_list "youtube_links.txt" --output_dir "./output"

# Generate a summary
python -m scripts.youtube_summary_extractor --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "summary.md"
```

### Test Scripts

```bash
# Test with default CUDA video
python -m scripts.test_scripts.test_cuda_video --method gemini

# Test LLM client
python -m scripts.test_scripts.test_llm_client

# Test direct Gemini API
python -m scripts.test_scripts.test_gemini_video
```