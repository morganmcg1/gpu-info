# GPU Info Test Scripts

This directory contains test scripts for verifying the functionality of the GPU Info system.

## Test Scripts

- `test_cuda_video.py`: Test script for processing a CUDA video using both direct Gemini API and the full pipeline
- `test_llm_client.py`: Test script for the LLM client with structured output using Pydantic models
- `test_gemini_video.py`: Test script for direct Gemini API video processing

## Usage

### Test with Default CUDA Video

Process the default test CUDA video (a short CUDA crash course):

```bash
python -m scripts.test_scripts.test_cuda_video --method gemini
```

Options:
- `--method`: Choose between `gemini` (direct Gemini API), `pipeline` (full processing pipeline), or `both` (default)
- `--output_dir`: Specify the output directory (default: `test_output`)

### Test LLM Client

Test the LLM client functionality with structured output:

```bash
python -m scripts.test_scripts.test_llm_client
```

### Test Direct Gemini API

Test direct YouTube video processing with the Gemini API:

```bash
python -m scripts.test_scripts.test_gemini_video
```

## Configuration

The test scripts use the default test video URL from the `gpu_info/config/test_videos.json` file. You can modify this file to use a different test video.