# GPU Mode Knowledge Base

This repository contains tools for extracting high-quality information from YouTube videos about CUDA and Triton kernels. It uses the Gemini 2.5 Pro model to analyze videos and extract key learnings, code examples, equations, and best practices.

## Overview

This project aims to create a high-quality knowledge base from YouTube videos about CUDA and Triton kernel programming. It uses the Gemini API and the Chain of Density method to extract detailed, high-signal information from educational content.

## Features

- Process YouTube videos directly using the Gemini API
- Apply the Chain of Density method to create high-quality summaries
- Extract specific types of information:
  - Code examples with explanations
  - Mathematical equations and formulas
  - Step-by-step processes
  - Common gotchas and warnings
  - Performance optimization techniques
- Generate well-structured Markdown reports
- Process multiple videos in parallel for improved efficiency
- Centralized configuration system for easy customization
- Robust logging with colored console output

## Installation



1. Clone the repository:
   ```
   git clone https://github.com/morganmcg1/gpu-info.git
   cd gpu-info
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```
   export GOOGLE_API_KEY="your_api_key_here"
   ```

4. (Optional) Configure additional settings via environment variables:
   ```
   # Example: Set maximum parallel workers
   export GPU_INFO_MAX_WORKERS=5
   
   # Example: Set output directory
   export GPU_INFO_OUTPUT_DIR="./my_summaries"
   
   # Example: Set logging level
   export GPU_INFO_LOG_LEVEL="DEBUG"
   ```



## Usage

### Process a single video (Main Script)

```
python main.py --video_url "https://www.youtube.com/watch?v=example"
```

### Process multiple videos from a list

```
python main.py --video_list "video_list.txt"
```

Or use the dedicated batch processing script with parallel processing:

```
python batch_process_videos.py --video_list "video_list.txt" --output_dir "summaries" --max_workers 3
```

Where:
- `video_list.txt` contains one YouTube URL per line
- `--max_workers` controls the number of videos processed in parallel (default: 3)
- `--model_id` can be used to specify a different Gemini model (default: gemini-2.5-pro-preview-03-25)
- `--log_level` sets the logging level (DEBUG, INFO, WARNING, ERROR)
- `--force_reprocess` forces reprocessing of already processed videos
- `--api_key` can be used to provide a Google API key directly

### Specify an output directory

```
python main.py --video_url "https://www.youtube.com/watch?v=example" --output_dir "./my_output"
```

### Direct YouTube Summary Extractor

For a simpler approach that directly processes a YouTube video and applies the Chain of Density method:

```
python youtube_summary_extractor.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "output_summary.md"
```

Example:

```
python youtube_summary_extractor.py --url "https://www.youtube.com/watch?v=D7_ipDqhtwk" --output "cuda_profiling_summary.md"
```



## Output

The system generates two types of output files for each processed video:

1. **Markdown Report** (.md): A well-structured report containing:
   - Video information
   - Comprehensive summary
   - Code examples
   - Key equations
   - Step-by-step processes
   - Gotchas and warnings
   - Performance optimization tips

2. **JSON Data** (.json): Raw data containing all extracted information.

## Components

- `youtube_processor.py`: Handles YouTube video fetching and processing
- `chain_of_density.py`: Implements the Chain of Density summarization method
- `information_extractor.py`: Extracts specific types of information
- `output_formatter.py`: Formats the output into structured reports
- `gemini_video_processor.py`: Processes videos directly with the Gemini API
- `main.py`: The main script that ties everything together
- `youtube_summary_extractor.py`: Standalone script for direct YouTube video processing with Chain of Density
- `batch_process_videos.py`: Script for processing multiple YouTube videos in batch
- `parallel_processor.py`: Handles parallel processing of multiple videos
- `prompts.py`: Contains all prompts used throughout the system
- `config.py`: Centralized configuration system
- `logger.py`: Enhanced logging functionality



## Example Summaries

The repository includes example summaries that demonstrate the level of detail and structure we aim to achieve:

1. [Triton Kernel Basics](summaries/triton_kernel_basics.md)
2. [CUDA Optimization Techniques](summaries/cuda_optimization_techniques.md)

## Configuration

The system uses a centralized configuration system that can be customized in several ways:

### Environment Variables

You can configure the system using environment variables with the prefix `GPU_INFO_`:

```bash
# API Settings
export GPU_INFO_GEMINI_MODEL_ID="gemini-2.5-pro-preview-03-25"
export GPU_INFO_GEMINI_API_KEY="your_api_key_here"  # Alternative to GOOGLE_API_KEY

# Processing Settings
export GPU_INFO_MAX_WORKERS=3
export GPU_INFO_TIMEOUT=300
export GPU_INFO_MAX_RETRIES=3
export GPU_INFO_RETRY_DELAY=2
export GPU_INFO_FORCE_REPROCESS=false

# Output Settings
export GPU_INFO_OUTPUT_DIR="./summaries"
export GPU_INFO_CACHE_DIR="./cache"

# Logging Settings
export GPU_INFO_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export GPU_INFO_LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
export GPU_INFO_LOG_FILE="gpu_info.log"

# Chain of Density Settings
export GPU_INFO_COD_ITERATIONS=3
```

### Command Line Arguments

Most scripts accept command line arguments that override the configuration:

```bash
python batch_process_videos.py \
  --video_list "video_list.txt" \
  --output_dir "summaries" \
  --max_workers 3 \
  --model_id "gemini-2.5-pro-preview-03-25" \
  --log_level "INFO" \
  --force_reprocess
```

### Programmatic Configuration

You can also modify the configuration programmatically:

```python
from config import Config

# Get configuration values
model_id = Config.get("GEMINI_MODEL_ID")
max_workers = Config.get("MAX_WORKERS")

# Set configuration values
Config.set("MAX_WORKERS", 5)
Config.set("LOG_LEVEL", "DEBUG")
```

## Limitations

There are some limitations with directly processing YouTube videos using the Gemini API. See [README_LIMITATIONS.md](README_LIMITATIONS.md) for details and workarounds.

## License

[MIT License](LICENSE)