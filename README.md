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

## Installation

### Standard Installation

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

### Docker Installation

1. Clone the repository:
   ```
   git clone https://github.com/morganmcg1/gpu-info.git
   cd gpu-info
   ```

2. Create a `.env` file with your Google API key:
   ```
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up -d web
   ```

4. To process videos in batch mode:
   ```
   docker-compose up processor
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

Or use the dedicated batch processing script:

```
python batch_process_videos.py --video_list "video_list.txt" --output_dir "summaries" --max_workers 1
```

Where `video_list.txt` contains one YouTube URL per line.

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

### Web Viewer for Summaries

To view the generated summaries in a web browser:

```
python web_viewer.py --host 0.0.0.0 --port 12000 --summaries_dir "summaries"
```

Then open your browser and navigate to `http://localhost:12000` to view the summaries.

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
- `web_viewer.py`: Web interface for viewing generated summaries
- `final_youtube_test.py`: Test script for the Gemini API with YouTube videos

## Example Summaries

The repository includes example summaries that demonstrate the level of detail and structure we aim to achieve:

1. [Triton Kernel Basics](summaries/triton_kernel_basics.md)
2. [CUDA Optimization Techniques](summaries/cuda_optimization_techniques.md)

## Limitations

There are some limitations with directly processing YouTube videos using the Gemini API. See [README_LIMITATIONS.md](README_LIMITATIONS.md) for details and workarounds.

## License

[MIT License](LICENSE)