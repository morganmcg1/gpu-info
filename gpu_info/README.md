# GPU Info Package

This package contains modules for extracting high-quality information from YouTube videos about CUDA and Triton kernels.

## Package Structure

- `core/`: Core functionality for processing YouTube videos and interacting with the Gemini API
  - `youtube_processor.py`: Handles YouTube video fetching and processing
  - `gemini_video_processor.py`: Processes videos directly with the Gemini API
  - `llm_client.py`: Centralized client for interacting with the Gemini API

- `analysis/`: Analysis modules for extracting information from video content
  - `chain_of_density.py`: Implements the Chain of Density method for refining summaries
  - `information_extractor.py`: Extracts specific types of information from video content

- `output/`: Output formatting and saving
  - `output_formatter.py`: Formats extracted information into structured Markdown reports

- `models/`: Data models
  - `models.py`: Pydantic models for structured data

- `utils/`: Utility modules
  - `config.py`: Configuration management
  - `logger.py`: Logging utilities
  - `prompts.py`: Centralized prompts for the system

- `parallel/`: Parallel processing utilities
  - `parallel_processor.py`: Processes multiple videos concurrently

- `config/`: Configuration files
  - `test_videos.json`: Test video URLs

## Usage

The package is designed to be used as a library or through the provided scripts in the `scripts/` directory.

### As a Library

```python
from gpu_info.core.youtube_processor import YouTubeProcessor
from gpu_info.analysis.chain_of_density import ChainOfDensity
from gpu_info.analysis.information_extractor import InformationExtractor
from gpu_info.output.output_formatter import OutputFormatter
from gpu_info.core.gemini_video_processor import GeminiVideoProcessor
from gpu_info.core.llm_client import get_llm_client

# Initialize components
youtube_processor = YouTubeProcessor(cache_dir="./cache")
llm_client = get_llm_client(api_key="YOUR_API_KEY")
chain_of_density = ChainOfDensity(api_key="YOUR_API_KEY")
information_extractor = InformationExtractor(api_key="YOUR_API_KEY")
output_formatter = OutputFormatter(output_dir="./output")
gemini_processor = GeminiVideoProcessor(api_key="YOUR_API_KEY")

# Process a video
video_info, _ = youtube_processor.process_video("https://www.youtube.com/watch?v=VIDEO_ID")
detailed_content = gemini_processor.extract_detailed_content("https://www.youtube.com/watch?v=VIDEO_ID")
content = detailed_content["detailed_content"]

# Apply Chain of Density
summaries = chain_of_density.apply_chain_of_density(content, iterations=3)
final_summary = summaries[-1]

# Extract specific information
extracted_info = information_extractor.extract_all_information(content)

# Format and save the results
output_paths = output_formatter.format_and_save(video_info, final_summary, extracted_info)
```

### Using the Scripts

The package provides several scripts for processing YouTube videos:

- `process-video`: Process a single YouTube video
- `batch-process`: Process multiple YouTube videos from a list
- `youtube-summary`: Generate a summary of a YouTube video using the Chain of Density method

These scripts can be run from the command line after installing the package:

```bash
# Process a single video
process-video --video_url "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir "./output"

# Process multiple videos
batch-process --video_list "youtube_links.txt" --output_dir "./output"

# Generate a summary
youtube-summary --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "summary.md"
```

Or you can run the Python modules directly:

```bash
# Process a single video
python -m scripts.main --video_url "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir "./output"

# Process multiple videos
python -m scripts.batch_process_videos --video_list "youtube_links.txt" --output_dir "./output"

# Generate a summary
python -m scripts.youtube_summary_extractor --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "summary.md"
```

## Configuration

The package can be configured using environment variables or programmatically:

### Environment Variables

- `GOOGLE_API_KEY`: Google API key for the Gemini API
- `GEMINI_MODEL_ID`: Gemini model ID to use (default: "gemini-2.5-pro-preview-03-25")
- `OUTPUT_DIR`: Directory to save output (default: "./output")
- `MAX_WORKERS`: Maximum number of concurrent workers for parallel processing (default: 3)
- `LOG_LEVEL`: Logging level (default: "INFO")

### Programmatic Configuration

```python
from gpu_info.utils.config import Config

Config.set("GEMINI_MODEL_ID", "gemini-2.5-pro-preview-03-25")
Config.set("OUTPUT_DIR", "./output")
Config.set("MAX_WORKERS", 3)
Config.set("LOG_LEVEL", "INFO")
```