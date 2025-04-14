# GPU Info Examples

This directory contains example scripts and output files demonstrating how to use the GPU Info system.

## Example Scripts

### Process Video Example

The `process_video_example.py` script demonstrates how to use the GPU Info system to process a YouTube video and extract information about CUDA/Triton kernels.

Usage:
```bash
./process_video_example.py <video_url> [output_dir]
```

Example:
```bash
./process_video_example.py "https://www.youtube.com/watch?v=D7_ipDqhtwk" "./my_output"
```

This script:
1. Processes the video with the Gemini API
2. Applies the Chain of Density method to refine the summary
3. Extracts specific information about code examples, equations, performance tips, etc.
4. Formats the output as a structured Markdown report

## Example Output

The `sample_output.md` file demonstrates the expected output format for a processed video. It includes:

- Video information (title, URL, duration, author)
- Comprehensive summary
- Detailed content:
  - Key concepts
  - Code examples
  - Equations
  - Performance tips
  - Gotchas and warnings
  - Profiling workflow
  - Conclusion

This sample output is based on a CUDA profiling and optimization video and shows the level of detail and structure that the system aims to achieve.

## Using the Examples

These examples are intended to help you understand how to use the GPU Info system. You can:

1. Study the example script to understand the processing pipeline
2. Examine the sample output to see what kind of information is extracted
3. Modify the example script to customize the processing for your needs
4. Use the example script as a starting point for your own applications

For more detailed information about the GPU Info system, see the main README.md file in the repository root.