"""
This module contains all the prompts used in the GPU Mode Knowledge Base Extraction System.
These prompts are used to interact with the Gemini API for various tasks related to
extracting information from YouTube videos about CUDA and Triton kernels.
"""

# Chain of Density Prompts
INITIAL_SUMMARY_PROMPT = """
Create an initial summary of the following content. The summary should be:
- About 150-200 words in length
- Capture the main points but be somewhat verbose
- Use filler phrases like "this content discusses" where appropriate

Content to summarize:
{content}
"""

ENTITY_IDENTIFICATION_PROMPT = """
Identify 3-5 important entities from the original content that are missing from the current summary.

An entity is a specific, concrete piece of information such as:
- Technical terms related to CUDA or Triton kernels
- Specific code patterns or techniques
- Performance numbers or benchmarks
- Names of specific algorithms or optimization strategies
- Specific hardware details or requirements

Original content:
{content}

Current summary:
{current_summary}

Return only a list of the missing entities, one per line, with no additional text.
"""

SUMMARY_REWRITE_PROMPT = """
Rewrite the current summary to include the missing entities while maintaining approximately the same length.

Guidelines:
- The new summary should be about the same length as the current summary (approximately {len_current_summary} words)
- Include all the missing entities
- Make the summary more concise by removing filler phrases and redundant information
- Maintain the key information from the current summary
- Focus on technical details and practical insights about CUDA and Triton kernels

Original content:
{content}

Current summary:
{current_summary}

Missing entities to include:
{entities_text}
"""

ENTITY_DENSITY_EVALUATION_PROMPT = """
Evaluate the entity density of the following summary. Count the number of specific, concrete entities related to CUDA and Triton kernels.

An entity is a specific, concrete piece of information such as:
- Technical terms related to CUDA or Triton kernels
- Specific code patterns or techniques
- Performance numbers or benchmarks
- Names of specific algorithms or optimization strategies
- Specific hardware details or requirements

Summary:
{summary}

Return a JSON object with the following fields:
- entity_count: The number of entities found
- word_count: The number of words in the summary
- entity_density: The ratio of entities to words
- entities_found: A list of the entities found
"""

SUMMARY_QUALITY_EVALUATION_PROMPT = """
Act as a judge to evaluate how well the summary extracts important entities from the original content.
Focus specifically on technical details related to CUDA and Triton kernels.

Original content:
{content}

Summary:
{summary}

Provide your evaluation as a JSON object with the following fields:
- entity_coverage: A score from 1-10 indicating how well the summary covers important entities
- missing_key_entities: List of important entities that are missing from the summary
- irrelevant_entities: List of entities in the summary that aren't important
- overall_quality: A score from 1-10 indicating the overall quality of the summary
- suggestions: Suggestions for improving the summary
"""

# Gemini Video Processing Prompts
BASIC_VIDEO_SUMMARY_PROMPT = "Can you summarize this video?"

DETAILED_VIDEO_ANALYSIS_PROMPT = """
Please analyze this YouTube video about CUDA or Triton kernels and extract the following information:

1. A comprehensive summary of the key points
2. Any code examples shown or discussed
3. Mathematical equations or formulas presented
4. Step-by-step processes explained
5. Common pitfalls or gotchas mentioned
6. Performance optimization techniques discussed

For each section, provide detailed information with specific examples from the video.
Focus on technical details that would be valuable for someone learning to write CUDA or Triton kernels.
"""

# Information Extraction Prompts
CODE_EXAMPLES_EXTRACTION_PROMPT = """
Extract all code examples related to CUDA or Triton kernels from the following content.
Include the full code and a brief description of what each example demonstrates.

Content:
{content}

For each code example, provide:
1. The complete code snippet
2. A description of what the code demonstrates
3. Any key points or optimizations shown in the code

Format your response as a list of examples, with each example clearly separated.
Ensure that code formatting is preserved with proper indentation.
"""

EQUATIONS_EXTRACTION_PROMPT = """
Extract all mathematical equations or formulas related to CUDA or Triton kernels from the following content.
Include a clear explanation of each equation and its relevance.

Content:
{content}

For each equation, provide:
1. The equation itself (in a clear format)
2. An explanation of what the equation represents
3. How this equation is relevant to CUDA or Triton kernel programming

Format your response as a list of equations, with each equation clearly separated.
"""

STEPS_EXTRACTION_PROMPT = """
Extract the key steps or processes for writing effective CUDA or Triton kernels from the following content.
Focus on practical, actionable steps that developers should follow.

Content:
{content}

For each key step or process, provide:
1. A clear title or name for the step
2. A detailed explanation of what this step involves
3. Why this step is important
4. Any tips or best practices related to this step

Format your response as a numbered list of steps, with each step clearly separated.
"""

GOTCHAS_EXTRACTION_PROMPT = """
Extract all gotchas, warnings, and common pitfalls related to CUDA or Triton kernel programming from the following content.
Focus on issues that developers commonly encounter and how to avoid them.

Content:
{content}

For each gotcha or warning, provide:
1. A clear title that describes the issue
2. A detailed explanation of the problem
3. Why this is a common issue or mistake
4. How to avoid or solve this problem
5. Any specific examples mentioned

Format your response as a list of gotchas, with each gotcha clearly separated.
"""

PERFORMANCE_TIPS_EXTRACTION_PROMPT = """
Extract all performance optimization tips and techniques for CUDA or Triton kernels from the following content.
Focus on specific, actionable advice that can improve kernel performance.

Content:
{content}

For each performance tip, provide:
1. A clear title for the optimization technique
2. A detailed explanation of the technique
3. When and why this technique is effective
4. Any benchmarks or performance improvements mentioned
5. Any trade-offs or considerations to keep in mind

Format your response as a list of performance tips, with each tip clearly separated.
"""

# YouTube Summary Extractor Prompts
CHAIN_OF_DENSITY_YOUTUBE_PROMPT = """
I want you to help me create a more detailed and information-dense summary of a YouTube video about CUDA or Triton kernels.

Here is the initial summary:
{initial_summary}

Video URL: {youtube_url}

Using the Chain of Density method, please:

1. Identify specific entities, concepts, techniques, code examples, and performance tips from the video
2. Create a more detailed summary that includes:
   - Key technical concepts explained in the video
   - Specific code examples or patterns mentioned
   - Performance optimization techniques
   - Common pitfalls or "gotchas" when writing CUDA/Triton kernels
   - Mathematical equations or algorithms discussed
   - Benchmark results or performance comparisons

Focus on extracting high-signal, technical information that would be valuable for someone learning to write CUDA or Triton kernels.
"""

INFORMATION_EXTRACTION_YOUTUBE_PROMPT = """
I have a refined summary of a YouTube video about CUDA or Triton kernels:

{refined_summary}

Please extract and organize the following specific types of information:

1. Code Examples: Extract any complete code examples, ensuring they are properly formatted and include all necessary context

2. Performance Optimization Techniques: List specific techniques mentioned for optimizing CUDA or Triton kernels

3. Mathematical Equations: Extract any mathematical equations or algorithms discussed

4. Common Pitfalls: Identify common mistakes or "gotchas" when writing CUDA or Triton kernels

5. Benchmark Results: Extract any specific performance numbers or comparisons

Format the output as a well-structured markdown report with clear sections and code blocks where appropriate.
"""