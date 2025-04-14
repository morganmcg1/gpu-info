"""
This module contains all the prompts used in the GPU Mode Knowledge Base Extraction System.
These prompts are used to interact with the Gemini API for various tasks related to
extracting information from YouTube videos about CUDA and Triton kernels.
"""

# Content Relevance Check Prompt
CONTENT_RELEVANCE_CHECK_PROMPT = """
Analyze this video content and determine if it contains specific technical information about CUDA or Triton kernel programming, GPU architecture directly relevant to kernel writing, or performance optimization techniques for GPU kernels.

Content to analyze:
---
{content}
---

Evaluate the content based on these criteria:
1. Does it contain specific CUDA or Triton code examples, function names, or API calls?
2. Does it explain GPU architecture concepts that directly impact kernel writing (e.g., memory hierarchy, thread organization, warp execution)?
3. Does it discuss specific performance optimization techniques for GPU kernels?
4. Does it provide concrete implementation steps, debugging techniques, or profiling methods for GPU kernels?

Respond with a JSON object containing these fields:
- is_relevant: boolean (true/false)
- relevance_score: number between 0-10 (where 10 is highly relevant)
- contains_cuda_content: boolean (true/false)
- contains_triton_content: boolean (true/false)
- primary_topic: string describing what the video is actually about
- key_technical_concepts: array of strings listing specific technical concepts if any
- explanation: string explaining why this content is or isn't relevant to CUDA/Triton kernel programming

Format your response as valid JSON without any additional text before or after."""

# Chain of Density Prompts
INITIAL_SUMMARY_PROMPT = """
Create a concise initial summary (around 150 words) of the following content.

Focus *exclusively* on extracting key concepts and practical techniques relevant to:
1. Writing CUDA or Triton kernels.
2. Running or executing these kernels (especially within frameworks like PyTorch).
3. Understanding how GPUs work in relation to kernel execution (e.g., memory hierarchy, threads, blocks, warps, SMs, caches).
4. Profiling and debugging CUDA/Triton code.

Prioritize:
- Specific function names, code snippets/patterns (e.g., indexing, loading/storing data, synchronization).
- Names of specific tools (e.g., `ncu`, `torch.profiler.profile`, `load_inline`, `@triton.jit`).
- Key performance metrics or heuristics mentioned.
- GPU architectural concepts directly tied to kernel performance or writing.
- Integration methods (e.g., how to call a kernel from Python/PyTorch).

Avoid filler phrases. Be direct and technically specific.

Content to summarize:
---
{content}
---

Initial Summary:
"""

ENTITY_IDENTIFICATION_PROMPT = """
Analyze the original content and the current summary. Identify 3-5 specific, named entities or concepts that are present in the original content, are *crucial* for understanding how to write/run/profile CUDA/Triton kernels or how GPUs work, but are either *missing* or *insufficiently detailed* in the current summary.

Examples of desired entities:
- `__shared__ memory`
- `cudaMemcpy HtoD`
- `torch.profiler.profile` (specific tool)
- `load_inline` (integration technique)
- `@triton.jit` (decorator)
- `tl.load` / `tl.store` (Triton functions)
- PTX assembly
- Kernel fusion (optimization concept)
- Memory coalescing (concept)
- Warp divergence (concept)
- Specific kernel launch parameters (e.g., block size, grid size)
- Specific profiler metrics (e.g., Achieved Occupancy)
- Architectural terms (e.g., SM, Tensor Core) *if explained in relation to kernels*.

Original content:
---
{content}
---

Current summary:
---
{current_summary}
---

Return *only* a list of the missing specific entities/concepts, one per line. Do not add explanations here.
Missing Entities/Concepts:
"""

SUMMARY_REWRITE_PROMPT = """
Rewrite the "Current Summary" below to integrate the "Missing Entities/Concepts". The goal is to create a *more technically dense and informative* summary about writing, running, or understanding CUDA/Triton kernels and related GPU concepts.

Guidelines:
1. **Integrate All Missing Entities:** Seamlessly weave *all* the listed missing entities/concepts into the summary's narrative, explaining their relevance briefly if possible within the length constraint.
2. **Increase Density:** Replace vague descriptions or filler phrases from the "Current Summary" with the specific information represented by the missing entities.
3. **Maintain Core Info:** Preserve the essential technical information already present in the "Current Summary".
4. **Conciseness:** Be direct and avoid unnecessary words.
5. **Target Length:** Aim for a length similar to or slightly longer than the "Current Summary" (around {len_current_summary} words), but prioritize incorporating the new information meaningfully over strictly adhering to the word count.
6. **Focus:** Ensure the rewritten summary remains focused on CUDA/Triton kernel writing/running/profiling and GPU concepts.

Original content (for context only):
---
{content}
---

Current summary:
---
{current_summary}
---

Missing entities/concepts to include:
---
{entities_text}
---

Rewritten Dense Summary:
"""

ENTITY_DENSITY_EVALUATION_PROMPT = """
Evaluate the technical density and specificity of the following summary regarding writing, running, profiling, or understanding CUDA/Triton kernels and GPU concepts.

Does it contain actionable, concrete technical information (specific function names, tool names, code patterns, architectural details, performance concepts)?

Identify 1-2 areas where the summary could be *more specific* or add *more technical detail* based on common knowledge about this topic.

Summary to Evaluate:
---
{summary}
---

Evaluation:
"""

SUMMARY_QUALITY_EVALUATION_PROMPT = """
Act as an expert judge in CUDA/GPU programming. Evaluate how well the provided "Summary" captures the *most critical technical information* from the "Original Content" for someone learning to **write, run, profile, or understand CUDA/Triton kernels and relevant GPU architecture**.

Original Content:
---
{content}
---

Summary:
---
{summary}
---

Provide your evaluation as a JSON object with the following fields:

- "entity_coverage_score": (Integer 1-10) How well the summary covers the *essential* technical entities/concepts from the content related to the goal (kernel writing/running/understanding/profiling, GPU concepts).
- "accuracy_score": (Integer 1-10) How accurately the summary represents the technical information from the content.
- "conciseness_score": (Integer 1-10) How concise the summary is while retaining technical detail (10 is very concise and dense).
- "missing_critical_entities": (List of strings) List up to 5 *specific, critical* technical entities or concepts (function names, tools, techniques, architectural details) *essential* to the learning goal that are present in the content but missing or poorly explained in the summary.
- "irrelevant_or_vague_parts": (List of strings) List any parts of the summary that are too vague, use filler language, or are irrelevant to the core goal.
- "overall_quality_score": (Integer 1-10) Overall quality and usefulness of the summary *for the specific learning goal*.
- "improvement_suggestions": (String) Brief suggestions on how to make the summary even better for this specific purpose.
"""

# Gemini Video Processing Prompts
BASIC_VIDEO_SUMMARY_PROMPT = "Provide a technical summary of this video focusing specifically on CUDA/Triton kernel programming concepts, GPU architecture details relevant to kernel performance, and practical implementation techniques."

DETAILED_VIDEO_ANALYSIS_PROMPT = """
Analyze this YouTube video about CUDA or Triton kernels and extract highly specific technical information for developers learning to write efficient GPU kernels. Focus exclusively on:

1. Core Technical Concepts: Extract specific named functions, classes, methods, and architectural elements (e.g., `__shared__ memory`, `tl.load`, `cudaMemcpy`, SM, warp, thread block) with their precise technical definitions and usage patterns.

2. Code Examples: Capture complete, executable code snippets with proper syntax, indentation, and comments explaining key optimizations. Include both CUDA C/C++ and/or Triton Python code as presented.

3. Mathematical Formulas: Document any equations, algorithms, or mathematical patterns used to explain kernel behavior, memory access patterns, or performance characteristics.

4. Implementation Techniques: Detail specific, actionable techniques for writing efficient kernels, including exact parameter values, configuration options, and their performance implications.

5. Performance Metrics & Profiling: Note specific profiling tools, metrics (e.g., achieved occupancy, memory bandwidth utilization), and benchmark results with precise numbers when mentioned.

6. Common Pitfalls & Solutions: Identify specific error patterns, anti-patterns, and their concrete solutions with code examples where possible.

7. GPU Architecture Insights: Extract hardware-specific details that directly impact kernel writing (e.g., warp size, shared memory bank conflicts, L1/L2 cache behavior).

For each section, provide highly detailed technical information with specific examples, function names, parameter values, and performance numbers from the video. Avoid general descriptions and focus exclusively on concrete, implementable details that would enable someone to write better CUDA or Triton kernels.
"""

# Information Extraction Prompts
CODE_EXAMPLES_EXTRACTION_PROMPT = """
Extract all code examples related to CUDA or Triton kernels from the following content.
Focus on complete, executable code snippets that demonstrate specific techniques or patterns.

Content:
---
{content}
---

For each code example:
1. Extract the complete code with exact syntax, preserving all indentation, variable names, and comments
2. Provide a technical title that precisely describes what the code demonstrates (e.g., "Shared Memory Allocation in CUDA Kernel")
3. Explain the specific technical purpose and functionality of the code
4. Identify key optimization techniques or patterns used (e.g., memory coalescing, warp-level primitives)
5. Note any performance implications mentioned (with specific numbers if available)
6. Include any parameter values or configuration options that affect performance

Format each example with proper code blocks using triple backticks with the appropriate language tag (```cuda, ```python, etc.).
Ensure all code is syntactically correct and could be copied directly into an editor and compiled/executed.
"""

EQUATIONS_EXTRACTION_PROMPT = """
Extract all mathematical equations, formulas, or algorithms related to CUDA or Triton kernel programming from the following content.
Focus on equations that explain performance characteristics, memory access patterns, or computational techniques.

Content:
---
{content}
---

For each equation or algorithm:
1. Present the equation in a clear, formatted way (using markdown math notation if possible)
2. Provide a precise technical name for the equation/algorithm
3. Define all variables and terms used in the equation
4. Explain exactly how this equation relates to kernel performance or implementation
5. Include any specific values, constants, or parameters mentioned
6. Note how this equation might be used to make implementation decisions

Format your response with proper mathematical notation. For complex equations, use multi-line formatting.
Include any diagrams or visual representations described in the content if they help explain the mathematical concepts.
"""

STEPS_EXTRACTION_PROMPT = """
Extract specific, actionable steps for implementing and optimizing CUDA or Triton kernels from the following content.
Focus on concrete technical procedures that can be directly followed by a developer.

Content:
---
{content}
---

For each implementation step or process:
1. Provide a precise technical title (e.g., "Configuring Thread Block Dimensions for Matrix Multiplication")
2. List exact commands, function calls, or code patterns needed to implement this step
3. Include specific parameter values, flags, or options mentioned (e.g., exact thread block sizes)
4. Explain the technical reasoning behind this step, including its impact on performance
5. Note any hardware-specific considerations (e.g., differences between GPU architectures)
6. Include any diagnostic steps to verify correct implementation

Format your response as a numbered sequence of steps, with substeps where appropriate.
Include code snippets with proper formatting where they help illustrate the steps.
Focus exclusively on technical details that would enable direct implementation.
"""

GOTCHAS_EXTRACTION_PROMPT = """
Extract specific technical pitfalls, errors, and gotchas related to CUDA or Triton kernel programming from the following content.
Focus on concrete issues that cause bugs, performance problems, or unexpected behavior.

Content:
---
{content}
---

For each gotcha or pitfall:
1. Provide a precise technical name for the issue (e.g., "Warp Divergence in Conditional Branches")
2. Include exact error messages, symptoms, or performance impacts (with numbers if available)
3. Show problematic code patterns with specific examples
4. Explain the technical root cause of the issue at the hardware or software level
5. Provide specific, implementable solutions or workarounds with code examples
6. Include any debugging techniques or tools mentioned for identifying this issue

Format your response with clear headings for each issue.
Use code blocks with proper formatting to show both problematic and corrected code.
Include any specific GPU architecture details that relate to the gotcha (e.g., "This issue is particularly severe on Ampere GPUs").
"""

PERFORMANCE_TIPS_EXTRACTION_PROMPT = """
Extract specific, technical performance optimization techniques for CUDA or Triton kernels from the following content.
Focus on concrete, implementable optimizations with measurable impact.

Content:
---
{content}
---

For each optimization technique:
1. Provide a precise technical name (e.g., "Memory Access Coalescing via Shared Memory Tiling")
2. Include exact code patterns that implement this optimization, with before/after examples if available
3. Specify the exact performance impact with numbers (e.g., "Reduces global memory transactions by 75%")
4. Explain the hardware mechanism that makes this optimization effective
5. Note specific parameter values that affect the optimization (e.g., optimal tile sizes)
6. Include any trade-offs, limitations, or GPU architecture dependencies

Format your response with clear technical headings.
Use code blocks with proper formatting to show implementation details.
Include any benchmark results or performance measurements mentioned with precise numbers.
Note any profiling metrics that can be used to measure the effectiveness of each optimization.
"""

# YouTube Summary Extractor Prompts
CHAIN_OF_DENSITY_YOUTUBE_PROMPT = """
I need you to create a highly technical, information-dense summary of a YouTube video about CUDA/Triton kernel programming or GPU architecture. Focus exclusively on extracting specific, actionable technical information that would help someone implement better GPU kernels.

Here is the initial summary:
{initial_summary}

Video URL: {youtube_url}

Using the Chain of Density method, create a technically precise summary that includes:

1. Specific Named Functions and Methods: Extract exact function names, method signatures, and API calls (e.g., `cudaMalloc`, `tl.load`, `@triton.jit`) with their parameters and usage patterns.

2. Exact Code Patterns: Identify specific coding techniques with exact syntax (e.g., thread indexing patterns, memory access patterns, synchronization methods).

3. GPU Architecture Details: Include specific hardware concepts directly relevant to kernel writing (e.g., warp size, shared memory bank width, L1/L2 cache behavior, tensor cores).

4. Performance Numbers: Extract any specific metrics, benchmarks, or performance characteristics with exact values when mentioned.

5. Implementation Techniques: Document specific, actionable techniques for writing, debugging, or optimizing kernels with exact steps.

6. Common Errors and Solutions: Identify specific error patterns and their exact solutions.

7. Mathematical Formulas: Include any equations, algorithms, or mathematical patterns that explain kernel behavior or performance.

Avoid general descriptions, background information, or content not directly related to the technical implementation of CUDA/Triton kernels. Focus exclusively on dense, specific technical details that would enable direct implementation.
"""

INFORMATION_EXTRACTION_YOUTUBE_PROMPT = """
Extract and organize highly specific technical information from this refined summary of a CUDA/Triton kernel programming video:

{refined_summary}

For each category below, extract only concrete, implementable technical details:

1. Code Examples:
   - Extract complete, executable code snippets with exact syntax
   - Preserve all variable names, types, and comments
   - Include function signatures and parameter values
   - Note any specific optimization techniques used in the code

2. Performance Optimization Techniques:
   - Name each technique with its precise technical term
   - Explain exactly how to implement each technique (specific steps)
   - Include any parameter values or configuration options mentioned
   - Note specific performance impacts with numbers when available
   - Explain the hardware mechanism that makes each optimization effective

3. GPU Architecture Details:
   - Extract specific hardware characteristics relevant to kernel writing
   - Include exact values (e.g., warp size = 32 threads)
   - Explain how each architectural feature affects kernel implementation
   - Note any architecture-specific optimizations

4. Common Pitfalls and Solutions:
   - Identify specific error patterns with exact symptoms or error messages
   - Include problematic code examples
   - Provide specific, implementable solutions
   - Explain the technical reason behind each issue

5. Profiling and Debugging Techniques:
   - Extract specific tools, commands, and metrics mentioned
   - Include exact command-line options or API calls
   - Note how to interpret specific profiling results
   - Document any debugging workflows or techniques

Format the output as a well-structured markdown report with clear technical headings, proper code blocks with language tags, and mathematical notation where appropriate. Focus exclusively on technical details that would enable direct implementation.
"""