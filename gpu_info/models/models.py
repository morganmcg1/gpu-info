"""
This module contains Pydantic models for structured output from the Gemini API.
These models define the expected structure of responses from various LLM prompts.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Information about a YouTube video."""
    video_id: str = Field(description="The YouTube video ID")
    title: str = Field(description="The title of the video")
    author: str = Field(description="The author/channel of the video")
    length: int = Field(description="The length of the video in seconds")
    url: str = Field(description="The URL of the video")
    description: Optional[str] = Field(description="The description of the video")
    transcript: Optional[str] = Field(description="The transcript of the video")
    thumbnail_url: Optional[str] = Field(description="The URL of the video thumbnail")
    publish_date: Optional[str] = Field(description="The publish date of the video")
    views: Optional[int] = Field(description="The number of views of the video")
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata about the video")


class DetailedContent(BaseModel):
    """Detailed content extracted from a video."""
    video_url: str = Field(description="The URL of the video")
    detailed_content: str = Field(description="The detailed content extracted from the video")
    is_cuda_content: bool = Field(description="Whether the video contains CUDA/Triton kernel content")
    content_type: str = Field(description="The type of content (e.g., 'tutorial', 'lecture', 'demo')")
    technical_level: str = Field(description="The technical level of the content (e.g., 'beginner', 'intermediate', 'advanced')")
    key_topics: List[str] = Field(description="The key topics covered in the content")


class CodeExample(BaseModel):
    """A code example extracted from the content."""
    title: str = Field(description="A technical title that precisely describes what the code demonstrates")
    code: str = Field(description="The complete code with exact syntax, preserving all indentation, variable names, and comments")
    language: str = Field(description="The programming language of the code (e.g., 'cuda', 'python', 'c++')")
    purpose: str = Field(description="The specific technical purpose and functionality of the code")
    optimization_techniques: List[str] = Field(description="Key optimization techniques or patterns used in the code")
    performance_implications: Optional[str] = Field(description="Any performance implications mentioned with specific numbers if available")
    parameters: Optional[dict] = Field(description="Parameter values or configuration options that affect performance")


class Equation(BaseModel):
    """A mathematical equation or algorithm extracted from the content."""
    equation: str = Field(description="The equation in a clear, formatted way using markdown math notation")
    name: str = Field(description="A precise technical name for the equation/algorithm")
    variables: dict = Field(description="Definitions of all variables and terms used in the equation")
    relation_to_performance: str = Field(description="Explanation of how this equation relates to kernel performance or implementation")
    specific_values: Optional[dict] = Field(description="Any specific values, constants, or parameters mentioned")
    implementation_usage: str = Field(description="How this equation might be used to make implementation decisions")


class ImplementationStep(BaseModel):
    """A specific, actionable step for implementing and optimizing CUDA or Triton kernels."""
    title: str = Field(description="A precise technical title for the implementation step")
    commands: List[str] = Field(description="Exact commands, function calls, or code patterns needed to implement this step")
    parameters: Optional[dict] = Field(description="Specific parameter values, flags, or options mentioned")
    technical_reasoning: str = Field(description="The technical reasoning behind this step, including its impact on performance")
    hardware_considerations: Optional[str] = Field(description="Any hardware-specific considerations")
    diagnostic_steps: Optional[List[str]] = Field(description="Any diagnostic steps to verify correct implementation")


class Gotcha(BaseModel):
    """A specific technical pitfall, error, or gotcha related to CUDA or Triton kernel programming."""
    name: str = Field(description="A precise technical name for the issue")
    symptoms: List[str] = Field(description="Exact error messages, symptoms, or performance impacts")
    problematic_code: Optional[str] = Field(description="Problematic code patterns with specific examples")
    root_cause: str = Field(description="The technical root cause of the issue at the hardware or software level")
    solutions: List[str] = Field(description="Specific, implementable solutions or workarounds")
    debugging_techniques: Optional[List[str]] = Field(description="Any debugging techniques or tools mentioned for identifying this issue")


class PerformanceTip(BaseModel):
    """A specific, technical performance optimization technique for CUDA or Triton kernels."""
    name: str = Field(description="A precise technical name for the optimization technique")
    code_pattern: Optional[str] = Field(description="Exact code patterns that implement this optimization")
    performance_impact: str = Field(description="The exact performance impact with numbers")
    hardware_mechanism: str = Field(description="The hardware mechanism that makes this optimization effective")
    parameters: Optional[dict] = Field(description="Specific parameter values that affect the optimization")
    trade_offs: Optional[List[str]] = Field(description="Any trade-offs, limitations, or GPU architecture dependencies")
    profiling_metrics: Optional[List[str]] = Field(description="Profiling metrics that can be used to measure the effectiveness")


class InitialSummary(BaseModel):
    """An initial summary of the content."""
    summary: str = Field(description="A concise initial summary of the content focusing on CUDA/Triton kernel programming concepts")


class EntityList(BaseModel):
    """A list of missing entities or concepts."""
    entities: List[str] = Field(description="A list of specific, named entities or concepts that are missing from the current summary")


class DenseSummary(BaseModel):
    """A technically dense and informative summary."""
    summary: str = Field(description="A more technically dense and informative summary about writing, running, or understanding CUDA/Triton kernels")


class SummaryEvaluation(BaseModel):
    """An evaluation of the technical density and specificity of a summary."""
    entity_coverage_score: int = Field(description="How well the summary covers the essential technical entities/concepts (1-10)")
    accuracy_score: int = Field(description="How accurately the summary represents the technical information (1-10)")
    conciseness_score: int = Field(description="How concise the summary is while retaining technical detail (1-10)")
    missing_critical_entities: List[str] = Field(description="List of specific, critical technical entities or concepts missing from the summary")
    irrelevant_or_vague_parts: List[str] = Field(description="Parts of the summary that are too vague, use filler language, or are irrelevant")
    overall_quality_score: int = Field(description="Overall quality and usefulness of the summary (1-10)")
    improvement_suggestions: str = Field(description="Brief suggestions on how to make the summary better")


class VideoAnalysis(BaseModel):
    """A comprehensive analysis of a video about CUDA or Triton kernels."""
    core_technical_concepts: List[str] = Field(description="Specific named functions, classes, methods, and architectural elements with their precise technical definitions and usage patterns")
    code_examples: List[CodeExample] = Field(description="Complete, executable code snippets with proper syntax, indentation, and comments")
    mathematical_formulas: List[Equation] = Field(description="Equations, algorithms, or mathematical patterns used to explain kernel behavior")
    implementation_techniques: List[ImplementationStep] = Field(description="Specific, actionable techniques for writing efficient kernels")
    performance_metrics: dict = Field(description="Specific profiling tools, metrics, and benchmark results with precise numbers")
    common_pitfalls: List[Gotcha] = Field(description="Specific error patterns, anti-patterns, and their concrete solutions")
    gpu_architecture_insights: List[str] = Field(description="Hardware-specific details that directly impact kernel writing")


class ExtractedInformation(BaseModel):
    """All information extracted from the content."""
    code_examples: List[CodeExample] = Field(description="All code examples related to CUDA or Triton kernels")
    equations: List[Equation] = Field(description="All mathematical equations, formulas, or algorithms")
    implementation_steps: List[ImplementationStep] = Field(description="Specific, actionable steps for implementing and optimizing kernels")
    gotchas: List[Gotcha] = Field(description="Specific technical pitfalls, errors, and gotchas")
    performance_tips: List[PerformanceTip] = Field(description="Specific, technical performance optimization techniques")
