"""
Output Formatter Module

This module formats the extracted information into well-structured reports.
"""

import os
import json
import logging
from typing import Dict, List, Any
import markdown
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputFormatter:
    """Class to format extracted information into structured reports."""
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the Output Formatter.
        
        Args:
            output_dir: Directory to save formatted outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized OutputFormatter with output directory: {output_dir}")
    
    def format_markdown(self, video_info: Dict[str, Any], summary: str, 
                        extracted_info: Dict[str, List[Dict[str, str]]]) -> str:
        """
        Format the extracted information as a Markdown document.
        
        Args:
            video_info: Information about the video
            summary: The final summary
            extracted_info: Dictionary of extracted information
            
        Returns:
            Formatted Markdown string
        """
        logger.info("Formatting output as Markdown")
        
        # Format the current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        md = f"""# {video_info['title']}

## Video Information
- **Author:** {video_info.get('author', 'Unknown')}
- **Publication Date:** {video_info.get('publish_date', 'Unknown')}
- **Video URL:** https://www.youtube.com/watch?v={video_info.get('video_id', '')}
- **Summary Date:** {current_date}

## Summary
{summary}

"""
        
        # Add code examples section if available
        if extracted_info.get('code_examples'):
            md += "## Code Examples\n\n"
            for i, example in enumerate(extracted_info['code_examples'], 1):
                if 'raw_extraction' in example:
                    md += example['raw_extraction'] + "\n\n"
                else:
                    md += f"### Example {i}\n\n"
                    if 'description' in example:
                        md += f"{example['description']}\n\n"
                    if 'code' in example:
                        md += f"```\n{example['code']}\n```\n\n"
                    if 'key_points' in example:
                        md += f"**Key Points:**\n{example['key_points']}\n\n"
        
        # Add equations section if available
        if extracted_info.get('equations'):
            md += "## Key Equations\n\n"
            for i, equation in enumerate(extracted_info['equations'], 1):
                if 'raw_extraction' in equation:
                    md += equation['raw_extraction'] + "\n\n"
                else:
                    md += f"### Equation {i}\n\n"
                    if 'equation' in equation:
                        md += f"```\n{equation['equation']}\n```\n\n"
                    if 'explanation' in equation:
                        md += f"{equation['explanation']}\n\n"
                    if 'relevance' in equation:
                        md += f"**Relevance:** {equation['relevance']}\n\n"
        
        # Add key steps section if available
        if extracted_info.get('key_steps'):
            md += "## Key Steps\n\n"
            for i, step in enumerate(extracted_info['key_steps'], 1):
                if 'raw_extraction' in step:
                    md += step['raw_extraction'] + "\n\n"
                else:
                    md += f"### Step {i}: {step.get('title', '')}\n\n"
                    if 'explanation' in step:
                        md += f"{step['explanation']}\n\n"
                    if 'importance' in step:
                        md += f"**Importance:** {step['importance']}\n\n"
                    if 'tips' in step:
                        md += f"**Tips:** {step['tips']}\n\n"
        
        # Add gotchas section if available
        if extracted_info.get('gotchas'):
            md += "## Gotchas and Warnings\n\n"
            for i, gotcha in enumerate(extracted_info['gotchas'], 1):
                if 'raw_extraction' in gotcha:
                    md += gotcha['raw_extraction'] + "\n\n"
                else:
                    md += f"### Gotcha {i}: {gotcha.get('title', '')}\n\n"
                    if 'explanation' in gotcha:
                        md += f"{gotcha['explanation']}\n\n"
                    if 'why_common' in gotcha:
                        md += f"**Why this is common:** {gotcha['why_common']}\n\n"
                    if 'solution' in gotcha:
                        md += f"**Solution:** {gotcha['solution']}\n\n"
                    if 'examples' in gotcha:
                        md += f"**Examples:** {gotcha['examples']}\n\n"
        
        # Add performance tips section if available
        if extracted_info.get('performance_tips'):
            md += "## Performance Optimization Tips\n\n"
            for i, tip in enumerate(extracted_info['performance_tips'], 1):
                if 'raw_extraction' in tip:
                    md += tip['raw_extraction'] + "\n\n"
                else:
                    md += f"### Tip {i}: {tip.get('title', '')}\n\n"
                    if 'explanation' in tip:
                        md += f"{tip['explanation']}\n\n"
                    if 'effectiveness' in tip:
                        md += f"**When effective:** {tip['effectiveness']}\n\n"
                    if 'benchmarks' in tip:
                        md += f"**Benchmarks:** {tip['benchmarks']}\n\n"
                    if 'trade_offs' in tip:
                        md += f"**Trade-offs:** {tip['trade_offs']}\n\n"
        
        return md
    
    def save_markdown(self, video_id: str, markdown_content: str) -> str:
        """
        Save the Markdown content to a file.
        
        Args:
            video_id: ID of the video
            markdown_content: Formatted Markdown content
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.output_dir, f"{video_id}.md")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Saved Markdown report to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving Markdown report: {str(e)}")
            raise
    
    def save_json(self, video_id: str, data: Dict[str, Any]) -> str:
        """
        Save the raw data as JSON.
        
        Args:
            video_id: ID of the video
            data: Data to save
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.output_dir, f"{video_id}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved JSON data to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving JSON data: {str(e)}")
            raise
    
    def format_and_save(self, video_info: Dict[str, Any], summary: str, 
                        extracted_info: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
        """
        Format and save the extracted information.
        
        Args:
            video_info: Information about the video
            summary: The final summary
            extracted_info: Dictionary of extracted information
            
        Returns:
            Dictionary with paths to saved files
        """
        video_id = video_info.get('video_id', 'unknown')
        
        # Format as Markdown
        markdown_content = self.format_markdown(video_info, summary, extracted_info)
        markdown_path = self.save_markdown(video_id, markdown_content)
        
        # Save raw data as JSON
        data = {
            'video_info': video_info,
            'summary': summary,
            'extracted_info': extracted_info
        }
        json_path = self.save_json(video_id, data)
        
        return {
            'markdown_path': markdown_path,
            'json_path': json_path
        }
        
    def format_output(self, result: Dict[str, Any]) -> str:
        """
        Format and save output from Gemini API results.
        
        Args:
            result: Dictionary with video URL, ID, summary, and detailed content
            
        Returns:
            Path to the saved markdown file
        """
        logger.info(f"Formatting output for video ID: {result.get('video_id', 'unknown')}")
        
        video_id = result.get('video_id', 'unknown')
        video_url = result.get('video_url', '')
        summary = result.get('summary', '')
        detailed_content = result.get('detailed_content', '')
        
        # Create video info dictionary
        video_info = {
            'video_id': video_id,
            'title': f"YouTube Video {video_id}",
            'author': 'Unknown',
            'publish_date': 'Unknown',
            'url': video_url
        }
        
        # Save the detailed content as the main summary
        file_path = os.path.join(self.output_dir, f"{video_id}_summary.md")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Summary for Video {video_id}\n\n")
                f.write(f"## Video URL\n{video_url}\n\n")
                f.write(f"## Basic Summary\n{summary}\n\n")
                f.write(f"## Detailed Content\n{detailed_content}\n\n")
            
            logger.info(f"Saved formatted output to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving formatted output: {str(e)}")
            raise