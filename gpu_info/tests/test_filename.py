#!/usr/bin/env python3
"""
Test script to verify the filename generation logic.
"""

import os
import re
from output_formatter import OutputFormatter

def main():
    """Test the filename generation logic."""
    # Create test directory
    test_dir = "./test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize formatter
    formatter = OutputFormatter(output_dir=test_dir)
    
    # Test with various titles
    test_cases = [
        {
            "video_id": "abc123",
            "video_title": "How to Write CUDA Kernels",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "summary": "This is a test summary",
            "detailed_content": "This is detailed content"
        },
        {
            "video_id": "def456",
            "video_title": "Advanced Triton Optimization Techniques: A Deep Dive into Performance Tuning for Machine Learning Workloads",
            "video_url": "https://www.youtube.com/watch?v=def456",
            "summary": "This is a test summary",
            "detailed_content": "This is detailed content"
        },
        {
            "video_id": "ghi789",
            "video_title": "Special Characters: !@#$%^&*()_+",
            "video_url": "https://www.youtube.com/watch?v=ghi789",
            "summary": "This is a test summary",
            "detailed_content": "This is detailed content"
        }
    ]
    
    # Process each test case
    for case in test_cases:
        output_file = formatter.format_output(case)
        print(f"Input title: {case['video_title']}")
        print(f"Output file: {os.path.basename(output_file)}")
        print()

if __name__ == "__main__":
    main()