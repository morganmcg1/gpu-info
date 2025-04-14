#!/usr/bin/env python3
"""
Test script to verify the code structure.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing all the modules."""
    logger.info("Testing imports...")
    
    # Core modules
    from gpu_info.core.llm_client import get_llm_client
    from gpu_info.core.youtube_processor import YouTubeProcessor
    from gpu_info.core.gemini_video_processor import GeminiVideoProcessor
    
    # Analysis modules
    from gpu_info.analysis.chain_of_density import ChainOfDensity
    from gpu_info.analysis.information_extractor import InformationExtractor
    
    # Output modules
    from gpu_info.output.output_formatter import OutputFormatter
    
    # Models
    from gpu_info.models.models import (
        InitialSummary, 
        EntityList, 
        DenseSummary, 
        SummaryEvaluation,
        VideoAnalysis,
        CodeExample,
        Equation,
        ImplementationStep,
        Gotcha,
        PerformanceTip,
        ExtractedInformation
    )
    
    # Utils
    from gpu_info.utils.config import Config
    from gpu_info.utils.logger import get_logger, setup_root_logger
    from gpu_info.utils.prompts import (
        INITIAL_SUMMARY_PROMPT,
        ENTITY_IDENTIFICATION_PROMPT,
        SUMMARY_REWRITE_PROMPT,
        ENTITY_DENSITY_EVALUATION_PROMPT,
        SUMMARY_QUALITY_EVALUATION_PROMPT,
        CODE_EXAMPLES_EXTRACTION_PROMPT,
        EQUATIONS_EXTRACTION_PROMPT,
        STEPS_EXTRACTION_PROMPT,
        GOTCHAS_EXTRACTION_PROMPT,
        PERFORMANCE_TIPS_EXTRACTION_PROMPT,
        BASIC_VIDEO_SUMMARY_PROMPT,
        DETAILED_VIDEO_ANALYSIS_PROMPT
    )
    
    # Parallel
    from gpu_info.parallel.parallel_processor import process_videos_parallel
    
    logger.info("All imports successful!")
    return True

def test_directory_structure():
    """Test the directory structure."""
    logger.info("Testing directory structure...")
    
    # Check that all required directories exist
    required_dirs = [
        "gpu_info/core",
        "gpu_info/analysis",
        "gpu_info/output",
        "gpu_info/models",
        "gpu_info/utils",
        "gpu_info/parallel",
        "gpu_info/logs",
        "gpu_info/cache",
        "gpu_info/tests",
        "gpu_info/config",
        "gpu_info/examples",
        "scripts"
    ]
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            logger.error(f"Directory {directory} does not exist!")
            return False
    
    logger.info("All required directories exist!")
    return True

def main():
    """Main function."""
    logger.info("Testing code structure...")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test directory structure
    dirs_ok = test_directory_structure()
    
    if imports_ok and dirs_ok:
        logger.info("All tests passed! The code structure is correct.")
        return 0
    else:
        logger.error("Some tests failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())