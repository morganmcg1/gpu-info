"""
Logging Module

This module provides a centralized logging configuration for the application.
It ensures consistent logging across all modules.
"""

import logging
import sys
import platform
from typing import Optional
from config import Config

# Initialize colorama for Windows compatibility if available
try:
    import colorama
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# Define a custom formatter with colors for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'    # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        log_message = super().format(record)
        
        # Only apply colors if terminal supports it and colorama is available on Windows
        if record.levelname in self.COLORS:
            if sys.stderr.isatty():
                # On Windows, we need colorama for ANSI color support
                if platform.system() == 'Windows' and not COLORAMA_AVAILABLE:
                    return log_message
                
                log_message = f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
                
        return log_message

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        # Get configuration
        log_level = Config.get("LOG_LEVEL", "INFO")
        log_format = Config.get("LOG_FORMAT")
        
        # Set level
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(log_format))
        logger.addHandler(console_handler)
        
        # Create file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
        elif Config.get("LOG_FILE"):
            file_handler = logging.FileHandler(Config.get("LOG_FILE"))
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
    
    return logger

def setup_root_logger():
    """Set up the root logger with the application's configuration."""
    # Get configuration
    log_level = Config.get("LOG_LEVEL", "INFO")
    log_format = Config.get("LOG_FORMAT")
    log_file = Config.get("LOG_FILE")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("pytube").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    
    return root_logger