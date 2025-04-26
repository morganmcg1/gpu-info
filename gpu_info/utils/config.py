"""
Configuration Module

This module centralizes all configuration settings for the application.
It provides default values and can be overridden by environment variables.
"""

import os
import logging
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # API Settings
    "GEMINI_MODEL_ID": "gemini-2.5-pro-preview-03-25",
    "GEMINI_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
    
    # Processing Settings
    "MAX_WORKERS": 3,
    "TIMEOUT": 300,  # seconds
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 2,  # seconds
    
    # Output Settings
    "OUTPUT_DIR": "./summaries",
    "CACHE_DIR": "./cache",
    
    # Logging Settings
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE": "gpu_info.log",
    
    # Chain of Density Settings
    "COD_ITERATIONS": 3,
    
    # YouTube Settings
    "YOUTUBE_TRANSCRIPT_LANG": "en",
}

# Override with environment variables
for key in DEFAULT_CONFIG:
    env_key = f"GPU_INFO_{key}"
    if env_key in os.environ:
        # Convert to appropriate type
        original_value = DEFAULT_CONFIG[key]
        env_value = os.environ[env_key]
        
        if isinstance(original_value, bool):
            DEFAULT_CONFIG[key] = env_value.lower() in ('true', 'yes', '1')
        elif isinstance(original_value, int):
            DEFAULT_CONFIG[key] = int(env_value)
        elif isinstance(original_value, float):
            DEFAULT_CONFIG[key] = float(env_value)
        else:
            DEFAULT_CONFIG[key] = env_value

# Ensure API key is set
if not DEFAULT_CONFIG["GEMINI_API_KEY"]:
    DEFAULT_CONFIG["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

class Config:
    """Configuration class to access settings."""
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return DEFAULT_CONFIG.get(key, default)
    
    @staticmethod
    def get_all() -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary with all configuration values
        """
        return DEFAULT_CONFIG.copy()
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        DEFAULT_CONFIG[key] = value

# Configure logging
def setup_logging(log_file: str = None, log_level: str = None):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (optional)
    """
    if log_level is None:
        log_level = Config.get("LOG_LEVEL", "INFO")
    
    if log_file is None:
        log_file = Config.get("LOG_FILE")
    
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=Config.get("LOG_FORMAT"),
        handlers=handlers
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("pytube").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)