"""
Base class for all forensic analysis layers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

class BaseForensicLayer(ABC):
    """
    Abstract base class for all forensic analysis layers
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze the video file and return results
        
        Args:
            video_path: Path to the video file
            options: Analysis options and parameters
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def validate_input(self, video_path: str) -> bool:
        """Validate input video file"""
        return os.path.isfile(video_path) and video_path.lower().endswith(
            ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        )
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about this layer"""
        return {
            "name": self.name,
            "description": self.__doc__ or "No description available",
            "config": self.config
        }
