"""
Base class for all forensic analysis layers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseForensicLayer(ABC):
    """
    Abstract base class for all forensic analysis layers
    """
    
    def __init__(self, name: str):
        """
        Initialize the forensic layer
        
        Args:
            name: Name of the layer
        """
        self.name = name
    
    @abstractmethod
    def analyze(self, video_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a video file
        
        Args:
            video_path: Path to the video file
            options: Optional analysis options
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this layer
        
        Returns:
            Layer name
        """
        return self.name
