"""
AI Layer - Integrates GenConViT model for deepfake detection
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Add GenConViT to path
genconvit_path = Path(__file__).parent.parent.parent / "GenConViT"
sys.path.append(str(genconvit_path))

from dff_framework.core.base_layer import BaseForensicLayer

# Import GenConViT modules with proper path handling
import os
original_cwd = os.getcwd()
try:
    os.chdir(str(genconvit_path))
    from model.pred_func import load_genconvit, df_face, pred_vid, real_or_fake
    from model.config import load_config
finally:
    os.chdir(original_cwd)

class AILayer(BaseForensicLayer):
    """
    AI Layer using GenConViT model for deepfake detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("AI_Layer", config)
        self.model = None
        self.config_genconvit = None
        self._load_model()
    
    def _load_model(self):
        """Load the GenConViT model"""
        try:
            # Change to GenConViT directory for proper config loading
            original_cwd = os.getcwd()
            genconvit_path = Path(__file__).parent.parent.parent / "GenConViT"
            
            try:
                os.chdir(str(genconvit_path))
                # Load GenConViT configuration
                self.config_genconvit = load_config()
                
                # Model parameters
                net = self.config.get('net', 'genconvit')
                ed_weight = self.config.get('ed_weight', 'genconvit_ed_inference')
                vae_weight = self.config.get('vae_weight', 'genconvit_vae_inference')
                fp16 = self.config.get('fp16', False)
                num_frames = self.config.get('num_frames', 15)
                
                # Load the model
                self.model = load_genconvit(
                    self.config_genconvit, 
                    net, 
                    ed_weight, 
                    vae_weight, 
                    fp16
                )
                
                print(f"GenConViT model loaded successfully (net: {net})")
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"Error loading GenConViT model: {str(e)}")
            self.model = None
    
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video using GenConViT model
        
        Args:
            video_path: Path to the video file
            options: Analysis options
            
        Returns:
            AI analysis results
        """
        if not self.validate_input(video_path):
            return {
                "status": "error",
                "error": "Invalid video file"
            }
        
        if self.model is None:
            return {
                "status": "error",
                "error": "Model not loaded"
            }
        
        try:
            # Extract options
            num_frames = options.get('num_frames', self.config.get('num_frames', 15)) if options else self.config.get('num_frames', 15)
            
            # Change to GenConViT directory for proper function calls
            original_cwd = os.getcwd()
            genconvit_path = Path(__file__).parent.parent.parent / "GenConViT"
            
            try:
                os.chdir(str(genconvit_path))
                
                # Extract faces from video
                print(f"Extracting {num_frames} frames from video...")
                df = df_face(video_path, num_frames)
                
                if len(df) == 0:
                    return {
                        "status": "error",
                        "error": "No faces detected in video"
                    }
                
                # Make prediction
                print("Running AI analysis...")
                y, y_val = pred_vid(df, self.model)
                
            finally:
                os.chdir(original_cwd)
            
            # Determine prediction
            prediction = real_or_fake(y)
            confidence = y_val
            
            # Calculate additional metrics
            fake_probability = y_val if prediction == "FAKE" else 1 - y_val
            real_probability = 1 - fake_probability
            
            # Generate detailed results
            results = {
                "status": "success",
                "prediction": prediction,
                "confidence": confidence,
                "fake_probability": fake_probability,
                "real_probability": real_probability,
                "model_info": {
                    "model_name": "GenConViT",
                    "net_type": self.config.get('net', 'genconvit'),
                    "num_frames_analyzed": len(df),
                    "frames_requested": num_frames
                },
                "technical_details": {
                    "raw_prediction": y,
                    "raw_confidence": y_val,
                    "model_loaded": True
                }
            }
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Analysis failed: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "GenConViT",
            "description": "Generative Convolutional Vision Transformer for deepfake detection",
            "net_type": self.config.get('net', 'genconvit'),
            "model_loaded": self.model is not None,
            "config": self.config
        }
