"""
Safe AI Layer - Works without GenConViT model weights for testing
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from dff_framework.core.base_layer import BaseForensicLayer

class SafeAILayer(BaseForensicLayer):
    """
    Safe AI Layer that provides mock results when GenConViT is not available
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("AI_Layer", config)
        self.model = None
        self.model_available = False
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if GenConViT model is available"""
        try:
            # Check if GenConViT directory exists
            genconvit_path = Path(__file__).parent.parent.parent / "GenConViT"
            if not genconvit_path.exists():
                print("GenConViT directory not found - using mock mode")
                return
            
            # Check if model weights exist
            weight_path = genconvit_path / "weight"
            ed_weight = weight_path / "genconvit_ed_inference.pth"
            vae_weight = weight_path / "genconvit_vae_inference.pth"
            
            if ed_weight.exists() and vae_weight.exists():
                print("GenConViT model weights found - attempting to load...")
                self._load_model()
            else:
                print("GenConViT model weights not found - using mock mode")
                print(f"Expected weights at: {ed_weight} and {vae_weight}")
                
        except Exception as e:
            print(f"Error checking model availability: {str(e)}")
            print("Using mock mode")
    
    def _load_model(self):
        """Load the GenConViT model"""
        try:
            # Add GenConViT to path
            genconvit_path = Path(__file__).parent.parent.parent / "GenConViT"
            sys.path.append(str(genconvit_path))
            
            # Change to GenConViT directory for proper config loading
            original_cwd = os.getcwd()
            
            try:
                os.chdir(str(genconvit_path))
                
                # Import GenConViT modules
                from model.pred_func import load_genconvit, df_face, pred_vid, real_or_fake
                from model.config import load_config
                
                # Load GenConViT configuration
                config_genconvit = load_config()
                
                # Model parameters
                net = self.config.get('net', 'genconvit')
                ed_weight = self.config.get('ed_weight', 'genconvit_ed_inference')
                vae_weight = self.config.get('vae_weight', 'genconvit_vae_inference')
                fp16 = self.config.get('fp16', False)
                
                # Load the model
                self.model = load_genconvit(
                    config_genconvit, 
                    net, 
                    ed_weight, 
                    vae_weight, 
                    fp16
                )
                
                # Store functions for later use
                self.df_face = df_face
                self.pred_vid = pred_vid
                self.real_or_fake = real_or_fake
                
                self.model_available = True
                print(f"✅ GenConViT model loaded successfully (net: {net})")
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"❌ Error loading GenConViT model: {str(e)}")
            self.model = None
            self.model_available = False
    
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video using GenConViT model or mock results
        
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
        
        if not self.model_available:
            return self._get_mock_results(video_path, options)
        
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
                # Use the original video path (it's already absolute from Gradio)
                df = self.df_face(video_path, num_frames)
                
                if len(df) == 0:
                    return {
                        "status": "error",
                        "error": "No faces detected in video"
                    }
                
                # Make prediction
                print("Running AI analysis...")
                y, y_val = self.pred_vid(df, self.model)
                
            finally:
                os.chdir(original_cwd)
            
            # Determine prediction
            prediction = self.real_or_fake(y)
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
    
    def _get_mock_results(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate mock results when model is not available"""
        import random
        
        # Generate mock prediction
        mock_prediction = random.choice(["REAL", "FAKE"])
        mock_confidence = random.uniform(0.6, 0.95)
        
        fake_probability = mock_confidence if mock_prediction == "FAKE" else 1 - mock_confidence
        real_probability = 1 - fake_probability
        
        return {
            "status": "success",
            "prediction": mock_prediction,
            "confidence": mock_confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "model_info": {
                "model_name": "GenConViT (Mock Mode)",
                "net_type": "mock",
                "num_frames_analyzed": 0,
                "frames_requested": options.get('num_frames', 15) if options else 15
            },
            "technical_details": {
                "raw_prediction": 0 if mock_prediction == "REAL" else 1,
                "raw_confidence": mock_confidence,
                "model_loaded": False,
                "note": "Mock results - GenConViT model not available"
            },
            "warning": "This is a mock result. Download GenConViT model weights for real analysis."
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "GenConViT",
            "description": "Generative Convolutional Vision Transformer for deepfake detection",
            "net_type": self.config.get('net', 'genconvit'),
            "model_loaded": self.model_available,
            "config": self.config
        }
