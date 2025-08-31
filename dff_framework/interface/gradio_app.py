"""
Gradio interface for the Deepfake Forensics Framework
"""

import gradio as gr
import json
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from dff_framework.core.framework import DeepfakeForensicsFramework
from dff_framework.layers.ai_layer_safe import SafeAILayer
from dff_framework.layers.metadata_layer import MetadataLayer
from dff_framework.layers.frequency_layer import FrequencyLayer
from dff_framework.layers.physics_layer import PhysicsLayer
from dff_framework.layers.contextual_layer import ContextualLayer
from dff_framework.layers.llm_analysis_layer import LLMAnalysisLayer

class DFFGradioInterface:
    """
    Gradio interface for the Deepfake Forensics Framework
    """
    
    def __init__(self):
        self.framework = DeepfakeForensicsFramework()
        self._setup_layers()
    
    def _setup_layers(self):
        """Setup and register analysis layers"""
        # Register AI Layer (GenConViT - Safe version)
        ai_layer = SafeAILayer({
            'net': 'genconvit',
            'ed_weight': 'genconvit_ed_inference',
            'vae_weight': 'genconvit_vae_inference',
            'fp16': False,
            'num_frames': 15
        })
        self.framework.register_layer("AI_Layer", ai_layer)
        
        # Register Metadata Layer
        metadata_layer = MetadataLayer()
        self.framework.register_layer("Metadata_Layer", metadata_layer)
        
        # Register Frequency Layer
        frequency_layer = FrequencyLayer({
            'dct_block_size': 8,
            'fft_threshold': 0.1,
            'compression_quality_range': (30, 100)
        })
        self.framework.register_layer("Frequency_Layer", frequency_layer)
        
        # Register Physics Layer
        physics_layer = PhysicsLayer({
            'shadow_threshold': 0.3,
            'reflection_threshold': 0.2,
            'geometry_threshold': 0.1,
            'continuity_threshold': 0.4,
            'min_object_area': 1000,
            'max_objects': 10
        })
        self.framework.register_layer("Physics_Layer", physics_layer)
        
        # Register Contextual Layer
        contextual_layer = ContextualLayer({
            'reverse_search_threshold': 0.8,
            'propagation_threshold': 0.6,
            'uploader_credibility_threshold': 0.5,
            'metadata_correlation_threshold': 0.7
        })
        self.framework.register_layer("Contextual_Layer", contextual_layer)
        
        # Register LLM Analysis Layer
        llm_layer = LLMAnalysisLayer()
        self.framework.register_layer("LLM_Analysis_Layer", llm_layer)
        
        print("Analysis layers registered successfully")
    
    def analyze_video(self, video_file, num_frames: int = 15, include_metadata: bool = True) -> tuple:
        """
        Analyze video file and return results
        
        Args:
            video_file: Uploaded video file
            num_frames: Number of frames to analyze
            include_metadata: Whether to include metadata analysis
            
        Returns:
            Tuple of (results_text, results_json, recommendations)
        """
        if video_file is None:
            return "Please upload a video file", "{}", "Please upload a video file to begin analysis."
        
        try:
            # Prepare analysis options
            options = {
                'num_frames': num_frames,
                'include_metadata': include_metadata
            }
            
            # Run analysis
            results = self.framework.analyze_video(video_file.name, options)
            
            # Format results for display
            results_text = self._format_results_text(results)
            try:
                results_json = json.dumps(results, indent=2)
            except (TypeError, ValueError) as json_error:
                results_json = json.dumps({"error": f"JSON serialization failed: {str(json_error)}", "status": "error"}, indent=2)
            recommendations = self._format_recommendations(results)
            
            return results_text, results_json, recommendations
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            error_json = json.dumps({"error": error_msg, "status": "failed"}, indent=2)
            return error_msg, error_json, error_msg
    
    def analyze_video_with_visuals(self, video_file, num_frames: int = 15, include_metadata: bool = True) -> tuple:
        """
        Analyze video file and return both visual and detailed results
        
        Args:
            video_file: Uploaded video file
            num_frames: Number of frames to analyze
            include_metadata: Whether to include metadata analysis
            
        Returns:
            Tuple of visual and detailed results
        """
        if video_file is None:
            # Return empty visual content for no video
            empty_visuals = (None, None, None, None, None, None, None, None)
            empty_llm = ("*No analysis performed yet*", "*No recommendations available*")
            return ("<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><h4>Overall Confidence: <span style='color: #666;'>0%</span></h4></div>",
                   "<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><h3>Risk Level: <span style='color: #666;'>Unknown</span></h3></div>",
                   "<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><p>No analysis performed yet</p></div>",
                   "<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>AI Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>",
                   "<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Frequency Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>",
                   "<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Physics Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>",
                   "<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Contextual Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>") + empty_visuals + empty_llm + ("Please upload a video file", "{}", "Please upload a video file to begin analysis.")
        
        try:
            # Prepare analysis options
            options = {
                'num_frames': num_frames,
                'include_metadata': include_metadata
            }
            
            # Run analysis
            results = self.framework.analyze_video(video_file.name, options)
            
            # Generate visual results
            visual_results = self._generate_visual_results(results)
            
            # Generate visual content (images, galleries, etc.)
            visual_content = self._generate_visual_content(results, video_file.name)
            
            # Generate LLM analysis results
            llm_results = self._generate_llm_results(results)
            
            # Format detailed results
            results_text = self._format_results_text(results)
            try:
                # Clean the results for JSON serialization
                cleaned_results = self._clean_for_json(results)
                results_json = json.dumps(cleaned_results, indent=2, ensure_ascii=False)
            except Exception as e:
                # If any JSON serialization fails, create a minimal safe JSON
                print(f"JSON serialization error: {str(e)}")
                safe_results = {
                    "status": "success",
                    "error": f"JSON serialization failed: {str(e)}",
                    "analysis_results": {
                        "AI_Layer": {"status": "success", "prediction": "Unknown", "confidence": 0.0},
                        "Metadata_Layer": {"status": "success", "anomalies_detected": 0},
                        "Frequency_Layer": {"status": "success", "anomalies_detected": 0},
                        "Physics_Layer": {"status": "success", "anomalies_detected": 0},
                        "Contextual_Layer": {"status": "success", "anomalies_detected": 0},
                        "LLM_Analysis_Layer": {"status": "success", "expert_opinion": "Analysis completed but JSON serialization failed"}
                    },
                    "summary": {"total_layers": 6, "successful_layers": 6, "failed_layers": 0, "overall_confidence": 0.0}
                }
                results_json = json.dumps(safe_results, indent=2)
            recommendations = self._format_recommendations(results)
            
            return visual_results + visual_content + llm_results + (results_text, results_json, recommendations)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            error_json = json.dumps({"error": error_msg, "status": "failed"}, indent=2)
            # Return empty visual content for errors
            empty_visuals = (None, None, None, None, None, None, None, None)
            empty_llm = ("*LLM analysis failed*", "*Error in analysis*")
            return ("<div style='text-align: center; padding: 10px; background: #ffebee; border-radius: 5px;'><h4>Overall Confidence: <span style='color: #d32f2f;'>Error</span></h4></div>",
                   "<div style='text-align: center; padding: 10px; background: #ffebee; border-radius: 5px;'><h3>Risk Level: <span style='color: #d32f2f;'>Error</span></h3></div>",
                   "<div style='text-align: center; padding: 10px; background: #ffebee; border-radius: 5px;'><p>Analysis failed</p></div>",
                   "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>AI Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>",
                   "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Frequency Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>",
                   "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Physics Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>",
                   "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Contextual Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>") + empty_visuals + empty_llm + (error_msg, error_json, error_msg)
    
    def _generate_visual_results(self, results: Dict[str, Any]) -> tuple:
        """Generate visual results for the interface"""
        analysis_results = results.get('analysis_results', {})
        summary = results.get('summary', {})
        
        # Calculate overall confidence
        overall_conf = summary.get('overall_confidence', 0.0)
        overall_conf_html = f"<div style='text-align: center; padding: 10px; background: #e3f2fd; border-radius: 5px;'><h4>Overall Confidence: <span style='color: #1976d2;'>{overall_conf:.1%}</span></h4></div>"
        
        # Determine risk level
        risk_level_html = self._generate_risk_level_html(results)
        
        # Generate layer status
        layer_status_html = self._generate_layer_status_html(analysis_results)
        
        # Generate individual layer visualizations
        ai_viz = self._generate_ai_layer_viz(analysis_results.get('AI_Layer', {}))
        frequency_viz = self._generate_frequency_layer_viz(analysis_results.get('Frequency_Layer', {}))
        physics_viz = self._generate_physics_layer_viz(analysis_results.get('Physics_Layer', {}))
        contextual_viz = self._generate_contextual_layer_viz(analysis_results.get('Contextual_Layer', {}))
        
        return (overall_conf_html, risk_level_html, layer_status_html, ai_viz, frequency_viz, physics_viz, contextual_viz)
    
    def _generate_visual_content(self, results: Dict[str, Any], video_path: str) -> tuple:
        """Generate visual content (images, galleries, etc.) for the visual analysis tab"""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import io
        import base64
        
        try:
            # Extract frames from video for visualization
            frames = self._extract_frames_for_visualization(video_path, max_frames=6)
            
            # Generate AI layer visualizations
            ai_frames_display, ai_heatmap = self._generate_ai_visualizations(results, frames)
            
            # Generate frequency layer visualizations
            frequency_analysis, dct_visualization = self._generate_frequency_visualizations(results, frames)
            
            # Generate physics layer visualizations
            physics_analysis, shadow_analysis = self._generate_physics_visualizations(results, frames)
            
            # Generate contextual layer visualizations
            contextual_analysis, reverse_search_results = self._generate_contextual_visualizations(results, frames)
            
            return (ai_frames_display, ai_heatmap, frequency_analysis, dct_visualization,
                   physics_analysis, shadow_analysis, contextual_analysis, reverse_search_results)
            
        except Exception as e:
            print(f"Error generating visual content: {e}")
            # Return None for all visual components on error
            return (None, None, None, None, None, None, None, None)
    
    def _extract_frames_for_visualization(self, video_path: str, max_frames: int = 6) -> list:
        """Extract frames from video for visualization"""
        import cv2
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _generate_ai_visualizations(self, results: Dict[str, Any], frames: list) -> tuple:
        """Generate AI layer visualizations"""
        from PIL import Image, ImageDraw, ImageFont
        
        ai_result = results.get('analysis_results', {}).get('AI_Layer', {})
        
        # Create AI frames display
        ai_frames_display = []
        if frames:
            for i, frame in enumerate(frames[:6]):
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Add frame number and AI prediction
                draw = ImageDraw.Draw(pil_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Add frame info
                frame_text = f"Frame {i+1}"
                if ai_result.get('status') == 'success':
                    prediction = ai_result.get('prediction', 'Unknown')
                    confidence = ai_result.get('confidence', 0.0)
                    frame_text += f"\n{prediction} ({confidence:.1%})"
                
                # Draw text with background
                try:
                    bbox = draw.textbbox((0, 0), frame_text, font=font)
                    if bbox and len(bbox) == 4:
                        # Add padding to the bbox
                        padding = 5
                        x1, y1, x2, y2 = bbox
                        expanded_bbox = (max(0, x1 - padding), max(0, y1 - padding), 
                                       min(pil_image.width, x2 + padding), min(pil_image.height, y2 + padding))
                        if expanded_bbox[2] > expanded_bbox[0] and expanded_bbox[3] > expanded_bbox[1]:
                            draw.rectangle(expanded_bbox, fill=(0, 0, 0))
                except Exception as e:
                    print(f"Warning: Could not draw background for AI frame {i+1}: {e}")
                
                draw.text((10, 10), frame_text, fill=(255, 255, 255), font=font)
                
                ai_frames_display.append(pil_image)
        
        # Create AI heatmap
        ai_heatmap = None
        if frames and ai_result.get('status') == 'success':
            # Create a simple heatmap showing confidence across frames
            confidence = ai_result.get('confidence', 0.0)
            prediction = ai_result.get('prediction', 'Unknown')
            
            # Create a heatmap image
            heatmap_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
            draw = ImageDraw.Draw(heatmap_img)
            
            # Draw confidence bar
            bar_width = int(confidence * 350)
            bar_color = (255, 0, 0) if prediction == 'FAKE' else (0, 255, 0)
            draw.rectangle([25, 100, 25 + bar_width, 150], fill=bar_color)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((25, 50), f"AI Prediction: {prediction}", fill=(0, 0, 0), font=font)
            draw.text((25, 170), f"Confidence: {confidence:.1%}", fill=(0, 0, 0), font=font)
            
            ai_heatmap = heatmap_img
        
        return ai_frames_display, ai_heatmap
    
    def _generate_frequency_visualizations(self, results: Dict[str, Any], frames: list) -> tuple:
        """Generate frequency layer visualizations"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        freq_result = results.get('analysis_results', {}).get('Frequency_Layer', {})
        
        # Create frequency analysis gallery
        frequency_analysis = []
        if frames:
            for i, frame in enumerate(frames[:4]):
                # Convert to grayscale for frequency analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Apply FFT
                fft = np.fft.fft2(gray_frame)
                fft_shift = np.fft.fftshift(fft)
                magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
                
                # Normalize to 0-255 with safety checks
                min_val = magnitude_spectrum.min()
                max_val = magnitude_spectrum.max()
                if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                    magnitude_spectrum = ((magnitude_spectrum - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    # Fallback: create a simple pattern
                    magnitude_spectrum = np.zeros_like(magnitude_spectrum, dtype=np.uint8)
                
                # Convert to PIL Image
                freq_img = Image.fromarray(magnitude_spectrum)
                
                # Add frame info
                draw = ImageDraw.Draw(freq_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                frame_text = f"FFT Frame {i+1}"
                draw.text((10, 10), frame_text, fill=(255, 255, 255), font=font)
                
                frequency_analysis.append(freq_img)
        
        # Create DCT visualization
        dct_visualization = None
        if frames and freq_result.get('status') == 'success':
            # Create DCT block visualization
            frame = frames[0]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply DCT to 8x8 blocks
            dct_img = np.zeros_like(gray_frame)
            for i in range(0, gray_frame.shape[0] - 8, 8):
                for j in range(0, gray_frame.shape[1] - 8, 8):
                    block = gray_frame[i:i+8, j:j+8]
                    dct_block = cv2.dct(block.astype(np.float32))
                    dct_img[i:i+8, j:j+8] = np.abs(dct_block)
            
            # Normalize and convert to PIL
            dct_img = ((dct_img - dct_img.min()) / (dct_img.max() - dct_img.min()) * 255).astype(np.uint8)
            dct_visualization = Image.fromarray(dct_img)
        
        return frequency_analysis, dct_visualization
    
    def _generate_physics_visualizations(self, results: Dict[str, Any], frames: list) -> tuple:
        """Generate physics layer visualizations"""
        from PIL import Image, ImageDraw, ImageFont
        import cv2
        import numpy as np
        
        physics_result = results.get('analysis_results', {}).get('Physics_Layer', {})
        
        # Create physics analysis gallery
        physics_analysis = []
        if frames:
            for i, frame in enumerate(frames[:4]):
                # Apply edge detection for physics analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray_frame, 50, 150)
                
                # Convert to RGB for display
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
                # Convert to PIL Image
                physics_img = Image.fromarray(edges_rgb)
                
                # Add frame info
                draw = ImageDraw.Draw(physics_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                frame_text = f"Physics Frame {i+1}"
                draw.text((10, 10), frame_text, fill=(255, 255, 255), font=font)
                
                physics_analysis.append(physics_img)
        
        # Create shadow analysis visualization
        shadow_analysis = None
        if frames and physics_result.get('status') == 'success':
            # Create shadow detection visualization
            frame = frames[0]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply shadow detection
            shadow_mask = cv2.adaptiveThreshold(
                gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Convert to RGB for display
            shadow_rgb = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            shadow_analysis = Image.fromarray(shadow_rgb)
        
        return physics_analysis, shadow_analysis
    
    def _generate_contextual_visualizations(self, results: Dict[str, Any], frames: list) -> tuple:
        """Generate contextual layer visualizations"""
        from PIL import Image, ImageDraw, ImageFont
        
        contextual_result = results.get('analysis_results', {}).get('Contextual_Layer', {})
        
        # Create contextual analysis gallery
        contextual_analysis = []
        if frames:
            for i, frame in enumerate(frames[:4]):
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Add contextual analysis overlay
                draw = ImageDraw.Draw(pil_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Add frame info and contextual data
                frame_text = f"Context Frame {i+1}"
                if contextual_result.get('status') == 'success':
                    reverse_search = contextual_result.get('reverse_search_analysis', {})
                    similar_images = reverse_search.get('similar_images_found', 0)
                    frame_text += f"\nSimilar: {similar_images}"
                
                # Draw text with background
                try:
                    bbox = draw.textbbox((0, 0), frame_text, font=font)
                    if bbox and len(bbox) == 4:
                        # Add padding to the bbox
                        padding = 5
                        x1, y1, x2, y2 = bbox
                        expanded_bbox = (max(0, x1 - padding), max(0, y1 - padding), 
                                       min(pil_image.width, x2 + padding), min(pil_image.height, y2 + padding))
                        if expanded_bbox[2] > expanded_bbox[0] and expanded_bbox[3] > expanded_bbox[1]:
                            draw.rectangle(expanded_bbox, fill=(0, 0, 0))
                except Exception as e:
                    print(f"Warning: Could not draw background for context frame {i+1}: {e}")
                
                draw.text((10, 10), frame_text, fill=(255, 255, 255), font=font)
                
                contextual_analysis.append(pil_image)
        
        # Create reverse search results visualization
        reverse_search_results = None
        if contextual_result.get('status') == 'success':
            reverse_search = contextual_result.get('reverse_search_analysis', {})
            similar_images = reverse_search.get('similar_images_found', 0)
            exact_matches = reverse_search.get('exact_matches', 0)
            
            # Create reverse search results image
            results_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
            draw = ImageDraw.Draw(results_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Add reverse search results
            draw.text((25, 50), f"Reverse Image Search Results:", fill=(0, 0, 0), font=font)
            draw.text((25, 100), f"Similar Images: {similar_images}", fill=(0, 0, 0), font=font)
            draw.text((25, 130), f"Exact Matches: {exact_matches}", fill=(0, 0, 0), font=font)
            
            # Add confidence indicator
            confidence = reverse_search.get('reverse_search_confidence', 0.0)
            bar_width = int(confidence * 350)
            bar_color = (255, 0, 0) if similar_images > 0 else (0, 255, 0)
            draw.rectangle([25, 180, 25 + bar_width, 200], fill=bar_color)
            draw.text((25, 210), f"Confidence: {confidence:.1%}", fill=(0, 0, 0), font=font)
            
            reverse_search_results = results_img
        
        return contextual_analysis, reverse_search_results
    
    def _generate_llm_results(self, results: Dict[str, Any]) -> tuple:
        """Generate LLM analysis results for the interface"""
        llm_result = results.get('analysis_results', {}).get('LLM_Analysis_Layer', {})
        
        if llm_result.get('status') != 'success':
            # Return error state
            error_msg = llm_result.get('error', 'LLM analysis failed')
            return (
                f"## ❌ LLM Analysis Error\n\n**Error:** {error_msg}\n\n*Please check your HF_TOKEN environment variable.*",
                "*LLM analysis failed. Please check your HF_TOKEN environment variable.*"
            )
        
        # Extract LLM analysis results
        expert_opinion = llm_result.get('expert_opinion', 'No expert opinion available')
        recommendations = llm_result.get('recommendations', 'No recommendations available')
        
        return (expert_opinion, recommendations)
    
    def _clean_for_json(self, obj):
        """Clean data structure for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, str):
            # Remove or replace problematic characters
            cleaned = obj.replace('\u2705', '✓').replace('\u26a0\ufe0f', '⚠').replace('\u2013', '-').replace('\u2014', '--')
            # Remove any other problematic Unicode characters and control characters
            cleaned = ''.join(char for char in cleaned if ord(char) >= 32 and ord(char) < 0x10000)
            # Remove any remaining problematic characters
            cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Ensure the string is not too long
            if len(cleaned) > 10000:
                cleaned = cleaned[:10000] + "... [truncated]"
            return cleaned
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, '__dict__'):
            # Handle custom objects
            return str(obj)
        else:
            return str(obj)
    
    def _generate_risk_level_html(self, results: Dict[str, Any]) -> str:
        """Generate risk level HTML"""
        analysis_results = results.get('analysis_results', {})
        
        # Count red flags and issues
        total_issues = 0
        high_risk_indicators = 0
        
        # AI Layer
        ai_result = analysis_results.get('AI_Layer', {})
        if ai_result.get('status') == 'success':
            if ai_result.get('prediction') == 'FAKE':
                total_issues += 1
                if ai_result.get('confidence', 0) > 0.8:
                    high_risk_indicators += 1
        
        # Frequency Layer
        freq_result = analysis_results.get('Frequency_Layer', {})
        if freq_result.get('status') == 'success':
            freq_summary = freq_result.get('summary', {})
            total_issues += freq_summary.get('total_anomalies_detected', 0)
        
        # Physics Layer
        physics_result = analysis_results.get('Physics_Layer', {})
        if physics_result.get('status') == 'success':
            physics_summary = physics_result.get('summary', {})
            total_issues += physics_summary.get('total_physics_inconsistencies', 0)
        
        # Contextual Layer
        contextual_result = analysis_results.get('Contextual_Layer', {})
        if contextual_result.get('status') == 'success':
            contextual_summary = contextual_result.get('summary', {})
            total_issues += contextual_summary.get('total_contextual_red_flags', 0)
            if contextual_summary.get('overall_risk_level') == 'high':
                high_risk_indicators += 1
        
        # Determine risk level
        if total_issues == 0:
            risk_level = "LOW"
            color = "#4caf50"
            bg_color = "#e8f5e8"
        elif total_issues <= 2 and high_risk_indicators == 0:
            risk_level = "MEDIUM"
            color = "#ff9800"
            bg_color = "#fff3e0"
        else:
            risk_level = "HIGH"
            color = "#f44336"
            bg_color = "#ffebee"
        
        return f"""
        <div style='text-align: center; padding: 15px; background: {bg_color}; border-radius: 8px; border: 2px solid {color};'>
            <h3 style='margin: 0; color: {color};'>Risk Level: {risk_level}</h3>
            <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>{total_issues} issues detected</p>
        </div>
        """
    
    def _generate_layer_status_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate layer status HTML"""
        total_layers = len(analysis_results)
        successful_layers = sum(1 for result in analysis_results.values() if result.get('status') == 'success')
        
        if total_layers == 0:
            return "<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><p>No analysis performed yet</p></div>"
        
        success_rate = (successful_layers / total_layers) * 100
        
        if success_rate == 100:
            color = "#4caf50"
            bg_color = "#e8f5e8"
            status = "All Layers Successful"
        elif success_rate >= 75:
            color = "#ff9800"
            bg_color = "#fff3e0"
            status = "Mostly Successful"
        else:
            color = "#f44336"
            bg_color = "#ffebee"
            status = "Some Issues Detected"
        
        return f"""
        <div style='text-align: center; padding: 15px; background: {bg_color}; border-radius: 8px; border: 2px solid {color};'>
            <h4 style='margin: 0; color: {color};'>{status}</h4>
            <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>{successful_layers}/{total_layers} layers completed</p>
        </div>
        """
    
    def _generate_ai_layer_viz(self, ai_result: Dict[str, Any]) -> str:
        """Generate AI layer visualization"""
        if ai_result.get('status') != 'success':
            return "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>AI Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>"
        
        prediction = ai_result.get('prediction', 'Unknown')
        confidence = ai_result.get('confidence', 0.0)
        
        if prediction == 'FAKE':
            color = "#f44336"
            bg_color = "#ffebee"
            icon = "🚨"
        else:
            color = "#4caf50"
            bg_color = "#e8f5e8"
            icon = "✅"
        
        return f"""
        <div style='padding: 12px; background: {bg_color}; border-radius: 8px; margin: 5px; border-left: 4px solid {color};'>
            <strong>AI Layer:</strong> {icon} <span style='color: {color}; font-weight: bold;'>{prediction}</span><br>
            <small style='color: #666;'>Confidence: {confidence:.1%}</small>
        </div>
        """
    
    def _generate_frequency_layer_viz(self, freq_result: Dict[str, Any]) -> str:
        """Generate frequency layer visualization"""
        if freq_result.get('status') != 'success':
            return "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Frequency Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>"
        
        summary = freq_result.get('summary', {})
        total_anomalies = summary.get('total_anomalies_detected', 0)
        
        if total_anomalies > 0:
            color = "#f44336"
            bg_color = "#ffebee"
            icon = "⚠️"
            status = f"{total_anomalies} anomalies"
        else:
            color = "#4caf50"
            bg_color = "#e8f5e8"
            icon = "✅"
            status = "No anomalies"
        
        return f"""
        <div style='padding: 12px; background: {bg_color}; border-radius: 8px; margin: 5px; border-left: 4px solid {color};'>
            <strong>Frequency Layer:</strong> {icon} <span style='color: {color}; font-weight: bold;'>{status}</span><br>
            <small style='color: #666;'>DCT/FFT analysis completed</small>
        </div>
        """
    
    def _generate_physics_layer_viz(self, physics_result: Dict[str, Any]) -> str:
        """Generate physics layer visualization"""
        if physics_result.get('status') != 'success':
            return "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Physics Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>"
        
        summary = physics_result.get('summary', {})
        total_issues = summary.get('total_physics_inconsistencies', 0)
        
        if total_issues > 0:
            color = "#f44336"
            bg_color = "#ffebee"
            icon = "⚠️"
            status = f"{total_issues} inconsistencies"
        else:
            color = "#4caf50"
            bg_color = "#e8f5e8"
            icon = "✅"
            status = "No issues"
        
        return f"""
        <div style='padding: 12px; background: {bg_color}; border-radius: 8px; margin: 5px; border-left: 4px solid {color};'>
            <strong>Physics Layer:</strong> {icon} <span style='color: {color}; font-weight: bold;'>{status}</span><br>
            <small style='color: #666;'>Shadow/reflection/geometry analysis</small>
        </div>
        """
    
    def _generate_contextual_layer_viz(self, contextual_result: Dict[str, Any]) -> str:
        """Generate contextual layer visualization"""
        if contextual_result.get('status') != 'success':
            return "<div style='padding: 10px; background: #ffebee; border-radius: 5px; margin: 5px;'><strong>Contextual Layer:</strong> <span style='color: #d32f2f;'>Error</span></div>"
        
        summary = contextual_result.get('summary', {})
        total_red_flags = summary.get('total_contextual_red_flags', 0)
        risk_level = summary.get('overall_risk_level', 'unknown')
        
        if risk_level == 'high' or total_red_flags > 2:
            color = "#f44336"
            bg_color = "#ffebee"
            icon = "🚨"
            status = f"{total_red_flags} red flags"
        elif risk_level == 'medium' or total_red_flags > 0:
            color = "#ff9800"
            bg_color = "#fff3e0"
            icon = "⚠️"
            status = f"{total_red_flags} red flags"
        else:
            color = "#4caf50"
            bg_color = "#e8f5e8"
            icon = "✅"
            status = "No red flags"
        
        return f"""
        <div style='padding: 12px; background: {bg_color}; border-radius: 8px; margin: 5px; border-left: 4px solid {color};'>
            <strong>Contextual Layer:</strong> {icon} <span style='color: {color}; font-weight: bold;'>{status}</span><br>
            <small style='color: #666;'>Reverse search & provenance analysis</small>
        </div>
        """
    
    def _format_results_text(self, results: Dict[str, Any]) -> str:
        """Format results as human-readable text"""
        text = "=== DEEPFAKE FORENSICS ANALYSIS REPORT ===\n\n"
        
        # Chain of custody
        coc = results.get('chain_of_custody', {})
        text += f"File: {coc.get('filename', 'Unknown')}\n"
        text += f"Analysis Time: {coc.get('analysis_timestamp', 'Unknown')}\n"
        text += f"File Hash: {coc.get('file_hash', 'Unknown')[:16]}...\n\n"
        
        # Analysis results
        analysis_results = results.get('analysis_results', {})
        
        # AI Layer results
        if 'AI_Layer' in analysis_results:
            ai_result = analysis_results['AI_Layer']
            if ai_result.get('status') == 'success':
                model_name = ai_result.get('model_info', {}).get('model_name', 'GenConViT')
                text += f"🤖 AI ANALYSIS ({model_name}):\n"
                text += f"  Prediction: {ai_result.get('prediction', 'Unknown')}\n"
                text += f"  Confidence: {ai_result.get('confidence', 0):.3f}\n"
                text += f"  Fake Probability: {ai_result.get('fake_probability', 0):.3f}\n"
                text += f"  Real Probability: {ai_result.get('real_probability', 0):.3f}\n"
                text += f"  Frames Analyzed: {ai_result.get('model_info', {}).get('num_frames_analyzed', 0)}\n"
                
                # Add warning if mock mode
                if ai_result.get('warning'):
                    text += f"  ⚠️  {ai_result.get('warning')}\n"
                
                text += "\n"
            else:
                text += f"🤖 AI ANALYSIS: ERROR - {ai_result.get('error', 'Unknown error')}\n\n"
        
        # Metadata Layer results
        if 'Metadata_Layer' in analysis_results:
            meta_result = analysis_results['Metadata_Layer']
            if meta_result.get('status') == 'success':
                text += "📋 METADATA ANALYSIS:\n"
                file_info = meta_result.get('file_info', {})
                text += f"  File Size: {file_info.get('file_size', 0):,} bytes\n"
                text += f"  Created: {file_info.get('creation_time', 'Unknown')}\n"
                text += f"  Modified: {file_info.get('modification_time', 'Unknown')}\n"
                text += "  Status: Basic analysis completed\n\n"
            else:
                text += f"📋 METADATA ANALYSIS: ERROR - {meta_result.get('error', 'Unknown error')}\n\n"
        
        # Frequency Layer results
        if 'Frequency_Layer' in analysis_results:
            freq_result = analysis_results['Frequency_Layer']
            if freq_result.get('status') == 'success':
                text += "🔍 FREQUENCY ANALYSIS:\n"
                text += f"  Frames Analyzed: {freq_result.get('frames_analyzed', 0)}\n"
                text += f"  Confidence: {freq_result.get('confidence', 0):.3f}\n"
                
                # DCT Analysis
                dct_analysis = freq_result.get('dct_analysis', {})
                text += f"  DCT Compression Artifacts: {'DETECTED' if dct_analysis.get('compression_artifacts_detected') else 'None'}\n"
                text += f"  DCT Blocks Analyzed: {dct_analysis.get('dct_blocks_analyzed', 0)}\n"
                
                # FFT Analysis
                fft_analysis = freq_result.get('fft_analysis', {})
                text += f"  FFT Anomalies: {'DETECTED' if fft_analysis.get('frequency_anomalies_detected') else 'None'}\n"
                text += f"  Frequency Consistency: {fft_analysis.get('frequency_consistency', 0):.3f}\n"
                
                # Compression Analysis
                comp_analysis = freq_result.get('compression_analysis', {})
                text += f"  Multiple Compression: {'DETECTED' if comp_analysis.get('multiple_compression_detected') else 'None'}\n"
                text += f"  Avg Quality: {comp_analysis.get('average_compression_quality', 0):.3f}\n"
                
                # Frequency Anomalies
                freq_anomalies = freq_result.get('frequency_anomalies', {})
                text += f"  Manipulation Patterns: {'DETECTED' if freq_anomalies.get('manipulation_detected') else 'None'}\n"
                text += f"  Anomaly Score: {freq_anomalies.get('average_anomaly_score', 0):.3f}\n"
                
                # Summary
                summary = freq_result.get('summary', {})
                text += f"  Total Anomalies: {summary.get('total_anomalies_detected', 0)}\n"
                text += f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}\n\n"
            else:
                text += f"🔍 FREQUENCY ANALYSIS: ERROR - {freq_result.get('error', 'Unknown error')}\n\n"
        
        # Physics Layer results
        if 'Physics_Layer' in analysis_results:
            physics_result = analysis_results['Physics_Layer']
            if physics_result.get('status') == 'success':
                text += "🌍 PHYSICS ANALYSIS:\n"
                text += f"  Frames Analyzed: {physics_result.get('frames_analyzed', 0)}\n"
                text += f"  Confidence: {physics_result.get('confidence', 0):.3f}\n"
                
                # Shadow Analysis
                shadow_analysis = physics_result.get('shadow_analysis', {})
                text += f"  Shadow Inconsistencies: {'DETECTED' if shadow_analysis.get('shadow_inconsistencies') else 'None'}\n"
                text += f"  Shadow Consistency: {shadow_analysis.get('shadow_consistency_score', 0):.3f}\n"
                
                # Reflection Analysis
                reflection_analysis = physics_result.get('reflection_analysis', {})
                text += f"  Reflection Inconsistencies: {'DETECTED' if reflection_analysis.get('reflection_inconsistencies') else 'None'}\n"
                text += f"  Reflection Consistency: {reflection_analysis.get('reflection_consistency_score', 0):.3f}\n"
                
                # Geometry Analysis
                geometry_analysis = physics_result.get('geometry_analysis', {})
                text += f"  Geometry Inconsistencies: {'DETECTED' if geometry_analysis.get('geometry_inconsistencies') else 'None'}\n"
                text += f"  Vanishing Point Consistency: {geometry_analysis.get('vanishing_point_consistency', 0):.3f}\n"
                
                # Object Continuity Analysis
                continuity_analysis = physics_result.get('continuity_analysis', {})
                text += f"  Continuity Violations: {'DETECTED' if continuity_analysis.get('continuity_violations') else 'None'}\n"
                text += f"  Object Consistency: {continuity_analysis.get('object_consistency_score', 0):.3f}\n"
                
                # Summary
                summary = physics_result.get('summary', {})
                text += f"  Total Physics Issues: {summary.get('total_physics_inconsistencies', 0)}\n"
                text += f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}\n\n"
            else:
                text += f"🌍 PHYSICS ANALYSIS: ERROR - {physics_result.get('error', 'Unknown error')}\n\n"
        
        # Contextual Layer results
        if 'Contextual_Layer' in analysis_results:
            contextual_result = analysis_results['Contextual_Layer']
            if contextual_result.get('status') == 'success':
                text += "🔍 CONTEXTUAL ANALYSIS:\n"
                text += f"  Frames Analyzed: {contextual_result.get('frames_analyzed', 0)}\n"
                text += f"  Confidence: {contextual_result.get('confidence', 0):.3f}\n"
                
                # Reverse Search Analysis
                reverse_search = contextual_result.get('reverse_search_analysis', {})
                text += f"  Similar Images Found: {reverse_search.get('similar_images_found', 0)}\n"
                text += f"  Exact Matches: {reverse_search.get('exact_matches', 0)}\n"
                text += f"  Reverse Search Confidence: {reverse_search.get('reverse_search_confidence', 0):.3f}\n"
                
                # Social Network Analysis
                social_network = contextual_result.get('social_network_analysis', {})
                text += f"  Platforms Detected: {', '.join(social_network.get('platforms_detected', []))}\n"
                text += f"  Suspicious Activity: {'DETECTED' if social_network.get('engagement_metrics', {}).get('suspicious_activity') else 'None'}\n"
                text += f"  Total Shares: {social_network.get('engagement_metrics', {}).get('total_shares', 0)}\n"
                
                # Uploader Analysis
                uploader = contextual_result.get('uploader_analysis', {})
                text += f"  Uploader Credibility: {uploader.get('credibility_score', 0):.3f}\n"
                text += f"  Account Age: {uploader.get('account_age_days', 0)} days\n"
                text += f"  Previous Deepfakes: {uploader.get('previous_deepfake_uploads', 0)}\n"
                
                # Metadata Correlation
                metadata_corr = contextual_result.get('metadata_correlation', {})
                text += f"  Data Consistency: {'GOOD' if metadata_corr.get('data_consistency') else 'ISSUES'}\n"
                text += f"  Camera Model: {metadata_corr.get('camera_model_detected', 'Unknown')}\n"
                text += f"  Correlation Confidence: {metadata_corr.get('correlation_confidence', 0):.3f}\n"
                
                # Summary
                summary = contextual_result.get('summary', {})
                text += f"  Total Red Flags: {summary.get('total_contextual_red_flags', 0)}\n"
                text += f"  Risk Level: {summary.get('overall_risk_level', 'unknown').upper()}\n"
                text += f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}\n\n"
            else:
                text += f"🔍 CONTEXTUAL ANALYSIS: ERROR - {contextual_result.get('error', 'Unknown error')}\n\n"
        
        # Summary
        summary = results.get('summary', {})
        text += "📊 SUMMARY:\n"
        text += f"  Total Layers: {summary.get('total_layers', 0)}\n"
        text += f"  Successful: {summary.get('successful_layers', 0)}\n"
        text += f"  Failed: {summary.get('failed_layers', 0)}\n"
        text += f"  Overall Confidence: {summary.get('overall_confidence', 0):.3f}\n\n"
        
        return text
    
    def _format_recommendations(self, results: Dict[str, Any]) -> str:
        """Format recommendations"""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            return "No specific recommendations at this time."
        
        text = "RECOMMENDATIONS:\n"
        for i, rec in enumerate(recommendations, 1):
            text += f"{i}. {rec}\n"
        
        return text
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        
        with gr.Blocks(
            title="Deepfake Forensics Framework",
            theme=gr.themes.Soft(),
            css="""
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .results-container {
                max-height: 600px;
                overflow-y: auto;
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Deepfake Forensics Framework</h1>
                <p>A comprehensive multi-layer analysis system for deepfake detection and forensic reporting</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Configure")
                    
                    video_input = gr.File(
                        label="Upload Video File",
                        file_types=["video"],
                        type="filepath"
                    )
                    
                    num_frames = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=15,
                        step=1,
                        label="Number of Frames to Analyze"
                    )
                    
                    include_metadata = gr.Checkbox(
                        value=True,
                        label="Include Metadata Analysis"
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Video",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    
                    with gr.Tabs():
                        with gr.Tab("📋 Detailed Report"):
                            results_text = gr.Textbox(
                                label="Analysis Report",
                                lines=20,
                                max_lines=30,
                                container=True,
                                elem_classes=["results-container"]
                            )
                            
                            with gr.Row():
                                with gr.Column():
                                    results_json = gr.JSON(
                                        label="Raw Results (JSON)"
                                    )
                                
                                with gr.Column():
                                    recommendations = gr.Textbox(
                                        label="Recommendations",
                                        lines=10
                                    )
                        
                        with gr.Tab("🎨 Visual Analysis"):
                            # Visual Summary Section
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### 🎯 Overall Summary")
                                    overall_confidence = gr.HTML(
                                        value="<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><h4>Overall Confidence: <span style='color: #666;'>0%</span></h4></div>"
                                    )
                                    risk_level = gr.HTML(
                                        value="<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><h3>Risk Level: <span style='color: #666;'>Unknown</span></h3></div>"
                                    )
                                    
                                with gr.Column(scale=1):
                                    gr.Markdown("#### 📊 Layer Status")
                                    layer_status = gr.HTML(
                                        value="<div style='text-align: center; padding: 10px; background: #f0f0f0; border-radius: 5px;'><p>No analysis performed yet</p></div>"
                                    )
                            
                            # Layer Results Visualization
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### 🔍 Layer Analysis")
                                    ai_result_viz = gr.HTML(
                                        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>AI Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>"
                                    )
                                    frequency_result_viz = gr.HTML(
                                        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Frequency Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>"
                                    )
                                    physics_result_viz = gr.HTML(
                                        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Physics Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>"
                                    )
                                    contextual_result_viz = gr.HTML(
                                        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px;'><strong>Contextual Layer:</strong> <span style='color: #666;'>Not analyzed</span></div>"
                                    )
                            
                            # Visual Analysis Section
                            gr.Markdown("#### 🖼️ Visual Analysis")
                            
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("##### 🤖 AI Layer - Frame Analysis")
                                    ai_frames_display = gr.Gallery(
                                        label="Analyzed Frames",
                                        show_label=True,
                                        elem_id="ai_frames",
                                        columns=3,
                                        rows=2,
                                        height="auto"
                                    )
                                    ai_heatmap = gr.Image(
                                        label="AI Detection Heatmap",
                                        type="pil"
                                    )
                                
                                with gr.Column():
                                    gr.Markdown("##### 🔍 Frequency Layer - Anomaly Detection")
                                    frequency_analysis = gr.Gallery(
                                        label="Frequency Analysis Results",
                                        show_label=True,
                                        elem_id="frequency_analysis",
                                        columns=2,
                                        rows=2,
                                        height="auto"
                                    )
                                    dct_visualization = gr.Image(
                                        label="DCT Block Analysis",
                                        type="pil"
                                    )
                            
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("##### 🌍 Physics Layer - Physical Inconsistencies")
                                    physics_analysis = gr.Gallery(
                                        label="Physics Analysis Results",
                                        show_label=True,
                                        elem_id="physics_analysis",
                                        columns=2,
                                        rows=2,
                                        height="auto"
                                    )
                                    shadow_analysis = gr.Image(
                                        label="Shadow Consistency Analysis",
                                        type="pil"
                                    )
                                
                                with gr.Column():
                                    gr.Markdown("##### 🔍 Contextual Layer - Provenance Analysis")
                                    contextual_analysis = gr.Gallery(
                                        label="Contextual Analysis Results",
                                        show_label=True,
                                        elem_id="contextual_analysis",
                                        columns=2,
                                        rows=2,
                                        height="auto"
                                    )
                                    reverse_search_results = gr.Image(
                                        label="Reverse Image Search Results",
                                        type="pil"
                                    )
                        
                        with gr.Tab("🤖 Expert Analysis"):
                            gr.Markdown("#### 🧠 AI Expert Analysis")
                            gr.Markdown("""
                            **gpt-oss-20b - Digital Forensics Expert**
                            
                            This analysis is performed by an AI system.
                            The AI system provides professional assessment based on all available evidence.
                            """)
                            
                            llm_expert_opinion = gr.Markdown(
                                label="Expert Opinion",
                                value="*No analysis performed yet*",
                                container=True,
                                elem_classes=["expert-analysis"]
                            )
                            
                            llm_recommendations = gr.Markdown(
                                label="Expert Recommendations",
                                value="*No recommendations available*",
                                container=True
                            )
            
            # Event handlers
            analyze_btn.click(
                fn=self.analyze_video_with_visuals,
                inputs=[video_input, num_frames, include_metadata],
                outputs=[overall_confidence, risk_level, layer_status, ai_result_viz, 
                        frequency_result_viz, physics_result_viz, contextual_result_viz,
                        ai_frames_display, ai_heatmap, frequency_analysis, dct_visualization,
                        physics_analysis, shadow_analysis, contextual_analysis, reverse_search_results,
                        llm_expert_opinion, llm_recommendations,
                        results_text, results_json, recommendations]
            )
            
            # Example section
            gr.Markdown("### 📋 Framework Layers")
            gr.Markdown("""
            **Current Implementation:**
            - 🤖 **AI Layer**: GenConViT model for deepfake detection
            - 📋 **Metadata Layer**: Basic file information analysis
            - 🔍 **Frequency Layer**: DCT/FFT analysis for compression artifacts and frequency anomalies
            - 🌍 **Physics Layer**: Shadow, reflection, geometry, and object continuity analysis
            - 🔍 **Contextual Layer**: Reverse-image search, social network analysis, uploader history, and metadata correlation
            
            **Planned Enhancements:**
            - 🎯 **Localization**: Spatial anomaly detection and heatmaps
            - 📊 **Explainability**: Attention maps and evidence visualization
            """)
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e0e0e0;">
                <p><strong>Deepfake Forensics Framework v1.0</strong></p>
                <p>Built with GenConViT and multi-layer forensic analysis</p>
            </div>
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": False,
            "show_error": True
        }
        
        # Update with user parameters
        launch_params.update(kwargs)
        
        print("🚀 Launching Deepfake Forensics Framework...")
        print(f"📱 Interface will be available at: http://localhost:{launch_params['server_port']}")
        
        interface.launch(**launch_params)

def main():
    """Main function to run the Gradio interface"""
    app = DFFGradioInterface()
    app.launch()

if __name__ == "__main__":
    main()
