"""
Frequency Analysis Layer - Analyzes DCT/FFT patterns for compression artifacts
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
# matplotlib import removed - not needed for current implementation

from dff_framework.core.base_layer import BaseForensicLayer

class FrequencyLayer(BaseForensicLayer):
    """
    Layer 2: Frequency & Compression Analysis
    Analyzes DCT/FFT patterns, compression artifacts, and frequency domain anomalies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Frequency_Layer", config)
        self.dct_block_size = self.config.get('dct_block_size', 8)
        self.fft_threshold = self.config.get('fft_threshold', 0.1)
        self.compression_quality_range = self.config.get('compression_quality_range', (30, 100))
    
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video frequency patterns and compression artifacts
        
        Args:
            video_path: Path to the video file
            options: Analysis options
            
        Returns:
            Frequency analysis results
        """
        if not self.validate_input(video_path):
            return {
                "status": "error",
                "error": "Invalid video file"
            }
        
        try:
            # Extract frames from video
            print(f"Frequency Layer: Extracting frames from {video_path}")
            frames = self._extract_frames(video_path, max_frames=10)
            
            if len(frames) == 0:
                return {
                    "status": "error",
                    "error": "No frames extracted from video"
                }
            
            print(f"Frequency Layer: Successfully extracted {len(frames)} frames")
            
            # Perform frequency analysis
            print("Frequency Layer: Starting DCT analysis...")
            try:
                dct_analysis = self._analyze_dct_patterns(frames)
                print("Frequency Layer: DCT analysis completed")
            except Exception as e:
                print(f"Frequency Layer: DCT analysis failed: {e}")
                dct_analysis = {"error": str(e), "compression_artifacts_detected": False, "dct_blocks_analyzed": 0}
            
            print("Frequency Layer: Starting FFT analysis...")
            try:
                fft_analysis = self._analyze_fft_patterns(frames)
                print("Frequency Layer: FFT analysis completed")
            except Exception as e:
                print(f"Frequency Layer: FFT analysis failed: {e}")
                fft_analysis = {"error": str(e), "frequency_anomalies_detected": False, "frequency_consistency": 0.0}
            
            print("Frequency Layer: Starting compression analysis...")
            try:
                compression_analysis = self._analyze_compression_artifacts(frames)
                print("Frequency Layer: Compression analysis completed")
            except Exception as e:
                print(f"Frequency Layer: Compression analysis failed: {e}")
                compression_analysis = {"error": str(e), "multiple_compression_detected": False, "average_compression_quality": 0.0}
            
            print("Frequency Layer: Starting anomaly detection...")
            try:
                frequency_anomalies = self._detect_frequency_anomalies(frames)
                print("Frequency Layer: Anomaly detection completed")
            except Exception as e:
                print(f"Frequency Layer: Anomaly detection failed: {e}")
                frequency_anomalies = {"error": str(e), "manipulation_detected": False, "average_anomaly_score": 0.0}
            
            # Calculate overall confidence
            confidence = self._calculate_frequency_confidence(
                dct_analysis, fft_analysis, compression_analysis, frequency_anomalies
            )
            
            # Generate results
            results = {
                "status": "success",
                "analysis_type": "Frequency Domain Analysis",
                "frames_analyzed": len(frames),
                "dct_analysis": dct_analysis,
                "fft_analysis": fft_analysis,
                "compression_analysis": compression_analysis,
                "frequency_anomalies": frequency_anomalies,
                "confidence": confidence,
                "summary": self._generate_frequency_summary(
                    dct_analysis, fft_analysis, compression_analysis, frequency_anomalies
                ),
                "recommendations": self._generate_frequency_recommendations(
                    dct_analysis, fft_analysis, compression_analysis, frequency_anomalies
                )
            }
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Frequency analysis failed: {str(e)}"
            }
    
    def _extract_frames(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale for frequency analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
        
        cap.release()
        return frames
    
    def _analyze_dct_patterns(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze DCT patterns for compression artifacts"""
        dct_coefficients = []
        dct_energy_distribution = []
        
        for frame in frames:
            # Resize frame to ensure it's divisible by block size
            h, w = frame.shape
            new_h = (h // self.dct_block_size) * self.dct_block_size
            new_w = (w // self.dct_block_size) * self.dct_block_size
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Compute DCT for each block
            frame_dct_coeffs = []
            frame_energy = []
            
            for i in range(0, new_h, self.dct_block_size):
                for j in range(0, new_w, self.dct_block_size):
                    block = frame_resized[i:i+self.dct_block_size, j:j+self.dct_block_size]
                    dct_block = cv2.dct(block.astype(np.float32))
                    
                    # Store DCT coefficients
                    frame_dct_coeffs.append(dct_block)
                    
                    # Calculate energy distribution
                    energy = np.sum(dct_block**2)
                    frame_energy.append(energy)
            
            dct_coefficients.append(frame_dct_coeffs)
            dct_energy_distribution.append(frame_energy)
        
        # Analyze DCT patterns
        avg_energy = np.mean([np.mean(energy) for energy in dct_energy_distribution])
        energy_variance = np.var([np.mean(energy) for energy in dct_energy_distribution])
        
        # Check for compression artifacts (high energy in high frequencies)
        high_freq_energy_ratio = self._calculate_high_frequency_energy_ratio(dct_coefficients)
        
        return {
            "average_energy": float(avg_energy),
            "energy_variance": float(energy_variance),
            "high_frequency_energy_ratio": float(high_freq_energy_ratio),
            "compression_artifacts_detected": bool(high_freq_energy_ratio > 0.3),
            "dct_blocks_analyzed": int(sum(len(frame_coeffs) for frame_coeffs in dct_coefficients))
        }
    
    def _analyze_fft_patterns(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze FFT patterns for frequency domain anomalies"""
        fft_magnitudes = []
        fft_phases = []
        
        for frame in frames:
            # Compute 2D FFT
            fft = np.fft.fft2(frame)
            fft_shifted = np.fft.fftshift(fft)
            
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            fft_magnitudes.append(magnitude)
            fft_phases.append(phase)
        
        # Analyze FFT patterns
        avg_magnitude = np.mean([np.mean(mag) for mag in fft_magnitudes])
        magnitude_variance = np.var([np.mean(mag) for mag in fft_magnitudes])
        
        # Check for frequency domain anomalies
        frequency_consistency = self._calculate_frequency_consistency(fft_magnitudes)
        spectral_centroid = self._calculate_spectral_centroid(fft_magnitudes)
        
        return {
            "average_magnitude": float(avg_magnitude),
            "magnitude_variance": float(magnitude_variance),
            "frequency_consistency": float(frequency_consistency),
            "spectral_centroid": float(spectral_centroid),
            "frequency_anomalies_detected": bool(frequency_consistency < 0.7)
        }
    
    def _analyze_compression_artifacts(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze compression artifacts and quality indicators"""
        compression_indicators = []
        quality_metrics = []
        
        for frame in frames:
            # Calculate compression indicators
            laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
            sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_var = np.var(gradient_magnitude)
            
            compression_indicators.append({
                "laplacian_variance": float(laplacian_var),
                "gradient_variance": float(gradient_var)
            })
            
            # Estimate compression quality
            quality_score = self._estimate_compression_quality(frame)
            quality_metrics.append(quality_score)
        
        avg_quality = np.mean(quality_metrics)
        quality_variance = np.var(quality_metrics)
        
        # Detect compression inconsistencies
        compression_inconsistency = self._detect_compression_inconsistency(compression_indicators)
        
        return {
            "average_compression_quality": float(avg_quality),
            "quality_variance": float(quality_variance),
            "compression_inconsistency": float(compression_inconsistency),
            "multiple_compression_detected": bool(compression_inconsistency > 0.5),
            "frames_with_artifacts": int(sum(1 for q in quality_metrics if q < 0.6))
        }
    
    def _detect_frequency_anomalies(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Detect frequency domain anomalies that might indicate manipulation"""
        anomalies = []
        
        for i, frame in enumerate(frames):
            # Check for periodic patterns (common in deepfakes)
            periodic_score = self._detect_periodic_patterns(frame)
            
            # Check for frequency domain inconsistencies
            freq_inconsistency = self._check_frequency_inconsistency(frame)
            
            # Check for edge artifacts
            edge_artifacts = self._detect_edge_artifacts(frame)
            
            anomalies.append({
                "frame_index": i,
                "periodic_pattern_score": float(periodic_score),
                "frequency_inconsistency": float(freq_inconsistency),
                "edge_artifacts": float(edge_artifacts),
                "anomaly_score": float((periodic_score + freq_inconsistency + edge_artifacts) / 3)
            })
        
        avg_anomaly_score = np.mean([a["anomaly_score"] for a in anomalies])
        max_anomaly_score = max([a["anomaly_score"] for a in anomalies])
        
        return {
            "average_anomaly_score": float(avg_anomaly_score),
            "maximum_anomaly_score": float(max_anomaly_score),
            "frames_with_anomalies": int(sum(1 for a in anomalies if a["anomaly_score"] > 0.5)),
            "anomaly_details": anomalies,
            "manipulation_detected": bool(avg_anomaly_score > 0.4)
        }
    
    def _calculate_high_frequency_energy_ratio(self, dct_coefficients: List[List[np.ndarray]]) -> float:
        """Calculate ratio of high frequency energy to total energy"""
        total_high_freq_energy = 0
        total_energy = 0
        
        for frame_coeffs in dct_coefficients:
            for dct_block in frame_coeffs:
                # High frequency coefficients are in the bottom-right corner
                h, w = dct_block.shape
                high_freq_region = dct_block[h//2:, w//2:]
                total_high_freq_energy += np.sum(high_freq_region**2)
                total_energy += np.sum(dct_block**2)
        
        return total_high_freq_energy / total_energy if total_energy > 0 else 0
    
    def _calculate_frequency_consistency(self, fft_magnitudes: List[np.ndarray]) -> float:
        """Calculate consistency of frequency patterns across frames"""
        if len(fft_magnitudes) < 2:
            return 1.0
        
        # Calculate correlation between consecutive frames
        correlations = []
        for i in range(len(fft_magnitudes) - 1):
            corr = np.corrcoef(
                fft_magnitudes[i].flatten(),
                fft_magnitudes[i + 1].flatten()
            )[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        return np.mean(correlations)
    
    def _calculate_spectral_centroid(self, fft_magnitudes: List[np.ndarray]) -> float:
        """Calculate spectral centroid of frequency distribution"""
        centroids = []
        for magnitude in fft_magnitudes:
            h, w = magnitude.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            # Calculate distance from center
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Weighted average of distances
            centroid = np.sum(distances * magnitude) / np.sum(magnitude)
            centroids.append(centroid)
        
        return np.mean(centroids)
    
    def _estimate_compression_quality(self, frame: np.ndarray) -> float:
        """Estimate compression quality based on image characteristics"""
        # Use Laplacian variance as a proxy for image quality
        laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher values indicate better quality)
        quality = min(1.0, laplacian_var / 1000.0)
        return quality
    
    def _detect_compression_inconsistency(self, compression_indicators: List[Dict]) -> float:
        """Detect inconsistencies in compression across frames"""
        if len(compression_indicators) < 2:
            return 0.0
        
        laplacian_vars = [ci["laplacian_variance"] for ci in compression_indicators]
        gradient_vars = [ci["gradient_variance"] for ci in compression_indicators]
        
        # Calculate coefficient of variation (standard deviation / mean)
        laplacian_cv = np.std(laplacian_vars) / np.mean(laplacian_vars) if np.mean(laplacian_vars) > 0 else 0
        gradient_cv = np.std(gradient_vars) / np.mean(gradient_vars) if np.mean(gradient_vars) > 0 else 0
        
        return (laplacian_cv + gradient_cv) / 2
    
    def _detect_periodic_patterns(self, frame: np.ndarray) -> float:
        """Detect periodic patterns that might indicate manipulation"""
        # Use autocorrelation to detect periodic patterns
        fft = np.fft.fft2(frame)
        autocorr = np.fft.ifft2(fft * np.conj(fft))
        autocorr = np.real(autocorr)
        
        # Normalize autocorrelation
        autocorr = autocorr / np.max(autocorr)
        
        # Look for strong periodic components
        h, w = autocorr.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for periodic patterns in different directions
        horizontal_pattern = np.max(autocorr[center_h, :])
        vertical_pattern = np.max(autocorr[:, center_w])
        
        return max(horizontal_pattern, vertical_pattern)
    
    def _check_frequency_inconsistency(self, frame: np.ndarray) -> float:
        """Check for frequency domain inconsistencies"""
        # Divide frame into regions and compare frequency characteristics
        h, w = frame.shape
        regions = [
            frame[:h//2, :w//2],  # Top-left
            frame[:h//2, w//2:],  # Top-right
            frame[h//2:, :w//2],  # Bottom-left
            frame[h//2:, w//2:]   # Bottom-right
        ]
        
        fft_magnitudes = []
        for region in regions:
            fft = np.fft.fft2(region)
            magnitude = np.abs(fft)
            fft_magnitudes.append(magnitude)
        
        # Calculate variance in frequency characteristics across regions
        mean_magnitudes = [np.mean(mag) for mag in fft_magnitudes]
        inconsistency = np.std(mean_magnitudes) / np.mean(mean_magnitudes) if np.mean(mean_magnitudes) > 0 else 0
        
        return min(1.0, inconsistency)
    
    def _detect_edge_artifacts(self, frame: np.ndarray) -> float:
        """Detect edge artifacts that might indicate manipulation"""
        # Use Canny edge detection
        edges = cv2.Canny(frame, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Look for unusual edge patterns
        # High edge density might indicate compression artifacts
        return min(1.0, edge_density * 10)
    
    def _calculate_frequency_confidence(self, dct_analysis: Dict, fft_analysis: Dict, 
                                      compression_analysis: Dict, frequency_anomalies: Dict) -> float:
        """Calculate overall confidence in frequency analysis"""
        confidence_factors = []
        
        # DCT analysis confidence
        if dct_analysis["compression_artifacts_detected"]:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # FFT analysis confidence
        if fft_analysis["frequency_anomalies_detected"]:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Compression analysis confidence
        if compression_analysis["multiple_compression_detected"]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        # Frequency anomalies confidence
        if frequency_anomalies["manipulation_detected"]:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    def _generate_frequency_summary(self, dct_analysis: Dict, fft_analysis: Dict,
                                  compression_analysis: Dict, frequency_anomalies: Dict) -> Dict[str, Any]:
        """Generate summary of frequency analysis"""
        total_anomalies = 0
        if dct_analysis["compression_artifacts_detected"]:
            total_anomalies += 1
        if fft_analysis["frequency_anomalies_detected"]:
            total_anomalies += 1
        if compression_analysis["multiple_compression_detected"]:
            total_anomalies += 1
        if frequency_anomalies["manipulation_detected"]:
            total_anomalies += 1
        
        return {
            "total_anomalies_detected": int(total_anomalies),
            "analysis_quality": "high" if total_anomalies > 0 else "normal",
            "primary_concerns": self._identify_primary_concerns(
                dct_analysis, fft_analysis, compression_analysis, frequency_anomalies
            )
        }
    
    def _identify_primary_concerns(self, dct_analysis: Dict, fft_analysis: Dict,
                                 compression_analysis: Dict, frequency_anomalies: Dict) -> List[str]:
        """Identify primary concerns from frequency analysis"""
        concerns = []
        
        if dct_analysis["compression_artifacts_detected"]:
            concerns.append("DCT compression artifacts detected")
        
        if fft_analysis["frequency_anomalies_detected"]:
            concerns.append("Frequency domain anomalies detected")
        
        if compression_analysis["multiple_compression_detected"]:
            concerns.append("Multiple compression cycles detected")
        
        if frequency_anomalies["manipulation_detected"]:
            concerns.append("Frequency manipulation patterns detected")
        
        if not concerns:
            concerns.append("No significant frequency anomalies detected")
        
        return concerns
    
    def _generate_frequency_recommendations(self, dct_analysis: Dict, fft_analysis: Dict,
                                          compression_analysis: Dict, frequency_anomalies: Dict) -> List[str]:
        """Generate recommendations based on frequency analysis"""
        recommendations = []
        
        if dct_analysis["compression_artifacts_detected"]:
            recommendations.append("Review DCT compression artifacts - may indicate recompression")
        
        if fft_analysis["frequency_anomalies_detected"]:
            recommendations.append("Investigate frequency domain inconsistencies")
        
        if compression_analysis["multiple_compression_detected"]:
            recommendations.append("Multiple compression cycles suggest potential manipulation")
        
        if frequency_anomalies["manipulation_detected"]:
            recommendations.append("Frequency manipulation patterns require further investigation")
        
        if not recommendations:
            recommendations.append("Frequency analysis shows no significant anomalies")
        
        return recommendations
