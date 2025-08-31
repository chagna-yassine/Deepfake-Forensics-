"""
Contextual & provenance analysis layer for deepfake detection.

This layer analyzes contextual and provenance information including:
- Reverse-image search capabilities
- Social network propagation analysis
- Uploader history analysis
- Metadata correlation analysis
"""

import cv2
import numpy as np
import hashlib
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import time
import random

from ..core.base_layer import BaseForensicLayer


class ContextualLayer(BaseForensicLayer):
    """
    Contextual & provenance analysis layer
    
    Analyzes contextual information and provenance data:
    - Reverse-image search for content verification
    - Social network propagation analysis
    - Uploader history and credibility assessment
    - Metadata correlation with known sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Contextual Layer
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        super().__init__("Contextual_Layer")
        self.config = config or {}
        
        # Analysis parameters
        self.reverse_search_threshold = self.config.get('reverse_search_threshold', 0.8)
        self.propagation_threshold = self.config.get('propagation_threshold', 0.6)
        self.uploader_credibility_threshold = self.config.get('uploader_credibility_threshold', 0.5)
        self.metadata_correlation_threshold = self.config.get('metadata_correlation_threshold', 0.7)
        
        # Mock data for demonstration (in real implementation, these would be API calls)
        self.mock_reverse_search_results = {
            "similar_images_found": 0,
            "exact_matches": 0,
            "similarity_scores": [],
            "source_urls": [],
            "first_seen_date": None,
            "reverse_search_confidence": 0.95
        }
        
        self.mock_social_network_data = {
            "platforms_detected": ["YouTube", "Twitter", "Reddit"],
            "sharing_patterns": {
                "viral_spread": False,
                "bot_activity": False,
                "coordinated_sharing": False,
                "organic_sharing": True
            },
            "engagement_metrics": {
                "total_shares": 150,
                "total_views": 25000,
                "engagement_rate": 0.006,
                "suspicious_activity": False
            },
            "propagation_timeline": [
                {"platform": "YouTube", "timestamp": "2025-08-30T10:00:00Z", "views": 1000},
                {"platform": "Twitter", "timestamp": "2025-08-30T11:30:00Z", "shares": 50},
                {"platform": "Reddit", "timestamp": "2025-08-30T14:15:00Z", "upvotes": 25}
            ]
        }
        
        self.mock_uploader_history = {
            "account_age_days": 365,
            "total_uploads": 45,
            "previous_deepfake_uploads": 2,
            "account_verification_status": "unverified",
            "suspicious_activity_flags": 1,
            "upload_frequency": "regular",
            "content_categories": ["entertainment", "news", "politics"],
            "credibility_score": 0.65
        }
        
        self.mock_metadata_correlation = {
            "exif_data_consistency": True,
            "camera_model_detected": "iPhone 13 Pro",
            "location_data_present": False,
            "timestamp_consistency": True,
            "compression_history": "single_pass",
            "editing_software_detected": None,
            "correlation_confidence": 0.78
        }
        
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video for contextual and provenance information
        
        Args:
            video_path: Path to the video file
            options: Analysis options
            
        Returns:
            Dictionary containing contextual analysis results
        """
        if not Path(video_path).exists():
            return {
                "status": "error",
                "error": "Invalid video file"
            }
        
        try:
            # Extract frames from video for analysis
            print(f"Contextual Layer: Extracting frames from {video_path}")
            frames = self._extract_frames(video_path, max_frames=5)
            
            if len(frames) == 0:
                return {
                    "status": "error",
                    "error": "No frames extracted from video"
                }
            
            print(f"Contextual Layer: Successfully extracted {len(frames)} frames")
            
            # Perform contextual analysis
            print("Contextual Layer: Starting reverse-image search...")
            try:
                reverse_search_analysis = self._perform_reverse_image_search(frames)
                print("Contextual Layer: Reverse-image search completed")
            except Exception as e:
                print(f"Contextual Layer: Reverse-image search failed: {e}")
                reverse_search_analysis = {"error": str(e), "similar_images_found": 0, "reverse_search_confidence": 0.0}
            
            print("Contextual Layer: Starting social network analysis...")
            try:
                social_network_analysis = self._analyze_social_network_propagation(video_path)
                print("Contextual Layer: Social network analysis completed")
            except Exception as e:
                print(f"Contextual Layer: Social network analysis failed: {e}")
                social_network_analysis = {"error": str(e), "viral_spread": False, "propagation_confidence": 0.0}
            
            print("Contextual Layer: Starting uploader history analysis...")
            try:
                uploader_analysis = self._analyze_uploader_history(video_path)
                print("Contextual Layer: Uploader history analysis completed")
            except Exception as e:
                print(f"Contextual Layer: Uploader history analysis failed: {e}")
                uploader_analysis = {"error": str(e), "credibility_score": 0.0, "suspicious_activity": False}
            
            print("Contextual Layer: Starting metadata correlation...")
            try:
                metadata_correlation = self._analyze_metadata_correlation(video_path)
                print("Contextual Layer: Metadata correlation completed")
            except Exception as e:
                print(f"Contextual Layer: Metadata correlation failed: {e}")
                metadata_correlation = {"error": str(e), "correlation_confidence": 0.0, "data_consistency": False}
            
            # Calculate overall confidence
            confidence = self._calculate_contextual_confidence(
                reverse_search_analysis, social_network_analysis, uploader_analysis, metadata_correlation
            )
            
            # Generate summary
            summary = self._generate_contextual_summary(
                reverse_search_analysis, social_network_analysis, uploader_analysis, metadata_correlation
            )
            
            return {
                "status": "success",
                "layer_name": "Contextual_Layer",
                "frames_analyzed": len(frames),
                "confidence": float(confidence),
                "reverse_search_analysis": reverse_search_analysis,
                "social_network_analysis": social_network_analysis,
                "uploader_analysis": uploader_analysis,
                "metadata_correlation": metadata_correlation,
                "summary": summary,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "layer_name": "Contextual_Layer"
            }
    
    def _extract_frames(self, video_path: str, max_frames: int = 5) -> List[np.ndarray]:
        """Extract frames from video for reverse-image search"""
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
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _perform_reverse_image_search(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Perform reverse-image search on video frames"""
        # In a real implementation, this would use APIs like Google Images, TinEye, etc.
        # For demonstration, we'll simulate the analysis
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Generate frame hashes for uniqueness detection
        frame_hashes = []
        for frame in frames:
            # Convert frame to grayscale and resize for consistent hashing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            frame_hash = hashlib.md5(resized.tobytes()).hexdigest()
            frame_hashes.append(frame_hash)
        
        # Simulate reverse search results
        similar_images_found = random.randint(0, 3)
        exact_matches = random.randint(0, 1) if similar_images_found > 0 else 0
        
        # Calculate similarity scores
        similarity_scores = []
        if similar_images_found > 0:
            for _ in range(similar_images_found):
                similarity_scores.append(random.uniform(0.6, 0.95))
        
        # Determine confidence based on results
        if exact_matches > 0:
            confidence = 0.95
        elif similar_images_found > 0:
            confidence = 0.8
        else:
            confidence = 0.9  # High confidence in uniqueness
        
        return {
            "similar_images_found": int(similar_images_found),
            "exact_matches": int(exact_matches),
            "similarity_scores": [float(score) for score in similarity_scores],
            "source_urls": [f"https://example.com/source_{i+1}.jpg" for i in range(similar_images_found)],
            "first_seen_date": "2025-08-30T10:00:00Z" if similar_images_found > 0 else None,
            "reverse_search_confidence": float(confidence),
            "frame_hashes": frame_hashes[:3],  # Store first 3 hashes
            "uniqueness_score": float(1.0 - (similar_images_found * 0.3))
        }
    
    def _analyze_social_network_propagation(self, video_path: str) -> Dict[str, Any]:
        """Analyze social network propagation patterns"""
        # In a real implementation, this would analyze social media APIs
        # For demonstration, we'll simulate the analysis
        
        # Simulate processing time
        time.sleep(0.3)
        
        # Generate file hash for tracking
        file_hash = hashlib.md5(Path(video_path).read_bytes()).hexdigest()
        
        # Simulate social network analysis
        platforms_detected = random.sample(["YouTube", "Twitter", "Reddit", "TikTok", "Instagram"], 
                                         random.randint(1, 3))
        
        # Simulate sharing patterns
        viral_spread = random.choice([True, False])
        bot_activity = random.choice([True, False]) if viral_spread else False
        coordinated_sharing = random.choice([True, False]) if bot_activity else False
        
        # Simulate engagement metrics
        total_shares = random.randint(10, 1000)
        total_views = random.randint(1000, 100000)
        engagement_rate = total_shares / total_views if total_views > 0 else 0
        
        # Determine if activity is suspicious
        suspicious_activity = (bot_activity or coordinated_sharing or 
                             engagement_rate > 0.1 or total_shares > 500)
        
        # Generate propagation timeline
        propagation_timeline = []
        base_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        
        for i, platform in enumerate(platforms_detected):
            timestamp = base_time + timedelta(hours=i*2)
            if platform == "YouTube":
                propagation_timeline.append({
                    "platform": platform,
                    "timestamp": timestamp.isoformat() + "Z",
                    "views": random.randint(100, 10000)
                })
            elif platform == "Twitter":
                propagation_timeline.append({
                    "platform": platform,
                    "timestamp": timestamp.isoformat() + "Z",
                    "shares": random.randint(5, 200)
                })
            else:
                propagation_timeline.append({
                    "platform": platform,
                    "timestamp": timestamp.isoformat() + "Z",
                    "upvotes": random.randint(10, 100)
                })
        
        return {
            "file_hash": file_hash,
            "platforms_detected": platforms_detected,
            "sharing_patterns": {
                "viral_spread": bool(viral_spread),
                "bot_activity": bool(bot_activity),
                "coordinated_sharing": bool(coordinated_sharing),
                "organic_sharing": bool(not bot_activity)
            },
            "engagement_metrics": {
                "total_shares": int(total_shares),
                "total_views": int(total_views),
                "engagement_rate": float(engagement_rate),
                "suspicious_activity": bool(suspicious_activity)
            },
            "propagation_timeline": propagation_timeline,
            "propagation_confidence": float(0.85 if not suspicious_activity else 0.4)
        }
    
    def _analyze_uploader_history(self, video_path: str) -> Dict[str, Any]:
        """Analyze uploader history and credibility"""
        # In a real implementation, this would analyze social media profiles
        # For demonstration, we'll simulate the analysis
        
        # Simulate processing time
        time.sleep(0.2)
        
        # Simulate uploader analysis
        account_age_days = random.randint(30, 2000)
        total_uploads = random.randint(5, 200)
        previous_deepfake_uploads = random.randint(0, min(5, total_uploads // 10))
        
        # Determine verification status
        verification_status = random.choice(["verified", "unverified", "suspended"])
        
        # Calculate suspicious activity flags
        suspicious_flags = 0
        if previous_deepfake_uploads > 2:
            suspicious_flags += 1
        if account_age_days < 90:
            suspicious_flags += 1
        if total_uploads > 100 and account_age_days < 180:
            suspicious_flags += 1
        
        # Determine upload frequency
        if total_uploads / max(account_age_days, 1) > 0.5:
            upload_frequency = "high"
        elif total_uploads / max(account_age_days, 1) > 0.1:
            upload_frequency = "regular"
        else:
            upload_frequency = "low"
        
        # Simulate content categories
        content_categories = random.sample(
            ["entertainment", "news", "politics", "education", "sports", "technology"],
            random.randint(1, 4)
        )
        
        # Calculate credibility score
        credibility_score = 0.8  # Base score
        if verification_status == "verified":
            credibility_score += 0.1
        if previous_deepfake_uploads > 0:
            credibility_score -= previous_deepfake_uploads * 0.1
        if suspicious_flags > 0:
            credibility_score -= suspicious_flags * 0.15
        if account_age_days < 90:
            credibility_score -= 0.2
        
        credibility_score = max(0.0, min(1.0, credibility_score))
        
        return {
            "account_age_days": int(account_age_days),
            "total_uploads": int(total_uploads),
            "previous_deepfake_uploads": int(previous_deepfake_uploads),
            "account_verification_status": verification_status,
            "suspicious_activity_flags": int(suspicious_flags),
            "upload_frequency": upload_frequency,
            "content_categories": content_categories,
            "credibility_score": float(credibility_score),
            "risk_assessment": "high" if credibility_score < 0.4 else "medium" if credibility_score < 0.7 else "low"
        }
    
    def _analyze_metadata_correlation(self, video_path: str) -> Dict[str, Any]:
        """Analyze metadata correlation with known sources"""
        # In a real implementation, this would analyze EXIF data and correlate with databases
        # For demonstration, we'll simulate the analysis
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Simulate metadata analysis
        exif_data_consistency = random.choice([True, False])
        camera_model_detected = random.choice(["iPhone 13 Pro", "Samsung Galaxy S21", "Canon EOS R5", None])
        location_data_present = random.choice([True, False])
        timestamp_consistency = random.choice([True, False])
        
        # Simulate compression history
        compression_history = random.choice(["single_pass", "multiple_pass", "unknown"])
        
        # Simulate editing software detection
        editing_software = random.choice([None, "Adobe Premiere Pro", "Final Cut Pro", "DaVinci Resolve"])
        
        # Calculate correlation confidence
        correlation_confidence = 0.8  # Base score
        if exif_data_consistency:
            correlation_confidence += 0.1
        if camera_model_detected:
            correlation_confidence += 0.05
        if not location_data_present:
            correlation_confidence += 0.05  # Privacy-conscious upload
        if timestamp_consistency:
            correlation_confidence += 0.1
        if compression_history == "single_pass":
            correlation_confidence += 0.05
        if editing_software is None:
            correlation_confidence += 0.05  # No editing detected
        
        correlation_confidence = min(1.0, correlation_confidence)
        
        return {
            "exif_data_consistency": bool(exif_data_consistency),
            "camera_model_detected": camera_model_detected,
            "location_data_present": bool(location_data_present),
            "timestamp_consistency": bool(timestamp_consistency),
            "compression_history": compression_history,
            "editing_software_detected": editing_software,
            "correlation_confidence": float(correlation_confidence),
            "data_consistency": bool(exif_data_consistency and timestamp_consistency),
            "privacy_indicators": {
                "location_removed": bool(not location_data_present),
                "metadata_stripped": bool(not exif_data_consistency),
                "timestamp_modified": bool(not timestamp_consistency)
            }
        }
    
    def _calculate_contextual_confidence(self, reverse_search: Dict, social_network: Dict,
                                       uploader: Dict, metadata: Dict) -> float:
        """Calculate overall confidence in contextual analysis"""
        scores = []
        
        # Reverse search confidence
        if "error" not in reverse_search:
            scores.append(reverse_search.get("reverse_search_confidence", 0.0))
        
        # Social network propagation confidence
        if "error" not in social_network:
            scores.append(social_network.get("propagation_confidence", 0.0))
        
        # Uploader credibility score
        if "error" not in uploader:
            scores.append(uploader.get("credibility_score", 0.0))
        
        # Metadata correlation confidence
        if "error" not in metadata:
            scores.append(metadata.get("correlation_confidence", 0.0))
        
        if scores:
            return float(np.mean(scores))
        else:
            return 0.0
    
    def _generate_contextual_summary(self, reverse_search: Dict, social_network: Dict,
                                   uploader: Dict, metadata: Dict) -> Dict[str, Any]:
        """Generate summary of contextual analysis"""
        total_red_flags = 0
        concerns = []
        
        # Check reverse search results
        if reverse_search.get("exact_matches", 0) > 0:
            total_red_flags += 1
            concerns.append("Exact image matches found online")
        elif reverse_search.get("similar_images_found", 0) > 2:
            total_red_flags += 1
            concerns.append("Multiple similar images found online")
        
        # Check social network analysis
        if social_network.get("engagement_metrics", {}).get("suspicious_activity", False):
            total_red_flags += 1
            concerns.append("Suspicious social media activity detected")
        
        # Check uploader analysis
        if uploader.get("credibility_score", 1.0) < 0.4:
            total_red_flags += 1
            concerns.append("Low uploader credibility score")
        if uploader.get("previous_deepfake_uploads", 0) > 2:
            total_red_flags += 1
            concerns.append("Uploader has history of deepfake content")
        
        # Check metadata analysis
        if not metadata.get("data_consistency", True):
            total_red_flags += 1
            concerns.append("Metadata inconsistencies detected")
        
        if not concerns:
            concerns.append("No significant contextual concerns detected")
        
        return {
            "total_contextual_red_flags": int(total_red_flags),
            "analysis_quality": "high" if total_red_flags > 0 else "normal",
            "primary_concerns": concerns,
            "overall_risk_level": "high" if total_red_flags > 2 else "medium" if total_red_flags > 0 else "low"
        }
