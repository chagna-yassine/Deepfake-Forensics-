"""
Scene-physics & semantics analysis layer for deepfake detection.

This layer analyzes physical and semantic inconsistencies in videos including:
- Shadow consistency analysis
- Reflection analysis  
- Geometry and vanishing point analysis
- Object continuity tracking
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime

from ..core.base_layer import BaseForensicLayer


class PhysicsLayer(BaseForensicLayer):
    """
    Scene-physics & semantics analysis layer
    
    Analyzes physical inconsistencies that might indicate video manipulation:
    - Shadow consistency across frames
    - Reflection analysis
    - Geometry and vanishing points
    - Object continuity and tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Physics Layer
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        super().__init__("Physics_Layer")
        self.config = config or {}
        
        # Analysis parameters
        self.shadow_threshold = self.config.get('shadow_threshold', 0.3)
        self.reflection_threshold = self.config.get('reflection_threshold', 0.2)
        self.geometry_threshold = self.config.get('geometry_threshold', 0.1)
        self.continuity_threshold = self.config.get('continuity_threshold', 0.4)
        
        # Object detection parameters
        self.min_object_area = self.config.get('min_object_area', 1000)
        self.max_objects = self.config.get('max_objects', 10)
        
    def analyze(self, video_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video for scene-physics and semantic inconsistencies
        
        Args:
            video_path: Path to the video file
            options: Analysis options
            
        Returns:
            Dictionary containing physics analysis results
        """
        if not Path(video_path).exists():
            return {
                "status": "error",
                "error": "Invalid video file"
            }
        
        try:
            # Extract frames from video
            print(f"Physics Layer: Extracting frames from {video_path}")
            frames = self._extract_frames(video_path, max_frames=15)
            
            if len(frames) == 0:
                return {
                    "status": "error",
                    "error": "No frames extracted from video"
                }
            
            print(f"Physics Layer: Successfully extracted {len(frames)} frames")
            
            # Perform physics analysis
            print("Physics Layer: Starting shadow analysis...")
            try:
                shadow_analysis = self._analyze_shadows(frames)
                print("Physics Layer: Shadow analysis completed")
            except Exception as e:
                print(f"Physics Layer: Shadow analysis failed: {e}")
                shadow_analysis = {"error": str(e), "shadow_inconsistencies": False, "shadow_consistency_score": 0.0}
            
            print("Physics Layer: Starting reflection analysis...")
            try:
                reflection_analysis = self._analyze_reflections(frames)
                print("Physics Layer: Reflection analysis completed")
            except Exception as e:
                print(f"Physics Layer: Reflection analysis failed: {e}")
                reflection_analysis = {"error": str(e), "reflection_inconsistencies": False, "reflection_consistency_score": 0.0}
            
            print("Physics Layer: Starting geometry analysis...")
            try:
                geometry_analysis = self._analyze_geometry(frames)
                print("Physics Layer: Geometry analysis completed")
            except Exception as e:
                print(f"Physics Layer: Geometry analysis failed: {e}")
                geometry_analysis = {"error": str(e), "geometry_inconsistencies": False, "vanishing_point_consistency": 0.0}
            
            print("Physics Layer: Starting object continuity analysis...")
            try:
                continuity_analysis = self._analyze_object_continuity(frames)
                print("Physics Layer: Object continuity analysis completed")
            except Exception as e:
                print(f"Physics Layer: Object continuity analysis failed: {e}")
                continuity_analysis = {"error": str(e), "continuity_violations": False, "object_consistency_score": 0.0}
            
            # Calculate overall confidence
            confidence = self._calculate_physics_confidence(
                shadow_analysis, reflection_analysis, geometry_analysis, continuity_analysis
            )
            
            # Generate summary
            summary = self._generate_physics_summary(
                shadow_analysis, reflection_analysis, geometry_analysis, continuity_analysis
            )
            
            return {
                "status": "success",
                "layer_name": "Physics_Layer",
                "frames_analyzed": len(frames),
                "confidence": float(confidence),
                "shadow_analysis": shadow_analysis,
                "reflection_analysis": reflection_analysis,
                "geometry_analysis": geometry_analysis,
                "continuity_analysis": continuity_analysis,
                "summary": summary,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "layer_name": "Physics_Layer"
            }
    
    def _extract_frames(self, video_path: str, max_frames: int = 15) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
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
    
    def _analyze_shadows(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze shadow consistency across frames"""
        shadow_maps = []
        shadow_directions = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect shadows using edge detection and intensity analysis
            shadow_map = self._detect_shadows(gray)
            shadow_maps.append(shadow_map)
            
            # Estimate shadow direction
            direction = self._estimate_shadow_direction(shadow_map)
            shadow_directions.append(direction)
        
        # Analyze shadow consistency
        consistency_score = self._calculate_shadow_consistency(shadow_directions)
        inconsistencies = consistency_score < self.shadow_threshold
        
        return {
            "shadow_consistency_score": float(consistency_score),
            "shadow_inconsistencies": bool(inconsistencies),
            "average_shadow_direction": float(np.mean(shadow_directions)) if shadow_directions else 0.0,
            "shadow_direction_variance": float(np.var(shadow_directions)) if shadow_directions else 0.0,
            "frames_with_shadows": int(sum(1 for sm in shadow_maps if np.any(sm > 0))),
            "total_frames_analyzed": len(frames)
        }
    
    def _detect_shadows(self, gray_frame: np.ndarray) -> np.ndarray:
        """Detect shadows in a grayscale frame"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Use adaptive thresholding to detect dark regions
        shadow_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def _estimate_shadow_direction(self, shadow_map: np.ndarray) -> float:
        """Estimate the direction of shadows"""
        if not np.any(shadow_map > 0):
            return 0.0
        
        # Find contours of shadow regions
        contours, _ = cv2.findContours(shadow_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Calculate the dominant direction using the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a line to the contour
        if len(largest_contour) > 1:
            [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            direction = np.arctan2(vy, vx)[0]
            return float(direction)
        
        return 0.0
    
    def _calculate_shadow_consistency(self, shadow_directions: List[float]) -> float:
        """Calculate consistency of shadow directions across frames"""
        if len(shadow_directions) < 2:
            return 1.0
        
        # Calculate variance in shadow directions
        directions_array = np.array(shadow_directions)
        variance = np.var(directions_array)
        
        # Convert variance to consistency score (0-1, higher is more consistent)
        consistency = np.exp(-variance)
        return float(consistency)
    
    def _analyze_reflections(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze reflection consistency across frames"""
        reflection_maps = []
        reflection_qualities = []
        
        for frame in frames:
            # Detect reflections using edge detection and symmetry analysis
            reflection_map = self._detect_reflections(frame)
            reflection_maps.append(reflection_map)
            
            # Analyze reflection quality
            quality = self._analyze_reflection_quality(frame, reflection_map)
            reflection_qualities.append(quality)
        
        # Analyze reflection consistency
        consistency_score = self._calculate_reflection_consistency(reflection_qualities)
        inconsistencies = consistency_score < self.reflection_threshold
        
        return {
            "reflection_consistency_score": float(consistency_score),
            "reflection_inconsistencies": bool(inconsistencies),
            "average_reflection_quality": float(np.mean(reflection_qualities)) if reflection_qualities else 0.0,
            "reflection_quality_variance": float(np.var(reflection_qualities)) if reflection_qualities else 0.0,
            "frames_with_reflections": int(sum(1 for rm in reflection_maps if np.any(rm > 0))),
            "total_frames_analyzed": len(frames)
        }
    
    def _detect_reflections(self, frame: np.ndarray) -> np.ndarray:
        """Detect reflections in a frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find potential reflection boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines that might indicate reflections
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        reflection_mask = cv2.dilate(horizontal_lines, kernel, iterations=2)
        
        return reflection_mask
    
    def _analyze_reflection_quality(self, frame: np.ndarray, reflection_map: np.ndarray) -> float:
        """Analyze the quality of reflections"""
        if not np.any(reflection_map > 0):
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate symmetry in reflection regions
        reflection_regions = gray * (reflection_map > 0)
        
        # Simple symmetry analysis
        height, width = reflection_regions.shape
        top_half = reflection_regions[:height//2, :]
        bottom_half = reflection_regions[height//2:, :]
        
        if top_half.size > 0 and bottom_half.size > 0:
            # Flip bottom half and compare with top half
            bottom_flipped = cv2.flip(bottom_half, 0)
            
            # Resize to match dimensions
            min_height = min(top_half.shape[0], bottom_flipped.shape[0])
            if min_height > 0:
                top_resized = cv2.resize(top_half, (width, min_height))
                bottom_resized = cv2.resize(bottom_flipped, (width, min_height))
                
                # Calculate similarity
                similarity = cv2.matchTemplate(top_resized, bottom_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                return float(max(0, similarity))
        
        return 0.0
    
    def _calculate_reflection_consistency(self, reflection_qualities: List[float]) -> float:
        """Calculate consistency of reflection qualities across frames"""
        if len(reflection_qualities) < 2:
            return 1.0
        
        # Calculate variance in reflection qualities
        qualities_array = np.array(reflection_qualities)
        variance = np.var(qualities_array)
        
        # Convert variance to consistency score
        consistency = np.exp(-variance * 10)  # Scale factor for reflection analysis
        return float(consistency)
    
    def _analyze_geometry(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze geometric consistency and vanishing points"""
        vanishing_points = []
        perspective_scores = []
        
        for frame in frames:
            # Detect lines and vanishing points
            vanishing_point = self._detect_vanishing_point(frame)
            vanishing_points.append(vanishing_point)
            
            # Analyze perspective consistency
            perspective_score = self._analyze_perspective_consistency(frame)
            perspective_scores.append(perspective_score)
        
        # Analyze geometric consistency
        consistency_score = self._calculate_geometry_consistency(vanishing_points)
        inconsistencies = consistency_score < self.geometry_threshold
        
        return {
            "vanishing_point_consistency": float(consistency_score),
            "geometry_inconsistencies": bool(inconsistencies),
            "average_perspective_score": float(np.mean(perspective_scores)) if perspective_scores else 0.0,
            "perspective_variance": float(np.var(perspective_scores)) if perspective_scores else 0.0,
            "frames_with_geometry": int(sum(1 for vp in vanishing_points if vp is not None)),
            "total_frames_analyzed": len(frames)
        }
    
    def _detect_vanishing_point(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect vanishing point in a frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            return None
        
        # Group lines by angle and find intersections
        line_groups = self._group_lines_by_angle(lines)
        
        if len(line_groups) < 2:
            return None
        
        # Find vanishing point as intersection of line groups
        vanishing_point = self._find_line_intersections(line_groups)
        
        return vanishing_point
    
    def _group_lines_by_angle(self, lines: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Group lines by their angle"""
        line_groups = []
        angle_tolerance = np.pi / 18  # 10 degrees
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # Find existing group with similar angle
            added_to_group = False
            for group in line_groups:
                if group and abs(angle - group[0][1]) < angle_tolerance:
                    group.append((line[0], angle))
                    added_to_group = True
                    break
            
            if not added_to_group:
                line_groups.append([(line[0], angle)])
        
        return line_groups
    
    def _find_line_intersections(self, line_groups: List[List[Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
        """Find intersection point of line groups"""
        if len(line_groups) < 2:
            return None
        
        # Take the two largest groups
        line_groups.sort(key=len, reverse=True)
        group1, group2 = line_groups[0], line_groups[1]
        
        intersections = []
        
        for (line1, _), (line2, _) in zip(group1, group2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Calculate intersection point
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) > 1e-10:  # Avoid division by zero
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                intersections.append((px, py))
        
        if intersections:
            # Return the average intersection point
            avg_x = np.mean([p[0] for p in intersections])
            avg_y = np.mean([p[1] for p in intersections])
            return (float(avg_x), float(avg_y))
        
        return None
    
    def _analyze_perspective_consistency(self, frame: np.ndarray) -> float:
        """Analyze perspective consistency in a frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is None or len(corners) < 4:
            return 0.0
        
        # Analyze corner distribution for perspective consistency
        corner_points = corners.reshape(-1, 2)
        
        # Calculate perspective score based on corner distribution
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Check if corners follow perspective rules (closer to center should be denser)
        distances = np.sqrt((corner_points[:, 0] - center_x)**2 + (corner_points[:, 1] - center_y)**2)
        
        # Simple perspective consistency metric
        perspective_score = 1.0 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0.0
        
        return float(max(0, perspective_score))
    
    def _calculate_geometry_consistency(self, vanishing_points: List[Optional[Tuple[float, float]]]) -> float:
        """Calculate consistency of vanishing points across frames"""
        valid_points = [vp for vp in vanishing_points if vp is not None]
        
        if len(valid_points) < 2:
            return 1.0
        
        # Calculate variance in vanishing point positions
        x_coords = [vp[0] for vp in valid_points]
        y_coords = [vp[1] for vp in valid_points]
        
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        # Convert variance to consistency score
        consistency = np.exp(-(x_variance + y_variance) / 10000)  # Scale factor
        return float(consistency)
    
    def _analyze_object_continuity(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze object continuity across frames"""
        object_tracks = []
        continuity_scores = []
        
        # Detect objects in each frame
        for i, frame in enumerate(frames):
            objects = self._detect_objects(frame)
            object_tracks.append(objects)
        
        # Track objects across frames
        for i in range(len(frames) - 1):
            continuity_score = self._calculate_frame_continuity(object_tracks[i], object_tracks[i + 1])
            continuity_scores.append(continuity_score)
        
        # Analyze overall continuity
        avg_continuity = np.mean(continuity_scores) if continuity_scores else 1.0
        violations = avg_continuity < self.continuity_threshold
        
        return {
            "object_consistency_score": float(avg_continuity),
            "continuity_violations": bool(violations),
            "average_objects_per_frame": float(np.mean([len(objs) for objs in object_tracks])) if object_tracks else 0.0,
            "object_count_variance": float(np.var([len(objs) for objs in object_tracks])) if object_tracks else 0.0,
            "frames_analyzed": len(frames),
            "continuity_scores": [float(score) for score in continuity_scores]
        }
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use contour detection to find objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_object_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                objects.append({
                    "centroid": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": area,
                    "contour": contour
                })
        
        # Limit number of objects
        objects.sort(key=lambda x: x["area"], reverse=True)
        return objects[:self.max_objects]
    
    def _calculate_frame_continuity(self, objects1: List[Dict], objects2: List[Dict]) -> float:
        """Calculate continuity score between two frames"""
        if not objects1 or not objects2:
            return 1.0 if not objects1 and not objects2 else 0.0
        
        # Calculate distances between object centroids
        continuity_score = 0.0
        matched_objects = 0
        
        for obj1 in objects1:
            best_match = None
            best_distance = float('inf')
            
            for obj2 in objects2:
                # Calculate distance between centroids
                dist = np.sqrt((obj1["centroid"][0] - obj2["centroid"][0])**2 + 
                             (obj1["centroid"][1] - obj2["centroid"][1])**2)
                
                if dist < best_distance:
                    best_distance = dist
                    best_match = obj2
            
            # If we found a reasonable match, calculate continuity
            if best_match and best_distance < 100:  # Threshold for object matching
                # Calculate area similarity
                area_ratio = min(obj1["area"], best_match["area"]) / max(obj1["area"], best_match["area"])
                continuity_score += area_ratio
                matched_objects += 1
        
        # Normalize by number of objects
        if matched_objects > 0:
            continuity_score /= max(len(objects1), len(objects2))
        else:
            continuity_score = 0.0
        
        return float(continuity_score)
    
    def _calculate_physics_confidence(self, shadow_analysis: Dict, reflection_analysis: Dict,
                                    geometry_analysis: Dict, continuity_analysis: Dict) -> float:
        """Calculate overall confidence in physics analysis"""
        scores = []
        
        # Shadow consistency score
        if "error" not in shadow_analysis:
            scores.append(shadow_analysis.get("shadow_consistency_score", 0.0))
        
        # Reflection consistency score
        if "error" not in reflection_analysis:
            scores.append(reflection_analysis.get("reflection_consistency_score", 0.0))
        
        # Geometry consistency score
        if "error" not in geometry_analysis:
            scores.append(geometry_analysis.get("vanishing_point_consistency", 0.0))
        
        # Object continuity score
        if "error" not in continuity_analysis:
            scores.append(continuity_analysis.get("object_consistency_score", 0.0))
        
        if scores:
            return float(np.mean(scores))
        else:
            return 0.0
    
    def _generate_physics_summary(self, shadow_analysis: Dict, reflection_analysis: Dict,
                                geometry_analysis: Dict, continuity_analysis: Dict) -> Dict[str, Any]:
        """Generate summary of physics analysis"""
        total_inconsistencies = 0
        
        if shadow_analysis.get("shadow_inconsistencies", False):
            total_inconsistencies += 1
        if reflection_analysis.get("reflection_inconsistencies", False):
            total_inconsistencies += 1
        if geometry_analysis.get("geometry_inconsistencies", False):
            total_inconsistencies += 1
        if continuity_analysis.get("continuity_violations", False):
            total_inconsistencies += 1
        
        return {
            "total_physics_inconsistencies": int(total_inconsistencies),
            "analysis_quality": "high" if total_inconsistencies > 0 else "normal",
            "primary_concerns": self._identify_physics_concerns(
                shadow_analysis, reflection_analysis, geometry_analysis, continuity_analysis
            )
        }
    
    def _identify_physics_concerns(self, shadow_analysis: Dict, reflection_analysis: Dict,
                                 geometry_analysis: Dict, continuity_analysis: Dict) -> List[str]:
        """Identify primary physics concerns"""
        concerns = []
        
        if shadow_analysis.get("shadow_inconsistencies", False):
            concerns.append("Shadow inconsistencies detected")
        if reflection_analysis.get("reflection_inconsistencies", False):
            concerns.append("Reflection inconsistencies detected")
        if geometry_analysis.get("geometry_inconsistencies", False):
            concerns.append("Geometric inconsistencies detected")
        if continuity_analysis.get("continuity_violations", False):
            concerns.append("Object continuity violations detected")
        
        if not concerns:
            concerns.append("No significant physics inconsistencies detected")
        
        return concerns
