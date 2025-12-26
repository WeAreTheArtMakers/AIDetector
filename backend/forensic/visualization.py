"""
Visualization Module - Forensic Heatmap Overlays
=================================================
Generates heatmap overlays for forensic analysis:
- Global edit mode: shows global intensity heatmap (not "edited region" claims)
- Local manipulation mode: shows localized suspicion heatmap + top regions (bboxes)

Key principles:
- Never show local bboxes unless local_manipulation with boundary corroboration
- Global edit mode shows processing intensity, not splice claims
- Deterministic generation with SHA256 hashes
"""

import logging
import numpy as np
import hashlib
import base64
import io
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VisualizationMode(str, Enum):
    """Visualization mode constants"""
    NONE = "none"
    GLOBAL_EDIT = "global_edit"
    LOCAL_MANIPULATION = "local_manipulation"


@dataclass
class HeatmapResult:
    """Result of heatmap generation"""
    overlay_base64: str
    raw_map_base64: str
    hash_overlay_sha256: str
    hash_raw_sha256: str
    heatmap_type: str


class ForensicVisualizer:
    """
    Generates forensic visualization overlays.
    """
    
    def __init__(self):
        self.overlay_alpha = 0.5  # Transparency for overlay
        self.colormap_global = "viridis"  # Blue-green-yellow for global
        self.colormap_local = "hot"  # Red-yellow for local manipulation
    
    def generate_visualization(self,
                                image: Image.Image,
                                edit_assessment: Dict[str, Any],
                                manipulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for visualization generation.
        
        Returns:
            Dict with visualization mode, heatmap, regions, legend, notes
        """
        result = {
            "mode": VisualizationMode.NONE.value,
            "heatmap": None,
            "regions": [],
            "legend": [],
            "notes": []
        }
        
        if not edit_assessment or not manipulation_result:
            return result
        
        try:
            edit_type = edit_assessment.get("edit_type", "none_detected")
            boundary_corroborated = edit_assessment.get("boundary_corroborated", False)
            
            # Determine visualization mode
            mode = self._determine_mode(edit_type, boundary_corroborated)
            result["mode"] = mode.value
            
            if mode == VisualizationMode.NONE:
                result["notes"].append("No significant edits detected for visualization")
                return result
            
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            
            if mode == VisualizationMode.GLOBAL_EDIT:
                heatmap_result = self._generate_global_intensity_map(img_array, manipulation_result)
                result["heatmap"] = {
                    "type": "global_intensity",
                    "overlay_base64": heatmap_result.overlay_base64,
                    "raw_map_base64": heatmap_result.raw_map_base64,
                    "hash_overlay_sha256": heatmap_result.hash_overlay_sha256,
                    "hash_raw_sha256": heatmap_result.hash_raw_sha256
                }
                result["legend"] = [
                    "Blue/Green: Low processing intensity",
                    "Yellow/Orange: Medium processing intensity",
                    "Red: High processing intensity"
                ]
                result["notes"] = [
                    "Global processing intensity overlay (not proof of splice)",
                    "Edges and textures typically show stronger effect from filters/sharpening"
                ]
                
            elif mode == VisualizationMode.LOCAL_MANIPULATION:
                heatmap_result, regions = self._generate_local_suspicion_map(
                    img_array, manipulation_result, edit_assessment
                )
                result["heatmap"] = {
                    "type": "local_suspicion",
                    "overlay_base64": heatmap_result.overlay_base64,
                    "raw_map_base64": heatmap_result.raw_map_base64,
                    "hash_overlay_sha256": heatmap_result.hash_overlay_sha256,
                    "hash_raw_sha256": heatmap_result.hash_raw_sha256
                }
                result["regions"] = regions
                result["legend"] = [
                    "Yellow: Low suspicion",
                    "Orange: Medium suspicion",
                    "Red: High suspicion (boundary-corroborated)"
                ]
                result["notes"] = [
                    "Suspicious regions overlay (boundary-corroborated)",
                    "Regions indicate potential manipulation with edge/boundary evidence"
                ]
            
            logger.info(f"Visualization generated: mode={mode.value}")
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _determine_mode(self, edit_type: str, boundary_corroborated: bool) -> VisualizationMode:
        """
        Determine visualization mode based on edit assessment.
        
        Rules:
        - global_postprocess OR generator_artifacts => global_edit mode
        - local_manipulation AND boundary_corroborated => local_manipulation mode
        - else => none
        """
        if edit_type in ["global_postprocess", "generator_artifacts"]:
            return VisualizationMode.GLOBAL_EDIT
        
        if edit_type == "local_manipulation" and boundary_corroborated:
            return VisualizationMode.LOCAL_MANIPULATION
        
        return VisualizationMode.NONE
    
    def _generate_global_intensity_map(self, 
                                        img: np.ndarray,
                                        manipulation_result: Dict) -> HeatmapResult:
        """
        Generate global processing intensity heatmap.
        
        Shows WHERE global processing has stronger effect (edges/high-frequency zones),
        without claiming local splice.
        
        Computes weighted sum of:
        1. Local contrast change proxy (luminance local std dev)
        2. High-frequency boost proxy (Laplacian magnitude)
        3. Color transform intensity proxy (chroma distribution)
        """
        from scipy.ndimage import gaussian_filter, uniform_filter, laplace
        
        h, w = img.shape[:2]
        
        # Convert to grayscale for luminance analysis
        gray = np.mean(img, axis=2)
        
        # 1. Local contrast proxy (32x32 windows)
        window_size = 32
        local_mean = uniform_filter(gray, size=window_size)
        local_sq_mean = uniform_filter(gray**2, size=window_size)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # Normalize
        contrast_map = local_std / (np.max(local_std) + 1e-6)
        
        # 2. High-frequency boost proxy (Laplacian)
        lap = np.abs(laplace(gray))
        lap_smooth = gaussian_filter(lap, sigma=3)
        hf_map = lap_smooth / (np.max(lap_smooth) + 1e-6)
        
        # 3. Color transform intensity proxy
        # Compare local chroma to global mean
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        chroma = np.sqrt((r - g)**2 + (g - b)**2 + (b - r)**2)
        global_chroma_mean = np.mean(chroma)
        chroma_deviation = np.abs(chroma - global_chroma_mean)
        chroma_smooth = gaussian_filter(chroma_deviation, sigma=5)
        chroma_map = chroma_smooth / (np.max(chroma_smooth) + 1e-6)
        
        # Weighted combination
        intensity_map = (
            0.4 * contrast_map +
            0.35 * hf_map +
            0.25 * chroma_map
        )
        
        # Normalize to [0, 1]
        intensity_map = intensity_map / (np.max(intensity_map) + 1e-6)
        
        # Light smoothing
        intensity_map = gaussian_filter(intensity_map, sigma=2)
        
        # Create overlay
        overlay, raw_map = self._create_overlay(img, intensity_map, colormap="viridis")
        
        return self._encode_heatmap(overlay, raw_map, "global_intensity")
    
    def _generate_local_suspicion_map(self,
                                       img: np.ndarray,
                                       manipulation_result: Dict,
                                       edit_assessment: Dict) -> Tuple[HeatmapResult, List[Dict]]:
        """
        Generate local manipulation suspicion heatmap.
        
        Only called if boundary_corroborated is True.
        
        Fuses:
        - splice_noise_inconsistency_map
        - blur_mismatch_map
        - edge_matte_map (emphasized)
        - copy_move_map (emphasized)
        
        Applies border masking and emphasizes boundary-type evidence.
        """
        from scipy.ndimage import gaussian_filter
        
        h, w = img.shape[:2]
        
        # Initialize suspicion map
        suspicion_map = np.zeros((h, w), dtype=np.float32)
        
        # Collect regions from all methods with weights
        # Boundary-type methods get higher weight
        method_weights = {
            "edge_matte": 1.5,      # Boundary evidence - high weight
            "copy_move": 1.5,       # Boundary evidence - high weight
            "splice": 0.8,          # Noise-based - lower weight
            "blur_noise_mismatch": 0.6  # Texture-based - lowest weight
        }
        
        regions_output = []
        
        for method, weight in method_weights.items():
            method_data = manipulation_result.get(method, {})
            regions = method_data.get("regions", [])
            method_score = method_data.get("score", 0)
            
            for r in regions:
                x, y, rw, rh = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
                score = r.get("score", 0.5)
                
                # Clamp to image bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + rw), min(h, y + rh)
                
                if x2 > x1 and y2 > y1:
                    # Apply weighted score
                    weighted_score = score * weight
                    suspicion_map[y1:y2, x1:x2] = np.maximum(
                        suspicion_map[y1:y2, x1:x2], 
                        weighted_score
                    )
                    
                    # Add to regions output (only high-scoring boundary regions)
                    if score >= 0.5 and method in ["edge_matte", "copy_move", "splice"]:
                        reason_map = {
                            "edge_matte": "edge_matte",
                            "copy_move": "copy_move",
                            "splice": "splice_noise"
                        }
                        regions_output.append({
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1),
                            "score": float(score),
                            "reason": reason_map.get(method, method)
                        })
        
        # Apply border mask (suppress 4% border)
        border_margin = int(min(h, w) * 0.04)
        border_mask = np.ones((h, w), dtype=np.float32)
        border_mask[:border_margin, :] = 0.3
        border_mask[-border_margin:, :] = 0.3
        border_mask[:, :border_margin] = 0.3
        border_mask[:, -border_margin:] = 0.3
        
        suspicion_map = suspicion_map * border_mask
        
        # Normalize
        if np.max(suspicion_map) > 0:
            suspicion_map = suspicion_map / np.max(suspicion_map)
        
        # Light smoothing
        suspicion_map = gaussian_filter(suspicion_map, sigma=1.5)
        
        # Create overlay with "hot" colormap
        overlay, raw_map = self._create_overlay(img, suspicion_map, colormap="hot")
        
        # Sort regions by score and limit
        regions_output = sorted(regions_output, key=lambda r: -r["score"])[:10]
        
        return self._encode_heatmap(overlay, raw_map, "local_suspicion"), regions_output
    
    def _create_overlay(self, 
                        img: np.ndarray, 
                        heatmap: np.ndarray,
                        colormap: str = "viridis") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create RGBA overlay by blending heatmap with original image.
        
        Returns:
            Tuple of (overlay_rgba, raw_heatmap_rgba)
        """
        h, w = img.shape[:2]
        
        # Apply colormap
        if colormap == "viridis":
            colored = self._apply_viridis(heatmap)
        elif colormap == "hot":
            colored = self._apply_hot(heatmap)
        else:
            colored = self._apply_viridis(heatmap)
        
        # Create raw heatmap image (no blend)
        raw_map = np.zeros((h, w, 4), dtype=np.uint8)
        raw_map[:, :, :3] = colored
        raw_map[:, :, 3] = (heatmap * 255).astype(np.uint8)
        
        # Create blended overlay
        overlay = img.copy().astype(np.float32)
        alpha = heatmap[:, :, np.newaxis] * self.overlay_alpha
        
        # Blend: result = (1 - alpha) * original + alpha * heatmap_color
        overlay = (1 - alpha) * overlay + alpha * colored.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Add alpha channel
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_rgba[:, :, :3] = overlay
        overlay_rgba[:, :, 3] = 255
        
        return overlay_rgba, raw_map
    
    def _apply_viridis(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply viridis-like colormap (blue -> green -> yellow)"""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simplified viridis: blue -> teal -> green -> yellow
        # Low values: blue (68, 1, 84)
        # Mid values: teal (59, 82, 139) -> green (33, 145, 140)
        # High values: yellow (253, 231, 37)
        
        for i in range(h):
            for j in range(w):
                v = heatmap[i, j]
                if v < 0.25:
                    t = v / 0.25
                    colored[i, j] = [
                        int(68 + t * (59 - 68)),
                        int(1 + t * (82 - 1)),
                        int(84 + t * (139 - 84))
                    ]
                elif v < 0.5:
                    t = (v - 0.25) / 0.25
                    colored[i, j] = [
                        int(59 + t * (33 - 59)),
                        int(82 + t * (145 - 82)),
                        int(139 + t * (140 - 139))
                    ]
                elif v < 0.75:
                    t = (v - 0.5) / 0.25
                    colored[i, j] = [
                        int(33 + t * (94 - 33)),
                        int(145 + t * (201 - 145)),
                        int(140 + t * (98 - 140))
                    ]
                else:
                    t = (v - 0.75) / 0.25
                    colored[i, j] = [
                        int(94 + t * (253 - 94)),
                        int(201 + t * (231 - 201)),
                        int(98 + t * (37 - 98))
                    ]
        
        return colored
    
    def _apply_hot(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply hot colormap (black -> red -> yellow -> white)"""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Hot colormap: black -> red -> orange -> yellow -> white
        r = np.clip(heatmap * 3, 0, 1) * 255
        g = np.clip((heatmap - 0.33) * 3, 0, 1) * 255
        b = np.clip((heatmap - 0.67) * 3, 0, 1) * 255
        
        colored[:, :, 0] = r.astype(np.uint8)
        colored[:, :, 1] = g.astype(np.uint8)
        colored[:, :, 2] = b.astype(np.uint8)
        
        return colored
    
    def _encode_heatmap(self, 
                        overlay: np.ndarray, 
                        raw_map: np.ndarray,
                        heatmap_type: str) -> HeatmapResult:
        """Encode heatmaps to base64 and compute SHA256 hashes"""
        
        # Encode overlay
        overlay_img = Image.fromarray(overlay, mode='RGBA')
        overlay_buffer = io.BytesIO()
        overlay_img.save(overlay_buffer, format='PNG')
        overlay_bytes = overlay_buffer.getvalue()
        overlay_base64 = base64.b64encode(overlay_bytes).decode('utf-8')
        overlay_hash = hashlib.sha256(overlay_bytes).hexdigest()
        
        # Encode raw map
        raw_img = Image.fromarray(raw_map, mode='RGBA')
        raw_buffer = io.BytesIO()
        raw_img.save(raw_buffer, format='PNG')
        raw_bytes = raw_buffer.getvalue()
        raw_base64 = base64.b64encode(raw_bytes).decode('utf-8')
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        
        return HeatmapResult(
            overlay_base64=overlay_base64,
            raw_map_base64=raw_base64,
            hash_overlay_sha256=overlay_hash,
            hash_raw_sha256=raw_hash,
            heatmap_type=heatmap_type
        )


# Global instance
_visualizer: Optional[ForensicVisualizer] = None

def get_forensic_visualizer() -> ForensicVisualizer:
    global _visualizer
    if _visualizer is None:
        _visualizer = ForensicVisualizer()
    return _visualizer
