"""
Manipulation Detection Module - Localization & Evidence
========================================================
Detects and localizes image manipulations:
- Copy-move detection
- Splice detection via noise inconsistency
- Edge matte/halo detection
- Blur/noise mismatch between regions
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass
import hashlib
import base64
import io

logger = logging.getLogger(__name__)


def _to_native(val):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


@dataclass
class Region:
    """Detected suspicious region"""
    x: int
    y: int
    w: int
    h: int
    score: float
    method: str
    note: str = ""


class ManipulationDetector:
    """
    Detects image manipulations and provides localization.
    Outputs heatmaps and bounding boxes for suspicious regions.
    """
    
    def __init__(self):
        self.patch_size = 64
        self.stride = 32
        self.min_region_score = 0.4
    
    def analyze(self, image: Image.Image, deep_scan: bool = False) -> Dict[str, Any]:
        """
        Main analysis entry point.
        
        Args:
            image: PIL Image to analyze
            deep_scan: If True, run all expensive detection methods
        
        Returns:
            Dict with manipulation scores, regions, and optional heatmap
        """
        result = {
            "enabled": True,
            "copy_move": {"score": 0.0, "regions": [], "notes": []},
            "splice": {"score": 0.0, "regions": [], "notes": []},
            "edge_matte": {"score": 0.0, "regions": [], "notes": []},
            "blur_noise_mismatch": {"score": 0.0, "regions": [], "notes": []},
            "overall_score": 0.0,
            "evidence_strong": False
        }
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            
            # Always run basic detection
            result["splice"] = self._detect_splice_noise(img_array)
            result["blur_noise_mismatch"] = self._detect_blur_noise_mismatch(img_array)
            
            if deep_scan:
                result["copy_move"] = self._detect_copy_move(img_array)
                result["edge_matte"] = self._detect_edge_matte(img_array)
            
            # Calculate overall score
            scores = [
                result["splice"]["score"],
                result["blur_noise_mismatch"]["score"],
                result["copy_move"]["score"],
                result["edge_matte"]["score"]
            ]
            result["overall_score"] = float(max(scores)) if scores else 0.0
            
            # Evidence is strong if any module has score >= 0.60 with regions
            result["evidence_strong"] = any([
                result["splice"]["score"] >= 0.60 and len(result["splice"]["regions"]) > 0,
                result["copy_move"]["score"] >= 0.60 and len(result["copy_move"]["regions"]) > 0,
                result["edge_matte"]["score"] >= 0.60 and len(result["edge_matte"]["regions"]) > 0,
                result["blur_noise_mismatch"]["score"] >= 0.60 and len(result["blur_noise_mismatch"]["regions"]) > 0
            ])
            
        except Exception as e:
            logger.error(f"Manipulation detection error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _detect_splice_noise(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Detect splicing via noise inconsistency analysis.
        Compares local noise statistics across image patches.
        """
        result = {"score": 0.0, "regions": [], "notes": []}
        
        try:
            from scipy.ndimage import gaussian_filter, uniform_filter
            
            # Convert to grayscale
            gray = np.mean(img, axis=2)
            
            # Extract noise residual (high-pass filter)
            smoothed = gaussian_filter(gray, sigma=2.0)
            noise = gray - smoothed
            
            h, w = gray.shape
            patch_scores = []
            
            # Compute local noise statistics per patch
            for y in range(0, h - self.patch_size, self.stride):
                for x in range(0, w - self.patch_size, self.stride):
                    patch = noise[y:y+self.patch_size, x:x+self.patch_size]
                    local_std = np.std(patch)
                    local_mean = np.mean(np.abs(patch))
                    patch_scores.append({
                        "x": x, "y": y,
                        "std": local_std,
                        "mean": local_mean
                    })
            
            if not patch_scores:
                return result
            
            # Compute global statistics
            all_stds = [p["std"] for p in patch_scores]
            global_std = np.std(all_stds)
            global_mean_std = np.mean(all_stds)
            
            # Find anomalous patches (significantly different noise level)
            threshold = global_mean_std + 2 * global_std
            anomalous = []
            
            for p in patch_scores:
                deviation = abs(p["std"] - global_mean_std) / (global_std + 1e-6)
                if deviation > 2.0:
                    anomalous.append({
                        "x": int(p["x"]), "y": int(p["y"]),
                        "w": int(self.patch_size), "h": int(self.patch_size),
                        "score": float(min(1.0, deviation / 4.0)),
                        "method": "splice_noise"
                    })
            
            # Merge nearby regions
            merged = self._merge_regions(anomalous)
            
            if merged:
                result["regions"] = merged[:5]  # Top 5 regions
                result["score"] = float(max(r["score"] for r in merged))
                result["notes"].append(f"Found {len(merged)} noise-inconsistent regions")
            
            # Add note about noise variance
            if global_std > 3.0:
                result["notes"].append("High noise variance across image")
            
        except Exception as e:
            logger.warning(f"Splice detection error: {e}")
            result["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _detect_blur_noise_mismatch(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Detect blur/noise mismatch between foreground and background.
        Composites often have inconsistent blur/noise characteristics.
        """
        result = {"score": 0.0, "regions": [], "notes": []}
        
        try:
            from scipy.ndimage import sobel, gaussian_filter, laplace
            
            gray = np.mean(img, axis=2)
            
            # Compute edge strength (blur indicator)
            edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
            
            # Compute local sharpness via Laplacian variance
            lap = np.abs(laplace(gray))
            
            h, w = gray.shape
            patch_data = []
            
            for y in range(0, h - self.patch_size, self.stride):
                for x in range(0, w - self.patch_size, self.stride):
                    edge_patch = edges[y:y+self.patch_size, x:x+self.patch_size]
                    lap_patch = lap[y:y+self.patch_size, x:x+self.patch_size]
                    
                    edge_mean = np.mean(edge_patch)
                    lap_var = np.var(lap_patch)
                    
                    patch_data.append({
                        "x": x, "y": y,
                        "edge_mean": edge_mean,
                        "lap_var": lap_var
                    })
            
            if not patch_data:
                return result
            
            # Compute statistics
            edge_means = [p["edge_mean"] for p in patch_data]
            lap_vars = [p["lap_var"] for p in patch_data]
            
            global_edge_std = np.std(edge_means)
            global_edge_mean = np.mean(edge_means)
            global_lap_std = np.std(lap_vars)
            global_lap_mean = np.mean(lap_vars)
            
            # Find patches with unusual blur/sharpness
            anomalous = []
            for p in patch_data:
                edge_dev = abs(p["edge_mean"] - global_edge_mean) / (global_edge_std + 1e-6)
                lap_dev = abs(p["lap_var"] - global_lap_mean) / (global_lap_std + 1e-6)
                
                combined_dev = (edge_dev + lap_dev) / 2
                
                if combined_dev > 2.0:
                    anomalous.append({
                        "x": int(p["x"]), "y": int(p["y"]),
                        "w": int(self.patch_size), "h": int(self.patch_size),
                        "score": float(min(1.0, combined_dev / 4.0)),
                        "method": "blur_noise_mismatch"
                    })
            
            merged = self._merge_regions(anomalous)
            
            if merged:
                result["regions"] = merged[:5]
                result["score"] = float(max(r["score"] for r in merged))
                result["notes"].append(f"Found {len(merged)} blur/noise inconsistent regions")
            
        except Exception as e:
            logger.warning(f"Blur/noise mismatch detection error: {e}")
            result["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _detect_copy_move(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Detect copy-move forgery using block matching.
        """
        result = {"score": 0.0, "regions": [], "notes": []}
        
        try:
            from scipy.fftpack import dct
            
            gray = np.mean(img, axis=2)
            h, w = gray.shape
            
            block_size = 16
            blocks = []
            
            # Extract DCT features from blocks
            for y in range(0, h - block_size, block_size // 2):
                for x in range(0, w - block_size, block_size // 2):
                    block = gray[y:y+block_size, x:x+block_size]
                    # Compute DCT and take low-frequency coefficients
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    features = dct_block[:4, :4].flatten()
                    blocks.append({
                        "x": x, "y": y,
                        "features": features
                    })
            
            if len(blocks) < 10:
                return result
            
            # Find similar blocks (potential copy-move)
            matches = []
            features_array = np.array([b["features"] for b in blocks])
            
            # Simple pairwise comparison (limited for performance)
            sample_size = min(500, len(blocks))
            indices = np.random.choice(len(blocks), sample_size, replace=False)
            
            for i in indices:
                for j in indices:
                    if i >= j:
                        continue
                    
                    # Skip nearby blocks
                    dist_x = abs(blocks[i]["x"] - blocks[j]["x"])
                    dist_y = abs(blocks[i]["y"] - blocks[j]["y"])
                    if dist_x < block_size * 2 and dist_y < block_size * 2:
                        continue
                    
                    # Compare features
                    diff = np.linalg.norm(features_array[i] - features_array[j])
                    if diff < 5.0:  # Threshold for similarity
                        matches.append({
                            "block1": blocks[i],
                            "block2": blocks[j],
                            "similarity": 1.0 - diff / 10.0
                        })
            
            if matches:
                # Group matches into regions
                for m in matches[:10]:
                    result["regions"].append({
                        "x": int(m["block1"]["x"]), "y": int(m["block1"]["y"]),
                        "w": int(block_size), "h": int(block_size),
                        "score": float(m["similarity"]),
                        "method": "copy_move"
                    })
                    result["regions"].append({
                        "x": int(m["block2"]["x"]), "y": int(m["block2"]["y"]),
                        "w": int(block_size), "h": int(block_size),
                        "score": float(m["similarity"]),
                        "method": "copy_move"
                    })
                
                result["score"] = float(max(m["similarity"] for m in matches))
                result["notes"].append(f"Found {len(matches)} potential copy-move pairs")
            
        except Exception as e:
            logger.warning(f"Copy-move detection error: {e}")
            result["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _detect_edge_matte(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Detect edge matting artifacts (halos around composited objects).
        """
        result = {"score": 0.0, "regions": [], "notes": []}
        
        try:
            from scipy.ndimage import sobel, gaussian_filter
            
            gray = np.mean(img, axis=2)
            
            # Detect edges
            edge_x = sobel(gray, axis=1)
            edge_y = sobel(gray, axis=0)
            edges = np.sqrt(edge_x**2 + edge_y**2)
            
            # Smooth edges to find halos
            edges_smooth = gaussian_filter(edges, sigma=3)
            
            # Halo = difference between sharp edge and smoothed
            halo_map = np.abs(edges - edges_smooth)
            
            # Normalize
            halo_map = halo_map / (np.max(halo_map) + 1e-6)
            
            # Find high-halo regions
            threshold = 0.3
            halo_mask = halo_map > threshold
            
            # Find connected components
            from scipy.ndimage import label, find_objects
            labeled, num_features = label(halo_mask)
            
            if num_features > 0:
                slices = find_objects(labeled)
                regions = []
                
                for i, s in enumerate(slices[:10]):
                    if s is None:
                        continue
                    y_slice, x_slice = s
                    region_halo = halo_map[s]
                    score = float(np.mean(region_halo))
                    
                    if score > 0.2:
                        regions.append({
                            "x": int(x_slice.start),
                            "y": int(y_slice.start),
                            "w": int(x_slice.stop - x_slice.start),
                            "h": int(y_slice.stop - y_slice.start),
                            "score": score,
                            "method": "edge_matte"
                        })
                
                if regions:
                    result["regions"] = sorted(regions, key=lambda r: -r["score"])[:5]
                    result["score"] = float(max(r["score"] for r in regions))
                    result["notes"].append(f"Found {len(regions)} potential edge matte regions")
            
        except Exception as e:
            logger.warning(f"Edge matte detection error: {e}")
            result["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _merge_regions(self, regions: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping regions"""
        if not regions:
            return []
        
        # Sort by score descending
        sorted_regions = sorted(regions, key=lambda r: -r["score"])
        merged = []
        
        for region in sorted_regions:
            # Check if overlaps with existing merged region
            overlaps = False
            for m in merged:
                # Simple overlap check
                if (region["x"] < m["x"] + m["w"] and
                    region["x"] + region["w"] > m["x"] and
                    region["y"] < m["y"] + m["h"] and
                    region["y"] + region["h"] > m["y"]):
                    # Merge by expanding
                    new_x = min(m["x"], region["x"])
                    new_y = min(m["y"], region["y"])
                    new_w = max(m["x"] + m["w"], region["x"] + region["w"]) - new_x
                    new_h = max(m["y"] + m["h"], region["y"] + region["h"]) - new_y
                    m["x"] = int(new_x)
                    m["y"] = int(new_y)
                    m["w"] = int(new_w)
                    m["h"] = int(new_h)
                    m["score"] = float(max(m["score"], region["score"]))
                    overlaps = True
                    break
            
            if not overlaps:
                # Ensure all values are native Python types
                merged.append({
                    "x": int(region["x"]),
                    "y": int(region["y"]),
                    "w": int(region["w"]),
                    "h": int(region["h"]),
                    "score": float(region["score"]),
                    "method": region.get("method", "unknown")
                })
        
        return merged
    
    def generate_heatmap(self, image: Image.Image, manipulation_result: Dict) -> Optional[str]:
        """
        Generate a heatmap overlay showing suspicious regions.
        Returns base64-encoded PNG.
        """
        try:
            import io
            
            img_array = np.array(image.convert('RGBA'))
            h, w = img_array.shape[:2]
            
            # Create heatmap layer
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Add regions from all methods
            for method in ["splice", "copy_move", "edge_matte", "blur_noise_mismatch"]:
                regions = manipulation_result.get(method, {}).get("regions", [])
                for r in regions:
                    x, y, rw, rh = r["x"], r["y"], r["w"], r["h"]
                    score = r["score"]
                    # Clamp to image bounds
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + rw), min(h, y + rh)
                    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], score)
            
            # Normalize and colorize
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Create RGBA overlay (red channel for suspicious areas)
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red
            overlay[:, :, 3] = (heatmap * 180).astype(np.uint8)  # Alpha
            
            # Blend with original
            result = img_array.copy()
            alpha = overlay[:, :, 3:4] / 255.0
            result[:, :, :3] = (1 - alpha) * result[:, :, :3] + alpha * overlay[:, :, :3]
            
            # Convert to PIL and encode
            result_img = Image.fromarray(result.astype(np.uint8))
            buffer = io.BytesIO()
            result_img.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return None


# Global instance
_manipulation_detector: Optional[ManipulationDetector] = None

def get_manipulation_detector() -> ManipulationDetector:
    global _manipulation_detector
    if _manipulation_detector is None:
        _manipulation_detector = ManipulationDetector()
    return _manipulation_detector
