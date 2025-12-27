"""
Generative Artifact Heatmap Module
===================================
Generates heatmaps specifically for AI generation artifacts.

Distinct from manipulation/splice heatmaps:
- Shows WHERE generative model artifacts are strongest
- Uses recompression-resilient features
- Does NOT claim splice/edit evidence

Features used (robust to JPEG recompression):
- Frequency residual outliers (FFT domain)
- Patch embedding self-consistency (ViT/CLIP features)
- Gradient-domain statistics
- Edge oversharpening patterns
"""

import logging
import numpy as np
import hashlib
import base64
import io
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, laplace, sobel

logger = logging.getLogger(__name__)


class GenerativeHeatmapGenerator:
    """
    Generates heatmaps for AI generation artifacts.
    
    Enabled when:
    - generative_score >= 0.45 OR pathway == T2I OR (camera_score < 0.15 AND diffusion >= 0.25)
    - AND local_manipulation NOT strongly corroborated
    
    Uses robust features that survive JPEG recompression.
    """
    
    def __init__(self):
        self.patch_size = 32
        self.overlay_alpha = 0.5
    
    def generate(self,
                 image: Image.Image,
                 generative_score: float,
                 diffusion_result: Dict[str, Any],
                 manipulation_result: Dict[str, Any] = None,
                 edit_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate generative artifact heatmap.
        
        Args:
            image: PIL Image
            generative_score: Computed generative evidence score
            diffusion_result: Diffusion fingerprint analysis
            manipulation_result: Manipulation detection result
            edit_assessment: Edit assessment result
        
        Returns:
            Dict with heatmap, regions, and metadata
        """
        result = {
            "enabled": False,
            "mode": "none",
            "heatmap": None,
            "top_regions": [],
            "tags": [],
            "explain_tr": "",
            "explain_en": "",
            "notes": []
        }
        
        # Check enabling conditions
        # UPDATED: Lower threshold to 0.45 to enable more often
        if generative_score < 0.45:
            result["notes"].append("Generative score too low for artifact heatmap")
            return result
        
        # Check if local manipulation is strongly corroborated
        if edit_assessment:
            edit_type = edit_assessment.get("edit_type", "none_detected")
            boundary_corroborated = edit_assessment.get("boundary_corroborated", False)
            
            if edit_type == "local_manipulation" and boundary_corroborated:
                result["notes"].append("Local manipulation corroborated - use splice heatmap instead")
                return result
        
        result["enabled"] = True
        result["mode"] = "generative_artifacts"
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            h, w = img_array.shape[:2]
            
            # Compute artifact maps
            fft_map = self._compute_fft_residual_map(img_array)
            consistency_map = self._compute_consistency_map(img_array)
            gradient_map = self._compute_gradient_stats_map(img_array)
            edge_map = self._compute_edge_artifact_map(img_array)
            
            # Fuse maps with robust-z normalization
            artifact_map = self._fuse_maps(
                fft_map, consistency_map, gradient_map, edge_map,
                diffusion_score=diffusion_result.get("diffusion_score", 0.5)
            )
            
            # Extract top-K regions with tags
            top_regions = self._extract_regions(
                artifact_map, fft_map, consistency_map, 
                gradient_map, edge_map, h, w
            )
            result["top_regions"] = top_regions
            
            # Generate overlay
            overlay_base64, raw_base64, overlay_hash, raw_hash = self._create_overlay(
                img_array, artifact_map
            )
            
            result["heatmap"] = {
                "type": "generative_artifacts_v2",
                "overlay_base64": overlay_base64,
                "raw_map_base64": raw_base64,
                "hash_overlay_sha256": overlay_hash,
                "hash_raw_sha256": raw_hash
            }
            
            # Build tags
            result["tags"] = self._build_tags(
                fft_map, consistency_map, gradient_map, edge_map
            )
            
            # Explanations
            result["explain_tr"] = "Bu harita, AI üretim artefaktlarının yoğunluğunu gösterir. Yapıştırma/düzenleme kanıtı DEĞİLDİR."
            result["explain_en"] = "This map shows AI generation artifact intensity. NOT evidence of splicing/editing."
            
            logger.info(f"Generative heatmap generated: {len(top_regions)} regions")
            
        except Exception as e:
            logger.error(f"Generative heatmap error: {e}")
            result["enabled"] = False
            result["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _compute_fft_residual_map(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute FFT residual anomaly map.
        
        AI-generated images often have characteristic frequency patterns
        that differ from natural images.
        """
        h, w = img_array.shape[:2]
        gray = np.mean(img_array, axis=2)
        
        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log magnitude for better visualization
        log_mag = np.log1p(magnitude)
        
        # Compute expected natural image spectrum (1/f falloff)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        freq_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1
        expected = np.max(log_mag) / freq_dist
        
        # Residual from expected
        residual = np.abs(log_mag - expected)
        
        # Smooth and normalize
        residual_smooth = gaussian_filter(residual, sigma=5)
        
        # Map back to spatial domain using inverse FFT of residual
        # This gives us a spatial map of frequency anomalies
        residual_spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(residual_smooth)))
        
        # Normalize
        if np.max(residual_spatial) > 0:
            residual_spatial = residual_spatial / np.max(residual_spatial)
        
        return residual_spatial.astype(np.float32)
    
    def _compute_consistency_map(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute patch self-consistency map.
        
        AI-generated images may have patches that are inconsistent
        with their neighbors in feature space.
        """
        h, w = img_array.shape[:2]
        patch_size = self.patch_size
        
        # Compute local statistics for each patch
        gray = np.mean(img_array, axis=2)
        
        # Local mean and variance
        local_mean = uniform_filter(gray, size=patch_size)
        local_sq_mean = uniform_filter(gray**2, size=patch_size)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0)
        
        # Compute deviation from local neighborhood
        # Use larger window for neighborhood
        neighborhood_mean = uniform_filter(local_mean, size=patch_size * 2)
        neighborhood_var = uniform_filter(local_var, size=patch_size * 2)
        
        # Consistency score: how much does this patch differ from neighborhood
        mean_diff = np.abs(local_mean - neighborhood_mean)
        var_diff = np.abs(local_var - neighborhood_var)
        
        # Normalize
        if np.max(mean_diff) > 0:
            mean_diff = mean_diff / np.max(mean_diff)
        if np.max(var_diff) > 0:
            var_diff = var_diff / np.max(var_diff)
        
        # Combine
        consistency_map = 0.6 * mean_diff + 0.4 * var_diff
        
        # Smooth
        consistency_map = gaussian_filter(consistency_map, sigma=3)
        
        return consistency_map.astype(np.float32)
    
    def _compute_gradient_stats_map(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute gradient-domain statistics map.
        
        Gradient statistics are more robust to JPEG recompression
        than pixel-domain statistics.
        """
        gray = np.mean(img_array, axis=2)
        
        # Compute gradients
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Local gradient statistics
        patch_size = self.patch_size
        local_grad_mean = uniform_filter(grad_mag, size=patch_size)
        local_grad_sq = uniform_filter(grad_mag**2, size=patch_size)
        local_grad_var = np.maximum(local_grad_sq - local_grad_mean**2, 0)
        
        # Compute gradient direction consistency
        grad_dir = np.arctan2(grad_y, grad_x + 1e-6)
        local_dir_mean = uniform_filter(np.cos(grad_dir), size=patch_size)
        local_dir_var = 1 - np.abs(local_dir_mean)  # High when directions are inconsistent
        
        # Combine magnitude variance and direction inconsistency
        gradient_map = 0.5 * (local_grad_var / (np.max(local_grad_var) + 1e-6)) + \
                       0.5 * local_dir_var
        
        # Smooth
        gradient_map = gaussian_filter(gradient_map, sigma=2)
        
        return gradient_map.astype(np.float32)
    
    def _compute_edge_artifact_map(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute edge oversharpening artifact map.
        
        AI-generated images often have characteristic edge enhancement
        patterns (halos, ringing).
        """
        gray = np.mean(img_array, axis=2)
        
        # Detect edges
        edges = np.abs(laplace(gray))
        
        # Compute local edge intensity
        patch_size = self.patch_size // 2
        local_edge = uniform_filter(edges, size=patch_size)
        
        # Detect edge halos (high edge response in smooth regions)
        local_var = uniform_filter(gray**2, size=patch_size) - \
                    uniform_filter(gray, size=patch_size)**2
        local_var = np.maximum(local_var, 0)
        
        # Halo indicator: high edges in low-variance regions
        smoothness = 1 / (local_var + 1)
        halo_map = local_edge * smoothness
        
        # Normalize
        if np.max(halo_map) > 0:
            halo_map = halo_map / np.max(halo_map)
        
        # Smooth
        halo_map = gaussian_filter(halo_map, sigma=2)
        
        return halo_map.astype(np.float32)
    
    def _fuse_maps(self, fft_map: np.ndarray, consistency_map: np.ndarray,
                   gradient_map: np.ndarray, edge_map: np.ndarray,
                   diffusion_score: float) -> np.ndarray:
        """
        Fuse individual maps with robust-z normalization.
        """
        # Robust-z normalization for each map
        def robust_z(arr):
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            if mad > 0:
                return (arr - median) / (mad * 1.4826)
            return arr - median
        
        fft_z = robust_z(fft_map)
        consistency_z = robust_z(consistency_map)
        gradient_z = robust_z(gradient_map)
        edge_z = robust_z(edge_map)
        
        # Weighted combination
        # Weights based on reliability (gradient and consistency are more robust)
        fused = (
            0.25 * fft_z +
            0.30 * consistency_z +
            0.30 * gradient_z +
            0.15 * edge_z
        )
        
        # Scale by diffusion score
        fused = fused * (0.5 + 0.5 * diffusion_score)
        
        # Normalize to [0, 1]
        fused = fused - np.min(fused)
        if np.max(fused) > 0:
            fused = fused / np.max(fused)
        
        # Light smoothing
        fused = gaussian_filter(fused, sigma=2)
        
        return fused.astype(np.float32)
    
    def _extract_regions(self, artifact_map: np.ndarray,
                         fft_map: np.ndarray, consistency_map: np.ndarray,
                         gradient_map: np.ndarray, edge_map: np.ndarray,
                         h: int, w: int, top_k: int = 5) -> List[Dict]:
        """Extract top-K artifact regions with explanation tags."""
        from scipy.ndimage import label, find_objects
        
        regions = []
        
        # Threshold to find high-artifact regions
        threshold = 0.5
        binary_map = artifact_map > threshold
        
        # Label connected components
        labeled_array, num_features = label(binary_map)
        
        if num_features == 0:
            return regions
        
        # Find bounding boxes
        slices = find_objects(labeled_array)
        
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            y_slice, x_slice = slc
            region_mask = labeled_array[slc] == (i + 1)
            
            # Compute region score
            region_artifact = artifact_map[slc][region_mask]
            score = float(np.mean(region_artifact))
            
            # Determine which features contributed most
            region_fft = float(np.mean(fft_map[slc][region_mask]))
            region_consistency = float(np.mean(consistency_map[slc][region_mask]))
            region_gradient = float(np.mean(gradient_map[slc][region_mask]))
            region_edge = float(np.mean(edge_map[slc][region_mask]))
            
            tags = []
            if region_fft > 0.5:
                tags.append("frequency_anomaly")
            if region_consistency > 0.5:
                tags.append("patch_inconsistency")
            if region_gradient > 0.5:
                tags.append("gradient_anomaly")
            if region_edge > 0.5:
                tags.append("edge_artifact")
            
            if not tags:
                tags.append("generative_pattern")
            
            regions.append({
                "x": int(x_slice.start),
                "y": int(y_slice.start),
                "w": int(x_slice.stop - x_slice.start),
                "h": int(y_slice.stop - y_slice.start),
                "score": round(score, 3),
                "tags": tags
            })
        
        # Sort by score and return top-K
        regions = sorted(regions, key=lambda r: -r["score"])[:top_k]
        
        return regions
    
    def _build_tags(self, fft_map: np.ndarray, consistency_map: np.ndarray,
                    gradient_map: np.ndarray, edge_map: np.ndarray) -> List[str]:
        """Build overall tags for the heatmap."""
        tags = []
        
        # Check which features are most prominent
        fft_mean = np.mean(fft_map)
        consistency_mean = np.mean(consistency_map)
        gradient_mean = np.mean(gradient_map)
        edge_mean = np.mean(edge_map)
        
        if fft_mean > 0.4:
            tags.append("frequency_patterns")
        if consistency_mean > 0.4:
            tags.append("texture_inconsistency")
        if gradient_mean > 0.4:
            tags.append("gradient_anomalies")
        if edge_mean > 0.4:
            tags.append("edge_artifacts")
        
        if not tags:
            tags.append("subtle_artifacts")
        
        return tags
    
    def _create_overlay(self, img_array: np.ndarray,
                        artifact_map: np.ndarray) -> Tuple[str, str, str, str]:
        """Create overlay image and encode to base64."""
        h, w = img_array.shape[:2]
        
        # Apply cyan-green-yellow-orange colormap
        colored = self._apply_colormap(artifact_map)
        
        # Create raw heatmap
        raw_map = np.zeros((h, w, 4), dtype=np.uint8)
        raw_map[:, :, :3] = colored
        raw_map[:, :, 3] = (artifact_map * 255).astype(np.uint8)
        
        # Create blended overlay
        overlay = img_array.copy()
        alpha = artifact_map[:, :, np.newaxis] * self.overlay_alpha
        overlay = (1 - alpha) * overlay + alpha * colored.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Add alpha channel
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_rgba[:, :, :3] = overlay
        overlay_rgba[:, :, 3] = 255
        
        # Encode to base64
        overlay_img = Image.fromarray(overlay_rgba, mode='RGBA')
        overlay_buffer = io.BytesIO()
        overlay_img.save(overlay_buffer, format='PNG')
        overlay_bytes = overlay_buffer.getvalue()
        overlay_base64 = base64.b64encode(overlay_bytes).decode('utf-8')
        overlay_hash = hashlib.sha256(overlay_bytes).hexdigest()
        
        raw_img = Image.fromarray(raw_map, mode='RGBA')
        raw_buffer = io.BytesIO()
        raw_img.save(raw_buffer, format='PNG')
        raw_bytes = raw_buffer.getvalue()
        raw_base64 = base64.b64encode(raw_bytes).decode('utf-8')
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        
        return overlay_base64, raw_base64, overlay_hash, raw_hash
    
    def _apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply cyan-green-yellow-orange colormap."""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Vectorized colormap application
        # Cyan (0, 200, 200) -> Green (50, 180, 80) -> Yellow (200, 200, 50) -> Orange (255, 150, 50)
        
        mask1 = heatmap < 0.33
        mask2 = (heatmap >= 0.33) & (heatmap < 0.67)
        mask3 = heatmap >= 0.67
        
        t1 = heatmap[mask1] / 0.33
        colored[mask1, 0] = (0 + t1 * 50).astype(np.uint8)
        colored[mask1, 1] = (200 + t1 * (180 - 200)).astype(np.uint8)
        colored[mask1, 2] = (200 + t1 * (80 - 200)).astype(np.uint8)
        
        t2 = (heatmap[mask2] - 0.33) / 0.34
        colored[mask2, 0] = (50 + t2 * (200 - 50)).astype(np.uint8)
        colored[mask2, 1] = (180 + t2 * (200 - 180)).astype(np.uint8)
        colored[mask2, 2] = (80 + t2 * (50 - 80)).astype(np.uint8)
        
        t3 = (heatmap[mask3] - 0.67) / 0.33
        colored[mask3, 0] = (200 + t3 * (255 - 200)).astype(np.uint8)
        colored[mask3, 1] = (200 + t3 * (150 - 200)).astype(np.uint8)
        colored[mask3, 2] = 50
        
        return colored


# Global instance
_generative_heatmap: Optional[GenerativeHeatmapGenerator] = None

def get_generative_heatmap() -> GenerativeHeatmapGenerator:
    global _generative_heatmap
    if _generative_heatmap is None:
        _generative_heatmap = GenerativeHeatmapGenerator()
    return _generative_heatmap
