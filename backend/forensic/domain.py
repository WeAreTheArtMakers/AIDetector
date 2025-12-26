"""
Domain Detection Module
========================
Detects image domain characteristics that affect AI detection accuracy.
"""

import logging
import numpy as np
from typing import Dict, Any
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

class DomainDetector:
    """Detect image domain characteristics"""
    
    def analyze(self, image: Image.Image, metadata: Dict) -> Dict[str, Any]:
        """
        Analyze image domain characteristics
        Returns domain classification and confidence adjustments
        """
        result = {
            "domain_type": "unknown",
            "characteristics": [],
            "confidence_penalty": 0.0,
            "ci_expansion": 0.0,
            "warnings": []
        }
        
        img_array = np.array(image.convert('RGB')).astype(np.float32)
        
        # Check for screenshot
        screenshot = self._detect_screenshot(image, img_array, metadata)
        if screenshot["is_screenshot"]:
            result["characteristics"].append("screenshot")
            result["confidence_penalty"] += 0.15
            result["ci_expansion"] += 0.10
            result["warnings"].append("Screenshot detected - AI detection less reliable")
        
        # Check for recompression
        recompress = self._detect_recompression(img_array)
        if recompress["is_recompressed"]:
            result["characteristics"].append("recompressed")
            result["confidence_penalty"] += 0.10
            result["ci_expansion"] += 0.08
            result["warnings"].append("Multiple compression detected")
        
        # Check for EXIF stripped
        if not metadata.get("exif") or len(metadata.get("exif", {})) < 3:
            result["characteristics"].append("exif_stripped")
            result["confidence_penalty"] += 0.05
            result["warnings"].append("EXIF metadata missing or stripped")
        
        # Check for resize artifacts
        resize = self._detect_resize(img_array)
        if resize["is_resized"]:
            result["characteristics"].append("resized")
            result["confidence_penalty"] += 0.05
            result["ci_expansion"] += 0.05
        
        # Check for crop
        crop = self._detect_crop(image, metadata)
        if crop["is_cropped"]:
            result["characteristics"].append("cropped")
            result["confidence_penalty"] += 0.03
        
        # Determine domain type
        if "screenshot" in result["characteristics"]:
            result["domain_type"] = "screenshot"
        elif "recompressed" in result["characteristics"]:
            result["domain_type"] = "recompressed"
        elif not result["characteristics"]:
            result["domain_type"] = "original"
        else:
            result["domain_type"] = "modified"
        
        return result
    
    def _detect_screenshot(self, image: Image.Image, img_array: np.ndarray, 
                          metadata: Dict) -> Dict:
        """Detect if image is a screenshot"""
        result = {"is_screenshot": False, "confidence": 0.0, "indicators": []}
        
        width, height = image.size
        
        # Common screenshot dimensions
        screenshot_sizes = [
            (1920, 1080), (2560, 1440), (1366, 768), (1440, 900),
            (1536, 864), (2880, 1800), (1280, 720), (3840, 2160),
            # Mobile
            (1170, 2532), (1284, 2778), (1080, 2400), (1440, 3200)
        ]
        
        if (width, height) in screenshot_sizes or (height, width) in screenshot_sizes:
            result["indicators"].append("screenshot_dimensions")
            result["confidence"] += 0.3
        
        # No EXIF but has specific dimensions
        if not metadata.get("exif") and result["confidence"] > 0:
            result["indicators"].append("no_exif_with_screen_dims")
            result["confidence"] += 0.2
        
        # Check for UI elements (solid color bars)
        top_bar = img_array[:50, :, :]
        bottom_bar = img_array[-50:, :, :]
        
        top_var = np.var(top_bar)
        bottom_var = np.var(bottom_bar)
        
        if top_var < 100 or bottom_var < 100:
            result["indicators"].append("ui_bars_detected")
            result["confidence"] += 0.2
        
        # Check for very uniform areas (UI backgrounds)
        gray = np.mean(img_array, axis=2)
        local_var = ndimage.uniform_filter(gray**2, size=20) - ndimage.uniform_filter(gray, size=20)**2
        very_uniform = np.sum(local_var < 5) / local_var.size
        
        if very_uniform > 0.3:
            result["indicators"].append("large_uniform_areas")
            result["confidence"] += 0.15
        
        result["is_screenshot"] = result["confidence"] >= 0.4
        return result
    
    def _detect_recompression(self, img_array: np.ndarray) -> Dict:
        """Detect double JPEG compression"""
        result = {"is_recompressed": False, "confidence": 0.0}
        
        try:
            gray = np.mean(img_array, axis=2)
            
            # DCT block analysis - look for 8x8 block artifacts
            h, w = gray.shape
            block_diffs = []
            
            # Check horizontal block boundaries
            for i in range(8, h - 8, 8):
                diff = np.abs(gray[i, :] - gray[i-1, :]).mean()
                block_diffs.append(diff)
            
            # Check vertical block boundaries
            for j in range(8, w - 8, 8):
                diff = np.abs(gray[:, j] - gray[:, j-1]).mean()
                block_diffs.append(diff)
            
            if block_diffs:
                avg_block_diff = np.mean(block_diffs)
                std_block_diff = np.std(block_diffs)
                
                # Double compression creates stronger block artifacts
                if avg_block_diff > 3 and std_block_diff < avg_block_diff * 0.5:
                    result["is_recompressed"] = True
                    result["confidence"] = min(0.8, avg_block_diff / 10)
                    
        except Exception as e:
            logger.warning(f"Recompression detection error: {e}")
        
        return result
    
    def _detect_resize(self, img_array: np.ndarray) -> Dict:
        """Detect resize artifacts"""
        result = {"is_resized": False, "confidence": 0.0}
        
        try:
            gray = np.mean(img_array, axis=2)
            
            # Check for interpolation artifacts using gradient analysis
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            
            # Resized images often have periodic patterns in gradients
            fft_x = np.abs(np.fft.fft(grad_x.mean(axis=0)))
            fft_y = np.abs(np.fft.fft(grad_y.mean(axis=1)))
            
            # Look for peaks that indicate interpolation
            x_peaks = np.sum(fft_x[10:] > fft_x.mean() * 3)
            y_peaks = np.sum(fft_y[10:] > fft_y.mean() * 3)
            
            if x_peaks > 5 or y_peaks > 5:
                result["is_resized"] = True
                result["confidence"] = 0.6
                
        except Exception as e:
            logger.warning(f"Resize detection error: {e}")
        
        return result
    
    def _detect_crop(self, image: Image.Image, metadata: Dict) -> Dict:
        """Detect if image was cropped"""
        result = {"is_cropped": False, "confidence": 0.0}
        
        # Check if dimensions don't match common camera ratios
        width, height = image.size
        ratio = width / height
        
        common_ratios = [4/3, 3/2, 16/9, 1.0, 3/4, 2/3, 9/16]
        
        is_common = any(abs(ratio - r) < 0.02 for r in common_ratios)
        
        if not is_common and metadata.get("camera", {}).get("has_camera_info"):
            result["is_cropped"] = True
            result["confidence"] = 0.5
        
        return result


def detect_domain(image: Image.Image, metadata: Dict) -> Dict[str, Any]:
    """Convenience function for domain detection"""
    detector = DomainDetector()
    return detector.analyze(image, metadata)