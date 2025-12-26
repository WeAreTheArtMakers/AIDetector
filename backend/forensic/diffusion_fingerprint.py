"""
Diffusion Fingerprint Module
=============================
Detects diffusion model artifacts through frequency-domain analysis,
noise residual statistics, and patch-level consistency checks.

Key signals:
- Radial power spectrum anomalies (diffusion roll-off patterns)
- SRM/high-pass residual statistics (kurtosis, cross-channel correlation)
- Patch-level diffusion score consistency
- Color channel correlation artifacts

Robust to resizing and recompression.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from scipy import fftpack
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


class DiffusionFingerprint:
    """
    Detects diffusion model fingerprints through multiple signal analysis.
    """
    
    def __init__(self):
        # Thresholds calibrated for diffusion detection
        self.diffusion_threshold = 0.65
        self.high_confidence_threshold = 0.80
        
        # SRM filter kernels (Spatial Rich Model - simplified)
        self.srm_filters = self._create_srm_filters()
    
    def _create_srm_filters(self) -> List[np.ndarray]:
        """Create simplified SRM high-pass filters"""
        filters = []
        
        # First-order edge filters
        filters.append(np.array([[-1, 1]], dtype=np.float32))
        filters.append(np.array([[-1], [1]], dtype=np.float32))
        
        # Second-order filters
        filters.append(np.array([[1, -2, 1]], dtype=np.float32))
        filters.append(np.array([[1], [-2], [1]], dtype=np.float32))
        
        # 3x3 high-pass
        filters.append(np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32) / 8.0)
        
        # Edge-aware filter
        filters.append(np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32) / 4.0)
        
        return filters
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Main analysis entry point.
        
        Returns:
            Dict with diffusion_score, confidence, and detailed signals
        """
        result = {
            "diffusion_score": 0.0,
            "confidence": "low",
            "signals": [],
            "notes": [],
            "spectrum_features": {},
            "residual_features": {},
            "patch_features": {},
            "channel_features": {}
        }
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            h, w = img_array.shape[:2]
            
            # Skip very small images
            if h < 128 or w < 128:
                result["notes"].append("Image too small for reliable analysis")
                return result
            
            # 1. Radial spectrum analysis
            spectrum_result = self._analyze_radial_spectrum(img_array)
            result["spectrum_features"] = spectrum_result
            
            # 2. SRM residual statistics
            residual_result = self._analyze_srm_residuals(img_array)
            result["residual_features"] = residual_result
            
            # 3. Patch-level consistency
            patch_result = self._analyze_patch_consistency(img_array)
            result["patch_features"] = patch_result
            
            # 4. Color channel correlation
            channel_result = self._analyze_channel_correlation(img_array)
            result["channel_features"] = channel_result
            
            # Combine signals into final score
            scores = []
            weights = []
            
            # Spectrum score (most reliable for diffusion)
            if spectrum_result.get("diffusion_indicator", 0) > 0:
                scores.append(spectrum_result["diffusion_indicator"])
                weights.append(0.35)
                if spectrum_result["diffusion_indicator"] > 0.6:
                    result["signals"].append("Radial spectrum shows diffusion roll-off pattern")
            
            # Residual score
            if residual_result.get("diffusion_indicator", 0) > 0:
                scores.append(residual_result["diffusion_indicator"])
                weights.append(0.30)
                if residual_result["diffusion_indicator"] > 0.6:
                    result["signals"].append("Noise residuals show diffusion characteristics")
            
            # Patch consistency score
            if patch_result.get("diffusion_indicator", 0) > 0:
                scores.append(patch_result["diffusion_indicator"])
                weights.append(0.20)
                if patch_result["diffusion_indicator"] > 0.6:
                    result["signals"].append("Patch-level texture consistency suggests AI generation")
            
            # Channel correlation score
            if channel_result.get("diffusion_indicator", 0) > 0:
                scores.append(channel_result["diffusion_indicator"])
                weights.append(0.15)
                if channel_result["diffusion_indicator"] > 0.6:
                    result["signals"].append("Color channel correlations suggest synthetic origin")
            
            # Weighted combination
            if scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
                result["diffusion_score"] = float(round(weighted_score, 4))
            
            # Determine confidence
            if result["diffusion_score"] >= self.high_confidence_threshold:
                result["confidence"] = "high"
            elif result["diffusion_score"] >= self.diffusion_threshold:
                result["confidence"] = "medium"
            else:
                result["confidence"] = "low"
            
            logger.info(f"Diffusion fingerprint: score={result['diffusion_score']:.3f}, "
                       f"confidence={result['confidence']}")
            
        except Exception as e:
            logger.error(f"Diffusion fingerprint analysis error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _analyze_radial_spectrum(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze radial power spectrum for diffusion artifacts.
        
        Diffusion models typically show:
        - Unnatural smoothness in high frequencies
        - Mid-frequency bumps from denoising process
        - Different roll-off pattern than camera photos
        """
        result = {
            "diffusion_indicator": 0.0,
            "roll_off_slope": 0.0,
            "mid_freq_bump": 0.0,
            "high_freq_energy": 0.0
        }
        
        try:
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            # Compute 2D FFT
            fft = fftpack.fft2(gray)
            fft_shift = fftpack.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Compute radial average
            center_y, center_x = h // 2, w // 2
            max_radius = min(center_y, center_x)
            
            radial_profile = []
            for r in range(1, max_radius):
                y, x = np.ogrid[:h, :w]
                mask = ((y - center_y)**2 + (x - center_x)**2 >= (r-1)**2) & \
                       ((y - center_y)**2 + (x - center_x)**2 < r**2)
                if np.sum(mask) > 0:
                    radial_profile.append(np.mean(magnitude[mask]))
            
            if len(radial_profile) < 10:
                return result
            
            radial_profile = np.array(radial_profile)
            radial_profile = radial_profile / (radial_profile[0] + 1e-6)  # Normalize
            
            # Analyze roll-off slope (log-log)
            log_freq = np.log(np.arange(1, len(radial_profile) + 1) + 1)
            log_power = np.log(radial_profile + 1e-10)
            
            # Fit slope in mid-frequency range
            mid_start = len(radial_profile) // 4
            mid_end = len(radial_profile) * 3 // 4
            
            if mid_end > mid_start + 5:
                slope = np.polyfit(log_freq[mid_start:mid_end], 
                                  log_power[mid_start:mid_end], 1)[0]
                result["roll_off_slope"] = float(slope)
                
                # Diffusion models typically have steeper roll-off (-2.5 to -3.5)
                # Camera photos typically have gentler roll-off (-1.5 to -2.5)
                if slope < -2.8:
                    result["diffusion_indicator"] += 0.4
                elif slope < -2.5:
                    result["diffusion_indicator"] += 0.2
            
            # Check for mid-frequency bump (diffusion artifact)
            mid_profile = radial_profile[mid_start:mid_end]
            if len(mid_profile) > 5:
                mid_mean = np.mean(mid_profile)
                mid_max = np.max(mid_profile)
                bump_ratio = mid_max / (mid_mean + 1e-6)
                result["mid_freq_bump"] = float(bump_ratio)
                
                if bump_ratio > 1.5:
                    result["diffusion_indicator"] += 0.3
                elif bump_ratio > 1.2:
                    result["diffusion_indicator"] += 0.15
            
            # Check high-frequency energy (diffusion tends to have less)
            high_start = len(radial_profile) * 3 // 4
            high_energy = np.mean(radial_profile[high_start:])
            result["high_freq_energy"] = float(high_energy)
            
            if high_energy < 0.01:
                result["diffusion_indicator"] += 0.3
            elif high_energy < 0.05:
                result["diffusion_indicator"] += 0.15
            
            result["diffusion_indicator"] = min(1.0, result["diffusion_indicator"])
            
        except Exception as e:
            logger.warning(f"Radial spectrum analysis error: {e}")
        
        return result
    
    def _analyze_srm_residuals(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze SRM (Spatial Rich Model) residuals for diffusion artifacts.
        
        Diffusion models show characteristic residual statistics:
        - Lower kurtosis (more Gaussian-like noise)
        - Different cross-channel correlations
        """
        result = {
            "diffusion_indicator": 0.0,
            "kurtosis_mean": 0.0,
            "skewness_mean": 0.0,
            "cross_channel_corr": 0.0
        }
        
        try:
            from scipy.signal import convolve2d
            
            # Process each channel
            channel_residuals = []
            kurtosis_values = []
            skewness_values = []
            
            for c in range(3):
                channel = img_array[:, :, c]
                
                # Apply SRM filters and collect residuals
                residuals = []
                for filt in self.srm_filters:
                    if filt.shape[0] == 1 or filt.shape[1] == 1:
                        # 1D filter - apply in appropriate direction
                        if filt.shape[0] == 1:
                            res = convolve2d(channel, filt, mode='valid', boundary='symm')
                        else:
                            res = convolve2d(channel, filt, mode='valid', boundary='symm')
                    else:
                        res = convolve2d(channel, filt, mode='valid', boundary='symm')
                    residuals.append(res.flatten())
                
                # Combine residuals
                combined = np.concatenate(residuals)
                channel_residuals.append(combined)
                
                # Compute statistics
                if len(combined) > 100:
                    k = kurtosis(combined, fisher=True)
                    s = skew(combined)
                    kurtosis_values.append(k)
                    skewness_values.append(s)
            
            if kurtosis_values:
                result["kurtosis_mean"] = float(np.mean(kurtosis_values))
                result["skewness_mean"] = float(np.mean(skewness_values))
                
                # Diffusion models tend to have lower kurtosis (more Gaussian)
                # Camera photos have higher kurtosis (more heavy-tailed)
                avg_kurtosis = np.mean(kurtosis_values)
                
                if avg_kurtosis < 3.0:  # Close to Gaussian (kurtosis=3)
                    result["diffusion_indicator"] += 0.4
                elif avg_kurtosis < 5.0:
                    result["diffusion_indicator"] += 0.2
                elif avg_kurtosis > 10.0:  # Heavy-tailed (camera-like)
                    result["diffusion_indicator"] -= 0.2
            
            # Cross-channel correlation
            if len(channel_residuals) == 3:
                # Truncate to same length
                min_len = min(len(r) for r in channel_residuals)
                r0 = channel_residuals[0][:min_len]
                r1 = channel_residuals[1][:min_len]
                r2 = channel_residuals[2][:min_len]
                
                corr_01 = np.corrcoef(r0, r1)[0, 1]
                corr_02 = np.corrcoef(r0, r2)[0, 1]
                corr_12 = np.corrcoef(r1, r2)[0, 1]
                
                avg_corr = (corr_01 + corr_02 + corr_12) / 3
                result["cross_channel_corr"] = float(avg_corr)
                
                # Diffusion models often have higher cross-channel correlation
                if avg_corr > 0.7:
                    result["diffusion_indicator"] += 0.3
                elif avg_corr > 0.5:
                    result["diffusion_indicator"] += 0.15
            
            result["diffusion_indicator"] = max(0, min(1.0, result["diffusion_indicator"]))
            
        except Exception as e:
            logger.warning(f"SRM residual analysis error: {e}")
        
        return result
    
    def _analyze_patch_consistency(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze patch-level consistency for diffusion artifacts.
        
        Diffusion models produce consistent texture across patches,
        while camera photos have more natural variation.
        """
        result = {
            "diffusion_indicator": 0.0,
            "texture_consistency": 0.0,
            "patch_variance_iqr": 0.0
        }
        
        try:
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            patch_size = 64
            stride = 32
            
            patch_variances = []
            patch_means = []
            
            for y in range(0, h - patch_size, stride):
                for x in range(0, w - patch_size, stride):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    
                    # Local variance (texture measure)
                    local_var = uniform_filter(patch**2, size=8) - \
                               uniform_filter(patch, size=8)**2
                    patch_variances.append(np.mean(local_var))
                    patch_means.append(np.mean(patch))
            
            if len(patch_variances) < 10:
                return result
            
            patch_variances = np.array(patch_variances)
            
            # Compute IQR of patch variances
            q25, q75 = np.percentile(patch_variances, [25, 75])
            iqr = q75 - q25
            median_var = np.median(patch_variances)
            
            result["patch_variance_iqr"] = float(iqr)
            
            # Texture consistency (lower IQR = more consistent = more likely diffusion)
            consistency = 1.0 - min(1.0, iqr / (median_var + 1e-6))
            result["texture_consistency"] = float(consistency)
            
            if consistency > 0.8:
                result["diffusion_indicator"] += 0.5
            elif consistency > 0.6:
                result["diffusion_indicator"] += 0.25
            
            # Also check if variance is unnaturally uniform
            var_of_var = np.var(patch_variances)
            if var_of_var < median_var * 0.1:
                result["diffusion_indicator"] += 0.3
            
            result["diffusion_indicator"] = min(1.0, result["diffusion_indicator"])
            
        except Exception as e:
            logger.warning(f"Patch consistency analysis error: {e}")
        
        return result
    
    def _analyze_channel_correlation(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze color channel correlations for diffusion artifacts.
        
        Diffusion models often produce different channel correlation
        patterns than camera sensors (which have CFA patterns).
        """
        result = {
            "diffusion_indicator": 0.0,
            "rg_correlation": 0.0,
            "rb_correlation": 0.0,
            "gb_correlation": 0.0,
            "cfa_artifact_score": 0.0
        }
        
        try:
            r = img_array[:, :, 0].flatten()
            g = img_array[:, :, 1].flatten()
            b = img_array[:, :, 2].flatten()
            
            # Sample for efficiency
            sample_size = min(100000, len(r))
            indices = np.random.choice(len(r), sample_size, replace=False)
            r, g, b = r[indices], g[indices], b[indices]
            
            # Compute correlations
            rg_corr = np.corrcoef(r, g)[0, 1]
            rb_corr = np.corrcoef(r, b)[0, 1]
            gb_corr = np.corrcoef(g, b)[0, 1]
            
            result["rg_correlation"] = float(rg_corr)
            result["rb_correlation"] = float(rb_corr)
            result["gb_correlation"] = float(gb_corr)
            
            # Diffusion models often have very high inter-channel correlation
            avg_corr = (rg_corr + rb_corr + gb_corr) / 3
            
            if avg_corr > 0.95:
                result["diffusion_indicator"] += 0.4
            elif avg_corr > 0.90:
                result["diffusion_indicator"] += 0.2
            
            # Check for CFA (Color Filter Array) artifacts
            # Camera photos often show slight checkerboard patterns
            cfa_score = self._detect_cfa_artifacts(img_array)
            result["cfa_artifact_score"] = float(cfa_score)
            
            # Low CFA score suggests non-camera origin
            if cfa_score < 0.1:
                result["diffusion_indicator"] += 0.3
            elif cfa_score > 0.3:
                result["diffusion_indicator"] -= 0.2
            
            result["diffusion_indicator"] = max(0, min(1.0, result["diffusion_indicator"]))
            
        except Exception as e:
            logger.warning(f"Channel correlation analysis error: {e}")
        
        return result
    
    def _detect_cfa_artifacts(self, img_array: np.ndarray) -> float:
        """
        Detect CFA (Bayer pattern) demosaicing artifacts.
        
        Camera photos often retain subtle CFA patterns,
        while AI-generated images don't have these.
        """
        try:
            # Use green channel (most sensitive to CFA)
            green = img_array[:, :, 1]
            h, w = green.shape
            
            # Check for 2x2 periodic patterns
            # Subsample at different phases
            even_even = green[0::2, 0::2]
            even_odd = green[0::2, 1::2]
            odd_even = green[1::2, 0::2]
            odd_odd = green[1::2, 1::2]
            
            # Compute variance differences between phases
            vars = [np.var(even_even), np.var(even_odd), 
                   np.var(odd_even), np.var(odd_odd)]
            
            var_range = max(vars) - min(vars)
            var_mean = np.mean(vars)
            
            # CFA artifacts create variance differences between phases
            cfa_score = var_range / (var_mean + 1e-6)
            
            return min(1.0, cfa_score)
            
        except Exception as e:
            logger.warning(f"CFA detection error: {e}")
            return 0.0


# Global instance
_diffusion_fingerprint: Optional[DiffusionFingerprint] = None

def get_diffusion_fingerprint() -> DiffusionFingerprint:
    global _diffusion_fingerprint
    if _diffusion_fingerprint is None:
        _diffusion_fingerprint = DiffusionFingerprint()
    return _diffusion_fingerprint
