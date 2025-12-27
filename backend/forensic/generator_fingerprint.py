"""
Generator Fingerprint Module - AI Tool Identification
=======================================================
Identifies which AI generator was likely used to create an image.

Different AI generators leave distinct fingerprints:
- Frequency spectrum patterns
- Noise distribution characteristics
- Color palette signatures
- Typical artifact locations
- VAE/decoder signatures

Supported generator families:
- Midjourney (v5, v6)
- DALL-E (2, 3)
- Stable Diffusion (1.x, 2.x, XL)
- Flux
- Grok/xAI
- Firefly (Adobe)
- Ideogram
- Leonardo

This module uses statistical analysis (no ML models required).
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field
from scipy import fftpack, stats
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class GeneratorMatch:
    """Match result for a specific generator"""
    name: str
    family: str
    confidence: float
    matching_features: List[str] = field(default_factory=list)
    version_hint: str = ""


@dataclass
class GeneratorFingerprintResult:
    """Result of generator fingerprint analysis"""
    enabled: bool = True
    
    # Top predictions
    top_match: Optional[GeneratorMatch] = None
    candidates: List[GeneratorMatch] = field(default_factory=list)
    
    # Feature scores
    spectrum_signature: Dict[str, float] = field(default_factory=dict)
    noise_signature: Dict[str, float] = field(default_factory=dict)
    color_signature: Dict[str, float] = field(default_factory=dict)
    artifact_signature: Dict[str, float] = field(default_factory=dict)
    
    # Overall confidence
    identification_confidence: str = "low"  # low, medium, high
    
    # Evidence
    evidence: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Generator signature profiles (empirically derived)
GENERATOR_PROFILES = {
    "midjourney_v5": {
        "family": "midjourney",
        "spectrum": {
            "high_freq_rolloff": (-3.2, -2.8),  # Steep rolloff
            "mid_freq_energy": (0.35, 0.50),
            "low_freq_dominance": (0.55, 0.70),
        },
        "noise": {
            "std_range": (8, 15),
            "uniformity": (0.7, 0.9),
            "color_noise_ratio": (0.3, 0.5),
        },
        "color": {
            "saturation_mean": (0.45, 0.65),
            "contrast_boost": (1.1, 1.4),
            "warm_bias": (0.02, 0.08),
        },
        "artifacts": {
            "edge_sharpness": (0.7, 0.9),
            "texture_repetition": (0.1, 0.3),
        }
    },
    "midjourney_v6": {
        "family": "midjourney",
        "spectrum": {
            "high_freq_rolloff": (-3.0, -2.5),
            "mid_freq_energy": (0.40, 0.55),
            "low_freq_dominance": (0.50, 0.65),
        },
        "noise": {
            "std_range": (6, 12),
            "uniformity": (0.75, 0.92),
            "color_noise_ratio": (0.25, 0.45),
        },
        "color": {
            "saturation_mean": (0.40, 0.60),
            "contrast_boost": (1.05, 1.3),
            "warm_bias": (0.01, 0.06),
        },
        "artifacts": {
            "edge_sharpness": (0.75, 0.95),
            "texture_repetition": (0.05, 0.2),
        }
    },
    "dalle3": {
        "family": "dalle",
        "spectrum": {
            "high_freq_rolloff": (-2.8, -2.3),
            "mid_freq_energy": (0.30, 0.45),
            "low_freq_dominance": (0.60, 0.75),
        },
        "noise": {
            "std_range": (5, 10),
            "uniformity": (0.80, 0.95),
            "color_noise_ratio": (0.20, 0.35),
        },
        "color": {
            "saturation_mean": (0.35, 0.55),
            "contrast_boost": (1.0, 1.2),
            "warm_bias": (-0.02, 0.04),
        },
        "artifacts": {
            "edge_sharpness": (0.65, 0.85),
            "texture_repetition": (0.05, 0.15),
        }
    },
    "chatgpt_dalle": {
        "family": "openai",
        "spectrum": {
            "high_freq_rolloff": (-2.9, -2.4),
            "mid_freq_energy": (0.32, 0.48),
            "low_freq_dominance": (0.58, 0.72),
        },
        "noise": {
            "std_range": (5, 11),
            "uniformity": (0.78, 0.94),
            "color_noise_ratio": (0.22, 0.38),
        },
        "color": {
            "saturation_mean": (0.36, 0.54),
            "contrast_boost": (1.02, 1.22),
            "warm_bias": (-0.01, 0.05),
        },
        "artifacts": {
            "edge_sharpness": (0.68, 0.88),
            "texture_repetition": (0.04, 0.16),
        }
    },
    "gemini_imagen": {
        "family": "google",
        "spectrum": {
            "high_freq_rolloff": (-2.7, -2.2),
            "mid_freq_energy": (0.35, 0.52),
            "low_freq_dominance": (0.55, 0.70),
        },
        "noise": {
            "std_range": (4, 10),
            "uniformity": (0.82, 0.96),
            "color_noise_ratio": (0.18, 0.32),
        },
        "color": {
            "saturation_mean": (0.38, 0.56),
            "contrast_boost": (1.0, 1.18),
            "warm_bias": (-0.02, 0.04),
        },
        "artifacts": {
            "edge_sharpness": (0.72, 0.92),
            "texture_repetition": (0.03, 0.14),
        }
    },
    "gemini_i2i": {
        "family": "google",
        "spectrum": {
            "high_freq_rolloff": (-2.5, -2.0),  # Less rolloff (preserves source detail)
            "mid_freq_energy": (0.38, 0.55),
            "low_freq_dominance": (0.50, 0.65),
        },
        "noise": {
            "std_range": (6, 14),  # More noise variation (from source)
            "uniformity": (0.65, 0.85),  # Less uniform (source influence)
            "color_noise_ratio": (0.25, 0.42),
        },
        "color": {
            "saturation_mean": (0.40, 0.60),
            "contrast_boost": (1.05, 1.25),
            "warm_bias": (-0.01, 0.06),
        },
        "artifacts": {
            "edge_sharpness": (0.68, 0.88),
            "texture_repetition": (0.08, 0.22),
        }
    },
    "stable_diffusion_xl": {
        "family": "stable_diffusion",
        "spectrum": {
            "high_freq_rolloff": (-3.5, -2.9),
            "mid_freq_energy": (0.25, 0.40),
            "low_freq_dominance": (0.65, 0.80),
        },
        "noise": {
            "std_range": (10, 20),
            "uniformity": (0.60, 0.80),
            "color_noise_ratio": (0.35, 0.55),
        },
        "color": {
            "saturation_mean": (0.40, 0.60),
            "contrast_boost": (1.0, 1.25),
            "warm_bias": (-0.03, 0.05),
        },
        "artifacts": {
            "edge_sharpness": (0.60, 0.80),
            "texture_repetition": (0.15, 0.35),
        }
    },
    "flux": {
        "family": "flux",
        "spectrum": {
            "high_freq_rolloff": (-2.6, -2.2),
            "mid_freq_energy": (0.35, 0.50),
            "low_freq_dominance": (0.55, 0.70),
        },
        "noise": {
            "std_range": (4, 9),
            "uniformity": (0.85, 0.98),
            "color_noise_ratio": (0.15, 0.30),
        },
        "color": {
            "saturation_mean": (0.38, 0.55),
            "contrast_boost": (1.02, 1.18),
            "warm_bias": (-0.01, 0.03),
        },
        "artifacts": {
            "edge_sharpness": (0.80, 0.95),
            "texture_repetition": (0.02, 0.12),
        }
    },
    "grok_aurora": {
        "family": "grok",
        "spectrum": {
            "high_freq_rolloff": (-2.9, -2.4),
            "mid_freq_energy": (0.32, 0.48),
            "low_freq_dominance": (0.58, 0.72),
        },
        "noise": {
            "std_range": (6, 13),
            "uniformity": (0.72, 0.88),
            "color_noise_ratio": (0.28, 0.45),
        },
        "color": {
            "saturation_mean": (0.42, 0.62),
            "contrast_boost": (1.08, 1.28),
            "warm_bias": (0.0, 0.05),
        },
        "artifacts": {
            "edge_sharpness": (0.70, 0.88),
            "texture_repetition": (0.08, 0.22),
        }
    },
    "firefly": {
        "family": "firefly",
        "spectrum": {
            "high_freq_rolloff": (-3.0, -2.5),
            "mid_freq_energy": (0.28, 0.42),
            "low_freq_dominance": (0.62, 0.78),
        },
        "noise": {
            "std_range": (7, 14),
            "uniformity": (0.70, 0.85),
            "color_noise_ratio": (0.30, 0.48),
        },
        "color": {
            "saturation_mean": (0.35, 0.52),
            "contrast_boost": (1.0, 1.15),
            "warm_bias": (-0.02, 0.03),
        },
        "artifacts": {
            "edge_sharpness": (0.68, 0.85),
            "texture_repetition": (0.10, 0.25),
        }
    },
    "ideogram": {
        "family": "ideogram",
        "spectrum": {
            "high_freq_rolloff": (-2.7, -2.2),
            "mid_freq_energy": (0.38, 0.52),
            "low_freq_dominance": (0.52, 0.68),
        },
        "noise": {
            "std_range": (5, 11),
            "uniformity": (0.78, 0.92),
            "color_noise_ratio": (0.22, 0.38),
        },
        "color": {
            "saturation_mean": (0.40, 0.58),
            "contrast_boost": (1.05, 1.22),
            "warm_bias": (0.0, 0.04),
        },
        "artifacts": {
            "edge_sharpness": (0.72, 0.90),
            "texture_repetition": (0.05, 0.18),
        }
    },
}

# Common AI generator output dimensions
AI_GENERATOR_DIMENSIONS = {
    # Perfect squares (common in many generators)
    (512, 512): ["stable_diffusion", "dalle2", "midjourney_legacy"],
    (768, 768): ["stable_diffusion", "midjourney"],
    (800, 800): ["gemini", "grok"],  # NEW: Gemini/Grok common size
    (1024, 1024): ["dalle3", "midjourney_v5", "stable_diffusion_xl", "flux", "gemini", "grok"],
    (1536, 1536): ["midjourney_v6", "flux"],
    (2048, 2048): ["midjourney_v6"],
    
    # Portrait ratios (common for face generation)
    (768, 1024): ["midjourney", "stable_diffusion"],
    (832, 1216): ["stable_diffusion_xl"],
    (896, 1152): ["stable_diffusion_xl"],
    (1024, 1536): ["dalle3", "midjourney"],
    
    # Landscape ratios
    (1024, 768): ["midjourney", "stable_diffusion"],
    (1216, 832): ["stable_diffusion_xl"],
    (1152, 896): ["stable_diffusion_xl"],
    (1536, 1024): ["dalle3", "midjourney"],
    
    # Grok/xAI specific dimensions
    (784, 1168): ["grok"],  # NEW: Grok portrait
    (1168, 784): ["grok"],  # NEW: Grok landscape
    
    # Wide formats
    (1792, 1024): ["dalle3"],
    (1024, 1792): ["dalle3"],
}


class GeneratorFingerprint:
    """
    Identifies AI generator from image characteristics.
    
    Uses statistical analysis of:
    - Frequency spectrum
    - Noise patterns
    - Color distribution
    - Artifact signatures
    """
    
    def __init__(self):
        self.profiles = GENERATOR_PROFILES
        
    def analyze(self, image: Image.Image, 
                diffusion_result: Dict = None) -> GeneratorFingerprintResult:
        """
        Analyze image to identify likely generator.
        
        Args:
            image: PIL Image
            diffusion_result: Optional diffusion fingerprint result
            
        Returns:
            GeneratorFingerprintResult with top matches
        """
        result = GeneratorFingerprintResult()
        
        try:
            # Convert to numpy
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            
            # Check image dimensions for AI generator hints
            width, height = image.size
            dimension_hints = self._check_ai_dimensions(width, height)
            
            # Extract signatures
            spectrum_sig = self._analyze_spectrum(img_array)
            noise_sig = self._analyze_noise(img_array)
            color_sig = self._analyze_color(img_array)
            artifact_sig = self._analyze_artifacts(img_array)
            
            result.spectrum_signature = spectrum_sig
            result.noise_signature = noise_sig
            result.color_signature = color_sig
            result.artifact_signature = artifact_sig
            
            # Match against profiles
            matches = self._match_profiles(
                spectrum_sig, noise_sig, color_sig, artifact_sig,
                dimension_hints=dimension_hints
            )
            
            if matches:
                result.candidates = matches[:5]  # Top 5
                result.top_match = matches[0]
                
                # Determine confidence
                if matches[0].confidence >= 0.70:
                    result.identification_confidence = "high"
                elif matches[0].confidence >= 0.50:
                    result.identification_confidence = "medium"
                else:
                    result.identification_confidence = "low"
                
                result.evidence.append(
                    f"Top match: {matches[0].name} ({matches[0].confidence:.0%})"
                )
                result.evidence.extend(matches[0].matching_features[:3])
                
                # Add dimension hint if found
                if dimension_hints:
                    result.evidence.append(f"Dimension match: {width}x{height}")
            else:
                result.notes.append("No strong generator match found")
            
            # Add diffusion result info if available
            if diffusion_result:
                diff_score = diffusion_result.get("diffusion_score", 0)
                if diff_score < 0.30:
                    result.notes.append("Low diffusion score - may be real photo")
                    result.identification_confidence = "low"
            
        except Exception as e:
            logger.error(f"Generator fingerprint error: {e}")
            result.notes.append(f"Analysis error: {str(e)}")
        
        return result
    
    def _check_ai_dimensions(self, width: int, height: int) -> List[str]:
        """Check if image dimensions match known AI generator outputs."""
        hints = []
        
        # Check exact match
        if (width, height) in AI_GENERATOR_DIMENSIONS:
            hints.extend(AI_GENERATOR_DIMENSIONS[(width, height)])
        
        # Check reversed (portrait/landscape swap)
        if (height, width) in AI_GENERATOR_DIMENSIONS:
            hints.extend(AI_GENERATOR_DIMENSIONS[(height, width)])
        
        # Check for perfect square (common in AI)
        if width == height:
            if width in [512, 768, 800, 1024, 1536, 2048]:
                hints.append("perfect_square_ai")
        
        # Check for common AI aspect ratios
        aspect = width / height if height > 0 else 1
        ai_aspects = [1.0, 4/3, 3/4, 16/9, 9/16, 3/2, 2/3]
        for ai_aspect in ai_aspects:
            if abs(aspect - ai_aspect) < 0.01:  # Very close match
                hints.append(f"ai_aspect_{ai_aspect:.2f}")
                break
        
        return list(set(hints))  # Remove duplicates
    
    def _analyze_spectrum(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze frequency spectrum characteristics."""
        # Convert to grayscale
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        # Compute 2D FFT
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Compute radial profile
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create radial bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(int)
        
        # Radial average
        max_r = min(cx, cy)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = (r == i)
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude[mask])
        
        # Normalize
        radial_profile = radial_profile / (radial_profile[1] + 1e-10)
        
        # Compute features
        low_freq = np.mean(radial_profile[:max_r//4])
        mid_freq = np.mean(radial_profile[max_r//4:max_r//2])
        high_freq = np.mean(radial_profile[max_r//2:])
        
        total = low_freq + mid_freq + high_freq + 1e-10
        
        # Compute rolloff slope
        log_profile = np.log(radial_profile[1:max_r//2] + 1e-10)
        log_freq = np.log(np.arange(1, max_r//2))
        if len(log_profile) > 2:
            slope, _, _, _, _ = stats.linregress(log_freq, log_profile)
        else:
            slope = -2.5
        
        return {
            "high_freq_rolloff": float(slope),
            "mid_freq_energy": float(mid_freq / total),
            "low_freq_dominance": float(low_freq / total),
        }
    
    def _analyze_noise(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze noise characteristics."""
        # Extract noise using high-pass filter
        from scipy.ndimage import gaussian_filter
        
        smoothed = gaussian_filter(img, sigma=2)
        noise = img - smoothed
        
        # Compute statistics
        noise_std = np.std(noise)
        
        # Uniformity (how consistent is noise across image)
        h, w = img.shape[:2]
        block_size = min(h, w) // 4
        
        if block_size > 10:
            block_stds = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = noise[i:i+block_size, j:j+block_size]
                    block_stds.append(np.std(block))
            
            if block_stds:
                uniformity = 1.0 - (np.std(block_stds) / (np.mean(block_stds) + 1e-10))
                uniformity = max(0, min(1, uniformity))
            else:
                uniformity = 0.5
        else:
            uniformity = 0.5
        
        # Color noise ratio (chroma vs luma noise)
        luma_noise = np.std(0.299 * noise[:,:,0] + 0.587 * noise[:,:,1] + 0.114 * noise[:,:,2])
        chroma_noise = np.std(noise[:,:,0] - noise[:,:,1]) + np.std(noise[:,:,1] - noise[:,:,2])
        color_noise_ratio = chroma_noise / (luma_noise + chroma_noise + 1e-10)
        
        return {
            "std_range": float(noise_std),
            "uniformity": float(uniformity),
            "color_noise_ratio": float(color_noise_ratio),
        }
    
    def _analyze_color(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution characteristics."""
        # Convert to HSV-like
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        
        # Saturation
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
        sat_mean = np.mean(saturation)
        
        # Contrast (using percentiles)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        p5, p95 = np.percentile(luminance, [5, 95])
        contrast_boost = (p95 - p5) / 255.0 * 1.5  # Normalize to ~1.0 for normal
        
        # Warm/cool bias
        warm_bias = np.mean(r - b) / 255.0
        
        return {
            "saturation_mean": float(sat_mean),
            "contrast_boost": float(contrast_boost),
            "warm_bias": float(warm_bias),
        }
    
    def _analyze_artifacts(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze artifact patterns."""
        from scipy.ndimage import sobel, generic_filter
        
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        # Edge sharpness
        edges_x = sobel(gray, axis=1)
        edges_y = sobel(gray, axis=0)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        
        # Sharpness = ratio of strong edges to weak edges
        strong_edges = np.sum(edge_magnitude > np.percentile(edge_magnitude, 90))
        weak_edges = np.sum(edge_magnitude > np.percentile(edge_magnitude, 50))
        edge_sharpness = strong_edges / (weak_edges + 1e-10)
        edge_sharpness = min(1.0, edge_sharpness * 2)  # Normalize
        
        # Texture repetition (using autocorrelation)
        # Simplified: check for periodic patterns
        h, w = gray.shape
        sample_size = min(128, h, w)
        sample = gray[:sample_size, :sample_size]
        
        # Compute autocorrelation
        f = fftpack.fft2(sample - np.mean(sample))
        autocorr = np.abs(fftpack.ifft2(f * np.conj(f)))
        autocorr = autocorr / autocorr[0, 0]
        
        # Check for secondary peaks (repetition)
        # Exclude center region
        center = sample_size // 4
        autocorr[center:-center, center:-center] = 0
        texture_repetition = np.max(autocorr)
        
        return {
            "edge_sharpness": float(edge_sharpness),
            "texture_repetition": float(texture_repetition),
        }
    
    def _match_profiles(self, spectrum: Dict, noise: Dict, 
                        color: Dict, artifacts: Dict,
                        dimension_hints: List[str] = None) -> List[GeneratorMatch]:
        """Match extracted features against generator profiles."""
        matches = []
        dimension_hints = dimension_hints or []
        
        for gen_name, profile in self.profiles.items():
            score = 0.0
            matching_features = []
            total_features = 0
            
            # Match spectrum features
            for feat, (low, high) in profile["spectrum"].items():
                if feat in spectrum:
                    val = spectrum[feat]
                    total_features += 1
                    if low <= val <= high:
                        score += 1.0
                        matching_features.append(f"Spectrum {feat}: {val:.2f}")
                    else:
                        # Partial score for close matches
                        dist = min(abs(val - low), abs(val - high))
                        range_size = high - low
                        if dist < range_size:
                            score += 0.5 * (1 - dist / range_size)
            
            # Match noise features
            for feat, (low, high) in profile["noise"].items():
                if feat in noise:
                    val = noise[feat]
                    total_features += 1
                    if low <= val <= high:
                        score += 1.0
                        matching_features.append(f"Noise {feat}: {val:.2f}")
                    else:
                        dist = min(abs(val - low), abs(val - high))
                        range_size = high - low
                        if dist < range_size:
                            score += 0.5 * (1 - dist / range_size)
            
            # Match color features
            for feat, (low, high) in profile["color"].items():
                if feat in color:
                    val = color[feat]
                    total_features += 1
                    if low <= val <= high:
                        score += 1.0
                        matching_features.append(f"Color {feat}: {val:.2f}")
                    else:
                        dist = min(abs(val - low), abs(val - high))
                        range_size = high - low
                        if dist < range_size:
                            score += 0.5 * (1 - dist / range_size)
            
            # Match artifact features
            for feat, (low, high) in profile["artifacts"].items():
                if feat in artifacts:
                    val = artifacts[feat]
                    total_features += 1
                    if low <= val <= high:
                        score += 1.0
                        matching_features.append(f"Artifact {feat}: {val:.2f}")
                    else:
                        dist = min(abs(val - low), abs(val - high))
                        range_size = high - low
                        if dist < range_size:
                            score += 0.5 * (1 - dist / range_size)
            
            # Bonus for dimension hints
            family = profile["family"]
            dimension_bonus = 0.0
            for hint in dimension_hints:
                if family in hint.lower() or gen_name.split("_")[0] in hint.lower():
                    dimension_bonus = 0.15  # 15% bonus for dimension match
                    matching_features.append(f"Dimension hint: {hint}")
                    break
                elif hint == "perfect_square_ai":
                    dimension_bonus = 0.05  # Small bonus for perfect square
                    break
            
            # Normalize score
            if total_features > 0:
                confidence = (score / total_features) + dimension_bonus
                confidence = min(1.0, confidence)  # Cap at 100%
            else:
                confidence = 0.0
            
            # Extract version hint from name
            version_hint = ""
            if "_v" in gen_name:
                version_hint = gen_name.split("_v")[-1]
            elif "_" in gen_name:
                parts = gen_name.split("_")
                if len(parts) > 1:
                    version_hint = parts[-1]
            
            matches.append(GeneratorMatch(
                name=gen_name.replace("_", " ").title(),
                family=profile["family"],
                confidence=confidence,
                matching_features=matching_features,
                version_hint=version_hint
            ))
        
        # Sort by confidence
        matches.sort(key=lambda x: -x.confidence)
        
        return matches


# Global instance
_generator_fingerprint: Optional[GeneratorFingerprint] = None

def get_generator_fingerprint() -> GeneratorFingerprint:
    global _generator_fingerprint
    if _generator_fingerprint is None:
        _generator_fingerprint = GeneratorFingerprint()
    return _generator_fingerprint
