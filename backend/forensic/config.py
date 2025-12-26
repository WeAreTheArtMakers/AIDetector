"""
Forensic Analysis Configuration
================================
Feature flags and configuration for forensic modules.
All new features are disabled by default for backward compatibility.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ForensicConfig:
    """Configuration with feature flags"""
    
    # Core features (always enabled)
    ENABLE_EXIF: bool = True
    ENABLE_GPS: bool = True
    ENABLE_AI_DETECTION: bool = True
    
    # Extended metadata (low cost)
    ENABLE_XMP_IPTC: bool = True
    ENABLE_SOFTWARE_HISTORY: bool = True
    
    # Format support
    ENABLE_HEIC: bool = True
    
    # JPEG Forensics (medium cost)
    ENABLE_JPEG_FORENSICS: bool = True
    ENABLE_DOUBLE_COMPRESSION: bool = True
    ENABLE_QUANT_TABLES: bool = True
    
    # Manipulation detection (medium cost for basic, high cost for deep scan)
    ENABLE_MANIPULATION_MODULES: bool = True  # Basic splice/blur detection
    ENABLE_COPY_MOVE: bool = False  # Deep scan only
    ENABLE_SPLICE_DETECTION: bool = True  # Basic noise analysis
    ENABLE_ELA: bool = False  # Deep scan only
    
    # Content signals (medium cost)
    ENABLE_OCR_SIGNALS: bool = False
    ENABLE_FACE_HAND_ANALYSIS: bool = False
    
    # Geometry/Lighting (high cost - Advanced)
    ENABLE_GEOMETRY_LIGHTING: bool = False
    
    # AI type classification
    ENABLE_AI_TYPE_CLASSIFIER: bool = True
    
    # Generation Pathway Classifier (T2I vs I2I vs Real)
    ENABLE_PATHWAY_CLASSIFIER: bool = True
    ENABLE_DIFFUSION_FINGERPRINT: bool = True
    
    # Content Type Gate (Photo vs Non-Photo/CGI)
    ENABLE_NON_PHOTO_GATE: bool = True
    ENABLE_CLIP_PROMPT_ENSEMBLE: bool = True
    NON_PHOTO_HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    NON_PHOTO_MARGIN_THRESHOLD: float = 0.10
    
    # Two-Axis Output (AI Likelihood + Evidential Quality)
    ENABLE_TWO_AXIS_OUTPUT: bool = True
    
    # Verdict Text Generator (UX text layer)
    ENABLE_VERDICT_TEXT: bool = True
    
    # Statistical features
    ENABLE_CALIBRATION: bool = True
    ENABLE_BENCHMARK_METRICS: bool = True
    ENABLE_UNCERTAINTY: bool = True
    
    # Model weights for ensemble
    MODEL_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "primary": 1.0,
        "secondary": 0.8,
        "clip_semantic": 0.6
    })
    
    # Uncertainty thresholds
    INCONCLUSIVE_BAND: tuple = (0.4, 0.6)
    DISAGREEMENT_THRESHOLD: float = 0.35
    
    # Domain penalties
    DOMAIN_PENALTIES: Dict[str, float] = field(default_factory=lambda: {
        "screenshot": 0.15,
        "recompressed": 0.10,
        "exif_stripped": 0.08,
        "social_media": 0.12
    })
    
    @classmethod
    def from_env(cls) -> 'ForensicConfig':
        """Load config from environment variables"""
        config = cls()
        for key in dir(config):
            if key.startswith('ENABLE_'):
                env_val = os.getenv(f'FORENSIC_{key}')
                if env_val is not None:
                    setattr(config, key, env_val.lower() in ('true', '1', 'yes'))
        return config

# Global config instance
CONFIG = ForensicConfig()

def get_config() -> ForensicConfig:
    return CONFIG
