"""
Forensic Analysis Configuration
================================
Feature flags and configuration for forensic modules.
All new features are disabled by default for backward compatibility.

Threshold naming convention:
- T_* = Threshold for classification decisions
- W_* = Weight for evidence fusion
- PROV_* = Provenance-related thresholds
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
    
    # Content Understanding (for prompt hypothesis)
    ENABLE_CONTENT_UNDERSTANDING: bool = True
    
    # Prompt Hypothesis V2 (content-rich)
    ENABLE_PROMPT_HYPOTHESIS_V2: bool = True
    
    # Prompt Recovery (metadata-based exact prompt extraction)
    ENABLE_PROMPT_RECOVERY: bool = True
    
    # BLIP Captioning (optional, requires transformers)
    ENABLE_BLIP_CAPTIONING: bool = True
    BLIP_MODEL_NAME: str = "Salesforce/blip-image-captioning-base"
    BLIP_MAX_LENGTH: int = 75
    BLIP_MIN_LENGTH: int = 10
    
    # RAM Tagging (optional, requires recognize-anything)
    ENABLE_RAM_TAGGING: bool = False  # Disabled by default (heavy dependency)
    RAM_MODEL_NAME: str = "ram_swin_large_14m"
    
    # Text Forensics (OCR + AI text artifact detection)
    ENABLE_TEXT_FORENSICS: bool = True
    
    # Generator Fingerprinting (AI tool identification)
    ENABLE_GENERATOR_FINGERPRINT: bool = True
    
    # Generative Artifact Heatmap
    ENABLE_GENERATIVE_HEATMAP: bool = True
    
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
    
    # =========================================================================
    # EVIDENCE FUSION THRESHOLDS (for final_label decision)
    # =========================================================================
    
    # Composite detection thresholds
    T_COMP: float = 0.60          # Composite score threshold
    T_LOCAL: float = 0.60         # Local manipulation score for composite
    T_LOCAL_SOFT: float = 0.40    # Soft threshold for local manipulation
    
    # Photo detection thresholds
    T_PHOTO: float = 0.55         # Photo score threshold
    T_AI_LOW: float = 0.45        # pAI must be below this for PHOTO verdict
    
    # AI detection thresholds
    T_AI_HIGH: float = 0.60       # pAI must be above this for AI verdict
    T_AI_MEDIUM: float = 0.50     # Medium AI threshold
    
    # Global edit threshold
    T_GLOBAL_MIN: float = 0.35    # Minimum for global edit detection
    
    # Provenance thresholds
    T_PROV_OK: float = 0.30       # Provenance score for "OK" status
    T_PROV_STRONG: float = 0.45   # Strong provenance (blocks AI verdict)
    PROV_FLOOR_GPS: float = 0.25  # Minimum provenance if GPS present
    
    # Camera evidence thresholds
    T_CAM_OK: float = 0.20        # Camera evidence for "OK" status
    T_CAM_LOW: float = 0.10       # Very low camera evidence
    T_CAM_HIGH: float = 0.35      # Strong camera evidence
    
    # Pathway/Diffusion thresholds
    T_T2I_HIGH: float = 0.70      # T2I pathway high confidence
    T_T2I_MEDIUM: float = 0.50    # T2I pathway medium confidence
    T_DIFF_OK: float = 0.30       # Diffusion fingerprint threshold
    T_DIFF_HIGH: float = 0.55     # High diffusion fingerprint
    
    # Confidence interval thresholds
    T_CI_NARROW: float = 0.25     # Narrow CI for high confidence
    T_CI_WIDE: float = 0.40       # Wide CI triggers uncertainty
    
    # =========================================================================
    # EVIDENCE FUSION WEIGHTS (for posterior computation)
    # =========================================================================
    
    W_GENERATIVE: float = 2.2     # Weight for generative evidence
    W_CAMERA: float = 2.0         # Weight for camera pipeline evidence
    W_PROVENANCE: float = 1.6     # Weight for provenance (EXIF/GPS)
    W_DOMAIN: float = 0.8         # Weight for domain penalty
    W_DISAGREEMENT: float = 0.9   # Weight for model disagreement
    W_LOCAL_TAMPER: float = 2.0   # Weight for local manipulation
    W_BOUNDARY: float = 1.2       # Weight for boundary corroboration
    W_AI_PROB: float = 1.5        # Weight for AI probability
    
    # =========================================================================
    # PROMPT HYPOTHESIS THRESHOLDS
    # =========================================================================
    
    # Enabling thresholds
    T_PROMPT_GENERATIVE: float = 0.45    # Generative score to enable prompt hypothesis
    T_PROMPT_CAMERA_LOW: float = 0.15    # Camera score below this enables prompt
    T_PROMPT_DIFFUSION: float = 0.25     # Diffusion score to enable prompt
    T_PROMPT_AI_PROB: float = 0.55       # AI probability to enable prompt
    
    # Confidence thresholds
    T_PROMPT_CONF_HIGH: float = 0.65     # High confidence prompt hypothesis
    T_PROMPT_CONF_MEDIUM: float = 0.40   # Medium confidence prompt hypothesis
    
    # Caption-tag agreement threshold
    T_CAPTION_TAG_AGREEMENT: float = 0.50  # Minimum overlap for good agreement
    
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
