"""
Prompt Analysis Module - Unified Prompt Recovery & Reconstruction
==================================================================
Main entry point for all prompt-related analysis.

Combines:
1. Prompt Recovery (metadata-based, exact)
2. Content Extraction (BLIP captioning, tagging)
3. Prompt Compilation (scene graph to prompt)

Output clearly distinguishes:
- RECOVERED: Exact prompt from metadata (high trust)
- RECONSTRUCTED: Inferred from visual content (hypothesis)

CRITICAL PRINCIPLES:
- Never claim certainty about reconstructed prompts
- Always include disclaimers
- Clearly label trust levels
- Support both TR and EN
"""

import logging
from typing import Dict, Any, Optional, List
from PIL import Image
from dataclasses import dataclass, field
from enum import Enum

from forensic.config import get_config
from forensic.prompt_recovery import get_prompt_recovery, RecoveryStatus, RecoveredPrompt
from forensic.content_extraction import get_content_extractor, ContentExtractionResult
from forensic.prompt_compiler import get_prompt_compiler, CompiledPrompt

logger = logging.getLogger(__name__)


class PromptSource(str, Enum):
    """Source of prompt data"""
    RECOVERED_SIGNED = "RECOVERED_SIGNED"      # From C2PA/signed (highest trust)
    RECOVERED_UNSIGNED = "RECOVERED_UNSIGNED"  # From metadata (medium trust)
    RECONSTRUCTED_HIGH = "RECONSTRUCTED_HIGH"  # Inferred, high confidence
    RECONSTRUCTED_MEDIUM = "RECONSTRUCTED_MEDIUM"  # Inferred, medium confidence
    RECONSTRUCTED_LOW = "RECONSTRUCTED_LOW"    # Inferred, low confidence
    NOT_AVAILABLE = "NOT_AVAILABLE"            # Could not determine


@dataclass
class PromptAnalysisResult:
    """Complete prompt analysis result"""
    # Status
    enabled: bool = True
    source: PromptSource = PromptSource.NOT_AVAILABLE
    
    # Recovered prompt (if found in metadata)
    recovered: Optional[RecoveredPrompt] = None
    has_recovered: bool = False
    
    # Reconstructed prompt (inferred from content)
    reconstructed: Optional[CompiledPrompt] = None
    has_reconstructed: bool = False
    
    # Content extraction details
    content_extraction: Optional[ContentExtractionResult] = None
    
    # Final prompt to display (recovered if available, else reconstructed)
    final_prompt_en: str = ""
    final_prompt_tr: str = ""
    final_negative_en: str = ""
    final_negative_tr: str = ""
    
    # Trust level
    trust_level: str = "none"  # high, medium, low, none
    
    # Generator info (if detected)
    generator: str = ""
    generator_version: str = ""
    
    # Parameters (if recovered)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Disclaimers (always present for reconstructed)
    disclaimer_tr: str = ""
    disclaimer_en: str = ""
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Disclaimers
DISCLAIMER_RECOVERED_TR = "Bu prompt dosya metadata'sından çıkarılmıştır. Metadata değiştirilebilir olduğundan kesin doğruluk garanti edilemez."
DISCLAIMER_RECOVERED_EN = "This prompt was extracted from file metadata. Since metadata can be modified, absolute accuracy cannot be guaranteed."

DISCLAIMER_RECOVERED_SIGNED_TR = "Bu prompt imzalı provenance'dan (C2PA) çıkarılmıştır. Yüksek güvenilirlik."
DISCLAIMER_RECOVERED_SIGNED_EN = "This prompt was extracted from signed provenance (C2PA). High reliability."

DISCLAIMER_RECONSTRUCTED_TR = "Bu prompt görsel içerikten TAHMİN edilmiştir. Orijinal prompt kesin olarak bilinemez."
DISCLAIMER_RECONSTRUCTED_EN = "This prompt was INFERRED from visual content. The original prompt cannot be known with certainty."


class PromptAnalyzer:
    """
    Main prompt analysis orchestrator.
    
    Pipeline:
    1. Try to recover exact prompt from metadata
    2. If not found, extract content and reconstruct
    3. Combine results with clear labeling
    """
    
    def __init__(self):
        self.config = get_config()
        self.recovery = get_prompt_recovery()
        self.extractor = get_content_extractor()
        self.compiler = get_prompt_compiler()
    
    def analyze(self,
                image: Image.Image,
                image_bytes: bytes,
                metadata: Dict[str, Any],
                pathway: Dict[str, Any] = None,
                content_type: Dict[str, Any] = None,
                generative_score: float = 0.0,
                camera_score: float = 0.0,
                ai_probability: float = 50.0) -> PromptAnalysisResult:
        """
        Perform complete prompt analysis.
        
        Args:
            image: PIL Image
            image_bytes: Raw image bytes
            metadata: Extracted metadata
            pathway: Pathway classification result
            content_type: Content type classification
            generative_score: Computed generative evidence score
            camera_score: Computed camera evidence score
            ai_probability: AI probability from ensemble
        
        Returns:
            PromptAnalysisResult with recovered and/or reconstructed prompts
        """
        result = PromptAnalysisResult()
        
        # === STEP 1: Try to recover exact prompt from metadata ===
        if self.config.ENABLE_PROMPT_RECOVERY:
            logger.info("Attempting prompt recovery from metadata...")
            recovered = self.recovery.recover(image_bytes, metadata)
            
            if recovered.status != RecoveryStatus.NOT_FOUND:
                result.recovered = recovered
                result.has_recovered = True
                result.generator = recovered.generator
                result.generator_version = recovered.generator_version
                result.parameters = recovered.parameters
                
                # Set source and trust level
                if recovered.status == RecoveryStatus.RECOVERED_SIGNED:
                    result.source = PromptSource.RECOVERED_SIGNED
                    result.trust_level = "high"
                    result.disclaimer_tr = DISCLAIMER_RECOVERED_SIGNED_TR
                    result.disclaimer_en = DISCLAIMER_RECOVERED_SIGNED_EN
                else:
                    result.source = PromptSource.RECOVERED_UNSIGNED
                    result.trust_level = "medium"
                    result.disclaimer_tr = DISCLAIMER_RECOVERED_TR
                    result.disclaimer_en = DISCLAIMER_RECOVERED_EN
                
                # Set final prompts from recovered
                result.final_prompt_en = recovered.prompt
                result.final_prompt_tr = recovered.prompt  # Same as EN for recovered
                result.final_negative_en = recovered.negative_prompt
                result.final_negative_tr = recovered.negative_prompt
                
                result.notes.append(f"Prompt recovered from {recovered.source}")
                logger.info(f"Prompt recovered: {len(recovered.prompt)} chars from {recovered.source}")
        
        # === STEP 2: Check if reconstruction should be attempted ===
        should_reconstruct = self._should_reconstruct(
            result, pathway, generative_score, camera_score, ai_probability
        )
        
        if should_reconstruct:
            logger.info("Attempting prompt reconstruction from visual content...")
            
            # Extract content
            clip_features = content_type.get("clip_features", {}) if content_type else {}
            content_result = self.extractor.extract(image, clip_features)
            result.content_extraction = content_result
            
            # Get style hints from analysis
            style_hints = self._build_style_hints(content_type, pathway)
            
            # Compile prompt
            compiled = self.compiler.compile(
                content_result.scene_graph,
                content_result,
                style_hints
            )
            
            result.reconstructed = compiled
            result.has_reconstructed = True
            
            # Set source based on confidence
            if compiled.confidence == "high":
                result.source = PromptSource.RECONSTRUCTED_HIGH
            elif compiled.confidence == "medium":
                result.source = PromptSource.RECONSTRUCTED_MEDIUM
            else:
                result.source = PromptSource.RECONSTRUCTED_LOW
            
            # Set trust level (always lower than recovered)
            result.trust_level = "low"
            result.disclaimer_tr = DISCLAIMER_RECONSTRUCTED_TR
            result.disclaimer_en = DISCLAIMER_RECONSTRUCTED_EN
            
            # Set final prompts from reconstructed (if no recovered)
            if not result.has_recovered:
                result.final_prompt_en = compiled.long_en
                result.final_prompt_tr = compiled.long_tr
                result.final_negative_en = compiled.negative_en
                result.final_negative_tr = compiled.negative_tr
            
            result.notes.extend(compiled.notes)
            logger.info(f"Prompt reconstructed: confidence={compiled.confidence}")
        
        # === STEP 3: Handle case where neither is available ===
        if not result.has_recovered and not result.has_reconstructed:
            result.source = PromptSource.NOT_AVAILABLE
            result.trust_level = "none"
            result.warnings.append("Could not recover or reconstruct prompt")
        
        return result
    
    def _should_reconstruct(self,
                            result: PromptAnalysisResult,
                            pathway: Dict,
                            generative_score: float,
                            camera_score: float,
                            ai_probability: float) -> bool:
        """
        Determine if prompt reconstruction should be attempted.
        
        Rules:
        - Always reconstruct if no recovered prompt
        - Reconstruct if generative signals are present
        - Don't reconstruct if strong camera evidence (real photo)
        """
        config = self.config
        
        # If we have recovered prompt, still reconstruct for comparison
        # but with lower priority
        
        # Check pathway
        pathway_pred = pathway.get("pred", "unknown") if pathway else "unknown"
        pathway_conf = pathway.get("confidence", "low") if pathway else "low"
        
        # Reconstruct if:
        # 1. Pathway is T2I or I2I
        if pathway_pred in ["t2i", "i2i"]:
            return True
        
        # 2. Generative score is above threshold
        if generative_score >= config.T_PROMPT_GENERATIVE:
            return True
        
        # 3. Camera score is very low with some AI signals
        if camera_score < config.T_PROMPT_CAMERA_LOW:
            diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0) if pathway else 0
            if diffusion_score >= config.T_PROMPT_DIFFUSION:
                return True
            if ai_probability >= config.T_PROMPT_AI_PROB:
                return True
        
        # 4. AI probability is high
        if ai_probability >= 60 and camera_score < 0.30:
            return True
        
        # 5. No recovered prompt - try reconstruction anyway
        if not result.has_recovered:
            # But only if there's some indication it might be AI
            if ai_probability >= 40 or generative_score >= 0.30:
                return True
        
        return False
    
    def _build_style_hints(self, content_type: Dict, pathway: Dict) -> Dict[str, Any]:
        """Build style hints from analysis results"""
        hints = {}
        
        if content_type:
            # Get style from content type
            ct_type = content_type.get("type", "")
            if ct_type == "non_photo_like":
                hints["style"] = "digital art"
            elif ct_type == "photo_like_generated":
                hints["style"] = "photorealistic"
            
            # Get top matches for style hints
            top_matches = content_type.get("top_matches", {})
            if top_matches.get("ai_photoreal"):
                hints["style"] = "photorealistic"
        
        if pathway:
            # Get generator family
            gen_family = pathway.get("generator_family", {}).get("pred", "")
            if gen_family == "vendor_photoreal_style":
                hints["style"] = "photorealistic"
                hints["mood"] = "professional"
        
        return hints


# Global instance
_prompt_analyzer: Optional[PromptAnalyzer] = None

def get_prompt_analyzer() -> PromptAnalyzer:
    global _prompt_analyzer
    if _prompt_analyzer is None:
        _prompt_analyzer = PromptAnalyzer()
    return _prompt_analyzer


# Convenience function for backward compatibility
def analyze_prompt(image: Image.Image,
                   image_bytes: bytes,
                   metadata: Dict[str, Any],
                   **kwargs) -> PromptAnalysisResult:
    """Convenience function for prompt analysis"""
    analyzer = get_prompt_analyzer()
    return analyzer.analyze(image, image_bytes, metadata, **kwargs)
