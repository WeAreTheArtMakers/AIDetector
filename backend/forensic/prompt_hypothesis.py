"""
Prompt Hypothesis Module - Visual Content to Prompt Inference
==============================================================
Generates plausible prompt hypotheses from visual content analysis.

CRITICAL PRINCIPLES:
- NEVER claim the original prompt can be recovered with certainty
- Output must clearly label prompt as "hypothesis" with confidence and reasons
- Include mandatory disclaimer in all outputs
- This is inference from visual features, NOT prompt recovery
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# Mandatory disclaimers - ALWAYS included
DISCLAIMER_TR = "Not: Bu yalnızca görsel içerikten çıkarılan bir prompt tahminidir; orijinal prompt kesin olarak bilinemez."
DISCLAIMER_EN = "Note: This is only a prompt hypothesis inferred from visual content; the original prompt cannot be known with certainty."


def compute_generative_candidate(
    camera_evidence_score: float,
    diffusion_score: float,
    ai_probability: float,
    primary_model_score: float,
    model_disagreement: float,
    non_photo_score: float
) -> Tuple[bool, List[str]]:
    """
    Compute generative_candidate flag.
    
    generative_candidate = (
        camera_evidence_score < 0.20
        AND (
            diffusion_fingerprint_score >= 0.25
            OR pAI >= 0.55
            OR (primary_model_score >= 0.70 AND model_disagreement <= 0.45)
            OR (clip_semantic_nonphoto_prob >= 0.55)
        )
    )
    
    Returns:
        Tuple of (is_generative_candidate, reasons)
    """
    reasons = []
    
    # Must have low camera evidence
    if camera_evidence_score >= 0.20:
        return False, ["camera_evidence_too_high"]
    
    reasons.append("low_camera_evidence")
    
    # Check generative signals
    generative_signals = []
    
    if diffusion_score >= 0.25:
        generative_signals.append(f"diffusion_score_{diffusion_score:.2f}")
    
    if ai_probability >= 55:
        generative_signals.append(f"pAI_{ai_probability:.0f}")
    
    if primary_model_score >= 0.70 and model_disagreement <= 0.45:
        generative_signals.append(f"primary_high_{primary_model_score:.2f}_low_disagreement")
    
    if non_photo_score >= 0.55:
        generative_signals.append(f"non_photo_{non_photo_score:.2f}")
    
    if len(generative_signals) > 0:
        reasons.extend(generative_signals)
        return True, reasons
    
    return False, ["no_generative_signals"]


class PromptHypothesisGenerator:
    """
    Generates plausible prompt hypotheses from visual analysis.
    
    Uses:
    - Content type classification (CLIP features)
    - Detected visual attributes
    - Style/aesthetic analysis
    - Composition analysis
    
    NEVER claims certainty about original prompts.
    """
    
    def __init__(self):
        # Common negative prompts for AI generation
        self.common_negatives = [
            "deformed hands", "extra fingers", "mutated hands",
            "poorly drawn hands", "poorly drawn face", "mutation",
            "deformed", "ugly", "blurry", "bad anatomy",
            "bad proportions", "extra limbs", "cloned face",
            "disfigured", "gross proportions", "malformed limbs",
            "missing arms", "missing legs", "extra arms", "extra legs",
            "fused fingers", "too many fingers", "long neck",
            "text", "watermark", "logo", "signature",
            "lowres", "jpeg artifacts", "cropped", "worst quality"
        ]
        
        # Style keywords mapping
        self.style_keywords = {
            "photorealistic": ["photorealistic", "hyperrealistic", "photo", "realistic"],
            "cinematic": ["cinematic", "movie still", "film grain", "dramatic lighting"],
            "portrait": ["portrait", "headshot", "close-up", "face"],
            "fashion": ["fashion photography", "editorial", "vogue", "high fashion"],
            "street": ["street photography", "candid", "urban", "city"],
            "studio": ["studio lighting", "professional", "softbox", "beauty dish"],
            "natural": ["natural light", "golden hour", "soft light", "ambient"],
            "dramatic": ["dramatic", "moody", "dark", "contrast", "chiaroscuro"],
            "bokeh": ["bokeh", "shallow depth of field", "f/1.4", "f/1.8", "blurred background"],
            "sharp": ["sharp", "detailed", "high resolution", "8k", "4k"]
        }
        
        # Camera/lens style keywords
        self.camera_styles = {
            "portrait_lens": "85mm portrait lens, f/1.8",
            "wide_angle": "24mm wide angle, environmental",
            "telephoto": "200mm telephoto, compressed perspective",
            "macro": "macro lens, extreme close-up",
            "standard": "50mm standard lens"
        }
    
    def generate(self,
                 image: Image.Image,
                 content_type: Dict[str, Any],
                 pathway: Dict[str, Any],
                 ai_probability: float,
                 metadata: Dict[str, Any] = None,
                 diffusion_result: Dict[str, Any] = None,
                 model_scores: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate prompt hypothesis from visual analysis.
        
        NEW: Enabled when pathway in {T2I, I2I} with >= medium confidence
             OR generative_candidate == true
        
        Returns:
            Dict with hypothesis, confidence, attributes, and mandatory disclaimer
        """
        result = {
            "enabled": True,
            "hypothesis_tr": "",
            "hypothesis_en": "",
            "hypothesis_short_en": "",  # NEW: 1-sentence short prompt
            "hypothesis_short_tr": "",
            "confidence": "low",
            "reasons": [],
            "generative_candidate": False,  # NEW: flag for generative candidate
            "generative_candidate_reasons": [],
            "extracted_attributes": {
                "scene": "",
                "subject": "",
                "setting": "",
                "camera_style": "",
                "lighting": "",
                "composition": "",
                "aesthetic": "",
                "negative_prompts": [],
                "model_family_guess": "unknown",
                "aspect_ratio": "",
                # NEW: Extended attributes
                "people_count": 0,
                "age_groups": [],
                "emotions": [],
                "clothing_cues": [],
                "time_of_day": "",
                "style_template": ""
            },
            "disclaimer_tr": DISCLAIMER_TR,
            "disclaimer_en": DISCLAIMER_EN
        }
        
        # Get pathway info
        pathway_pred = pathway.get("pred", "unknown") if pathway else "unknown"
        pathway_conf = pathway.get("confidence", "low") if pathway else "low"
        
        # Get diffusion score from pathway evidence or diffusion_result
        diffusion_score = 0.0
        if diffusion_result:
            diffusion_score = diffusion_result.get("diffusion_score", 0)
        elif pathway:
            diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0)
        
        # Get camera evidence score
        camera_evidence_score = 0.0
        if pathway:
            camera_evidence_score = pathway.get("evidence", {}).get("camera_evidence_score", 0)
        
        # Get model scores for generative_candidate computation
        primary_model_score = 0.0
        model_disagreement = 0.0
        if model_scores:
            scores = [m.get("raw_score", 0) for m in model_scores if m.get("raw_score")]
            if scores:
                primary_model_score = max(scores)
                model_disagreement = max(scores) - min(scores) if len(scores) > 1 else 0
        
        # Get non-photo score
        non_photo_score = content_type.get("non_photo_score", 0.5) if content_type else 0.5
        
        # === NEW: Compute generative_candidate ===
        is_generative_candidate, gen_reasons = compute_generative_candidate(
            camera_evidence_score=camera_evidence_score,
            diffusion_score=diffusion_score,
            ai_probability=ai_probability,
            primary_model_score=primary_model_score,
            model_disagreement=model_disagreement,
            non_photo_score=non_photo_score
        )
        result["generative_candidate"] = is_generative_candidate
        result["generative_candidate_reasons"] = gen_reasons
        
        # === NEW ENABLING RULE ===
        # Enable if:
        # 1. pathway in {T2I, I2I} with >= medium confidence
        # 2. OR generative_candidate == true
        is_pathway_suitable = (
            pathway_pred in ["t2i", "i2i"] and 
            pathway_conf in ["high", "medium"]
        )
        
        should_enable = is_pathway_suitable or is_generative_candidate
        
        if not should_enable:
            result["enabled"] = False
            result["reasons"].append("Neither suitable pathway nor generative_candidate")
            return result
        
        try:
            # Extract visual attributes
            attributes = self._extract_attributes(image, content_type, pathway, metadata)
            result["extracted_attributes"] = attributes
            
            # Build hypothesis prompts (detailed + short)
            hypothesis_en, hypothesis_tr = self._build_hypothesis(attributes)
            result["hypothesis_en"] = hypothesis_en
            result["hypothesis_tr"] = hypothesis_tr
            
            # Build short prompts (1 sentence)
            short_en, short_tr = self._build_short_hypothesis(attributes)
            result["hypothesis_short_en"] = short_en
            result["hypothesis_short_tr"] = short_tr
            
            # Determine confidence with NEW rules
            confidence, reasons = self._compute_confidence_v2(
                attributes, content_type, pathway, 
                diffusion_score, is_generative_candidate,
                ai_probability
            )
            result["confidence"] = confidence
            result["reasons"] = reasons
            
            # Add recommended negative prompts
            result["extracted_attributes"]["negative_prompts"] = self._suggest_negatives(attributes)
            
            logger.info(f"Prompt hypothesis generated: confidence={confidence}, generative_candidate={is_generative_candidate}")
            
        except Exception as e:
            logger.error(f"Prompt hypothesis error: {e}")
            result["enabled"] = False
            result["reasons"].append(f"Error: {str(e)}")
        
        return result
    
    def _extract_attributes(self,
                            image: Image.Image,
                            content_type: Dict[str, Any],
                            pathway: Dict[str, Any],
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual attributes from image analysis"""
        attributes = {
            "scene": "",
            "subject": "",
            "setting": "",
            "camera_style": "",
            "lighting": "",
            "composition": "",
            "aesthetic": "",
            "negative_prompts": [],
            "model_family_guess": "unknown",
            "aspect_ratio": ""
        }
        
        # Compute aspect ratio
        w, h = image.size
        ratio = w / h
        if abs(ratio - 1.0) < 0.05:
            attributes["aspect_ratio"] = "1:1 (square)"
        elif abs(ratio - 0.667) < 0.05:
            attributes["aspect_ratio"] = "2:3 (portrait)"
        elif abs(ratio - 1.5) < 0.05:
            attributes["aspect_ratio"] = "3:2 (landscape)"
        elif abs(ratio - 0.5625) < 0.05:
            attributes["aspect_ratio"] = "9:16 (vertical)"
        elif abs(ratio - 1.778) < 0.05:
            attributes["aspect_ratio"] = "16:9 (widescreen)"
        else:
            attributes["aspect_ratio"] = f"{w}:{h}"
        
        # Extract from content type top matches
        top_matches = content_type.get("top_matches", {})
        
        # Analyze photo matches for subject/scene hints
        photo_matches = top_matches.get("photo", [])
        non_photo_matches = top_matches.get("non_photo", [])
        ai_photoreal_matches = top_matches.get("ai_photoreal", [])
        
        # Determine subject type
        subject_hints = []
        for match in photo_matches + ai_photoreal_matches:
            prompt = match.get("prompt", "").lower()
            if "portrait" in prompt or "person" in prompt or "face" in prompt:
                subject_hints.append("portrait")
            if "woman" in prompt or "female" in prompt:
                subject_hints.append("female")
            if "man" in prompt or "male" in prompt:
                subject_hints.append("male")
            if "landscape" in prompt:
                subject_hints.append("landscape")
            if "street" in prompt or "urban" in prompt:
                subject_hints.append("street")
        
        if "portrait" in subject_hints:
            if "female" in subject_hints:
                attributes["subject"] = "female portrait"
            elif "male" in subject_hints:
                attributes["subject"] = "male portrait"
            else:
                attributes["subject"] = "portrait"
            attributes["composition"] = "close-up or medium shot"
        elif "landscape" in subject_hints:
            attributes["subject"] = "landscape/scene"
            attributes["composition"] = "wide shot"
        elif "street" in subject_hints:
            attributes["subject"] = "street scene"
            attributes["setting"] = "urban environment"
        
        # Analyze image characteristics for lighting/style
        img_array = np.array(image.convert('RGB'))
        
        # Brightness analysis
        brightness = np.mean(img_array)
        if brightness > 180:
            attributes["lighting"] = "bright, high-key lighting"
        elif brightness < 80:
            attributes["lighting"] = "dark, low-key, moody lighting"
        else:
            attributes["lighting"] = "balanced, natural lighting"
        
        # Contrast analysis
        contrast = np.std(img_array)
        if contrast > 70:
            attributes["aesthetic"] = "high contrast, dramatic"
        elif contrast < 40:
            attributes["aesthetic"] = "soft, low contrast, dreamy"
        else:
            attributes["aesthetic"] = "balanced contrast"
        
        # Color saturation analysis
        hsv = self._rgb_to_hsv_mean(img_array)
        saturation = hsv[1]
        if saturation > 0.6:
            attributes["aesthetic"] += ", vibrant colors"
        elif saturation < 0.3:
            attributes["aesthetic"] += ", muted/desaturated colors"
        
        # Determine camera style based on composition
        if attributes["subject"] == "portrait" or "portrait" in attributes["subject"]:
            attributes["camera_style"] = "85mm portrait lens, shallow depth of field, f/1.8"
        elif "landscape" in attributes["subject"]:
            attributes["camera_style"] = "wide angle lens, deep depth of field, f/8"
        elif "street" in attributes["subject"]:
            attributes["camera_style"] = "35mm street photography style"
        else:
            attributes["camera_style"] = "standard lens, natural perspective"
        
        # Determine model family from pathway
        generator_family = pathway.get("generator_family", {}).get("pred", "unknown")
        if generator_family in ["diffusion_t2i_modern", "vendor_photoreal_style"]:
            attributes["model_family_guess"] = "diffusion_like"
        elif generator_family == "gan_legacy":
            attributes["model_family_guess"] = "gan_like"
        else:
            attributes["model_family_guess"] = "unknown"
        
        # Setting inference
        if not attributes["setting"]:
            if "outdoor" in str(photo_matches).lower():
                attributes["setting"] = "outdoor"
            elif "indoor" in str(photo_matches).lower() or "studio" in str(photo_matches).lower():
                attributes["setting"] = "indoor/studio"
            else:
                attributes["setting"] = "unspecified"
        
        # Scene description
        if attributes["subject"] and attributes["setting"]:
            attributes["scene"] = f"{attributes['subject']} in {attributes['setting']}"
        elif attributes["subject"]:
            attributes["scene"] = attributes["subject"]
        
        return attributes
    
    def _rgb_to_hsv_mean(self, img_array: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean HSV values from RGB array"""
        # Normalize to 0-1
        img_norm = img_array.astype(np.float32) / 255.0
        
        r, g, b = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Value
        v = np.mean(max_c)
        
        # Saturation
        s = np.mean(np.where(max_c > 0, diff / (max_c + 1e-6), 0))
        
        # Hue (simplified)
        h = 0.0  # Not computing full hue for simplicity
        
        return (h, s, v)
    
    def _build_hypothesis(self, attributes: Dict[str, Any]) -> Tuple[str, str]:
        """Build hypothesis prompt strings in EN and TR"""
        
        # Build English hypothesis
        parts_en = []
        
        if attributes["subject"]:
            parts_en.append(attributes["subject"])
        
        if attributes["setting"] and attributes["setting"] != "unspecified":
            parts_en.append(f"in {attributes['setting']}")
        
        if attributes["lighting"]:
            parts_en.append(attributes["lighting"])
        
        if attributes["camera_style"]:
            parts_en.append(attributes["camera_style"])
        
        if attributes["aesthetic"]:
            parts_en.append(attributes["aesthetic"])
        
        # Add quality keywords
        parts_en.append("photorealistic, highly detailed, professional photography")
        
        hypothesis_en = ", ".join(parts_en)
        
        # Build Turkish hypothesis
        parts_tr = []
        
        subject_tr_map = {
            "female portrait": "kadın portresi",
            "male portrait": "erkek portresi",
            "portrait": "portre",
            "landscape/scene": "manzara/sahne",
            "street scene": "sokak sahnesi"
        }
        
        if attributes["subject"]:
            subject_tr = subject_tr_map.get(attributes["subject"], attributes["subject"])
            parts_tr.append(subject_tr)
        
        setting_tr_map = {
            "outdoor": "dış mekan",
            "indoor/studio": "iç mekan/stüdyo",
            "urban environment": "şehir ortamı"
        }
        
        if attributes["setting"] and attributes["setting"] != "unspecified":
            setting_tr = setting_tr_map.get(attributes["setting"], attributes["setting"])
            parts_tr.append(f"{setting_tr} ortamında")
        
        if attributes["lighting"]:
            lighting_tr_map = {
                "bright, high-key lighting": "parlak, yüksek anahtar aydınlatma",
                "dark, low-key, moody lighting": "karanlık, düşük anahtar, dramatik aydınlatma",
                "balanced, natural lighting": "dengeli, doğal aydınlatma"
            }
            lighting_tr = lighting_tr_map.get(attributes["lighting"], attributes["lighting"])
            parts_tr.append(lighting_tr)
        
        parts_tr.append("fotogerçekçi, yüksek detay, profesyonel fotoğrafçılık")
        
        hypothesis_tr = ", ".join(parts_tr)
        
        return hypothesis_en, hypothesis_tr
    
    def _compute_confidence(self,
                            attributes: Dict[str, Any],
                            content_type: Dict[str, Any],
                            pathway: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Compute confidence level for hypothesis (legacy method)"""
        return self._compute_confidence_v2(attributes, content_type, pathway, 0, False, 50)
    
    def _compute_confidence_v2(self,
                               attributes: Dict[str, Any],
                               content_type: Dict[str, Any],
                               pathway: Dict[str, Any],
                               diffusion_score: float,
                               is_generative_candidate: bool,
                               ai_probability: float) -> Tuple[str, List[str]]:
        """
        Compute confidence level for hypothesis with NEW rules.
        
        - high: pathway == T2I AND diffusion_fingerprint_score >= 0.55 AND evidence_quality >= 40
        - medium: generative_candidate true
        - low: evidence_quality < 20 but still generative_candidate (keep but warn)
        """
        reasons = []
        
        # Get pathway info
        pathway_pred = pathway.get("pred", "unknown") if pathway else "unknown"
        pathway_conf = pathway.get("confidence", "low") if pathway else "low"
        
        # Check if attributes are clear
        if attributes.get("subject"):
            reasons.append("Subject type identified")
        
        if attributes.get("setting") and attributes["setting"] != "unspecified":
            reasons.append("Setting identified")
        
        if attributes.get("lighting"):
            reasons.append("Lighting style analyzed")
        
        # HIGH confidence: T2I pathway + high diffusion + good evidence
        if pathway_pred == "t2i" and diffusion_score >= 0.55:
            # Check evidence quality (simplified)
            evidence_quality = 50  # Default
            if pathway_conf == "high":
                evidence_quality = 70
            elif pathway_conf == "medium":
                evidence_quality = 50
            else:
                evidence_quality = 30
            
            if evidence_quality >= 40:
                reasons.append("T2I pathway with strong diffusion fingerprint")
                return "high", reasons
        
        # MEDIUM confidence: generative_candidate true
        if is_generative_candidate:
            reasons.append("Generative candidate detected")
            return "medium", reasons
        
        # LOW confidence: fallback
        reasons.append("Hypothesis based on visual analysis only")
        return "low", reasons
    
    def _build_short_hypothesis(self, attributes: Dict[str, Any]) -> Tuple[str, str]:
        """Build short 1-sentence hypothesis prompts"""
        
        # Build English short hypothesis
        parts = []
        
        if attributes.get("subject"):
            parts.append(attributes["subject"])
        
        if attributes.get("setting") and attributes["setting"] != "unspecified":
            parts.append(f"in {attributes['setting']}")
        
        if attributes.get("lighting"):
            # Simplify lighting for short version
            lighting = attributes["lighting"]
            if "bright" in lighting.lower():
                parts.append("bright lighting")
            elif "dark" in lighting.lower() or "moody" in lighting.lower():
                parts.append("moody lighting")
            else:
                parts.append("natural lighting")
        
        parts.append("photorealistic")
        
        short_en = ", ".join(parts)
        
        # Build Turkish short hypothesis
        subject_tr_map = {
            "female portrait": "kadın portresi",
            "male portrait": "erkek portresi",
            "portrait": "portre",
            "landscape/scene": "manzara",
            "street scene": "sokak sahnesi"
        }
        
        parts_tr = []
        if attributes.get("subject"):
            subject_tr = subject_tr_map.get(attributes["subject"], attributes["subject"])
            parts_tr.append(subject_tr)
        
        parts_tr.append("fotogerçekçi")
        
        short_tr = ", ".join(parts_tr)
        
        return short_en, short_tr
    
    def _suggest_negatives(self, attributes: Dict[str, Any]) -> List[str]:
        """Suggest common negative prompts"""
        negatives = []
        
        # Always include common quality negatives
        negatives.extend([
            "lowres", "jpeg artifacts", "blurry", "worst quality"
        ])
        
        # Add anatomy negatives for portraits
        if "portrait" in attributes.get("subject", ""):
            negatives.extend([
                "deformed hands", "extra fingers", "mutated hands",
                "bad anatomy", "disfigured"
            ])
        
        # Add text/watermark negatives
        negatives.extend(["text", "watermark", "logo", "signature"])
        
        return negatives[:10]  # Limit to 10


# Global instance
_prompt_hypothesis_generator: Optional[PromptHypothesisGenerator] = None

def get_prompt_hypothesis_generator() -> PromptHypothesisGenerator:
    global _prompt_hypothesis_generator
    if _prompt_hypothesis_generator is None:
        _prompt_hypothesis_generator = PromptHypothesisGenerator()
    return _prompt_hypothesis_generator
