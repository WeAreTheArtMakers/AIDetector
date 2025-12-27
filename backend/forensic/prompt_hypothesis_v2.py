"""
Prompt Hypothesis V2 - Content-Rich Visual Description
=======================================================
Generates detailed, content-rich prompt hypotheses from visual analysis.

CRITICAL PRINCIPLES:
- NEVER claim the original prompt can be recovered
- Output is "hypothesis from visual content", NOT "recovered prompt"
- Provide confidence tiers with calibrated thresholds
- Include "why" explanations for each inference
- Support both TR and EN with proper translations

Uses:
- Content understanding (scene, subjects, composition)
- Style analysis (lighting, aesthetic, camera look)
- Structured slot extraction
- Template-based prompt generation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Mandatory disclaimers - ALWAYS included
DISCLAIMER_TR = "Bu, görsel içerikten çıkarılan bir tahmindir. Orijinal prompt kesin olarak bilinemez."
DISCLAIMER_EN = "This is a hypothesis inferred from visual content. The original prompt cannot be known with certainty."

# Calibrated footer
FOOTER_TR = "Bu rapor, teknik bulgulara dayalı olasılıksal bir değerlendirmedir. 'Yüksek Güven' seviyeleri kalibre edilmiş eşiklerle üretilir."
FOOTER_EN = "This report is a probabilistic assessment based on technical findings. 'High confidence' uses calibrated thresholds."


@dataclass
class PromptHypothesisV2Result:
    """Result of prompt hypothesis generation"""
    enabled: bool = False
    confidence: str = "low"  # low, medium, high
    confidence_score: float = 0.0
    
    # Short prompts (1 line)
    en_short: str = ""
    tr_short: str = ""
    
    # Long prompts (2-4 lines)
    en_long: str = ""
    tr_long: str = ""
    
    # Structured slots
    slots: Dict[str, Any] = field(default_factory=dict)
    
    # Negative prompt suggestions
    negative_suggestions: List[str] = field(default_factory=list)
    
    # Why explanations
    why_reasons: List[str] = field(default_factory=list)
    
    # Disclaimers
    disclaimer_tr: str = DISCLAIMER_TR
    disclaimer_en: str = DISCLAIMER_EN


class PromptHypothesisV2Generator:
    """
    Generates content-rich prompt hypotheses.
    
    Enabled when:
    - generative_score >= 0.45 OR
    - pathway in {T2I, I2I} with any confidence OR
    - camera_score < 0.15 AND diffusion >= 0.25 OR
    - ai_probability >= 55 AND camera_score < 0.30
    
    Even if pAI is moderate, if generative signals are present, generate hypothesis.
    """
    
    def __init__(self):
        # Common negative prompts by category
        self.negative_prompts = {
            "quality": ["lowres", "jpeg artifacts", "blurry", "worst quality", "low quality"],
            "anatomy": ["deformed hands", "extra fingers", "mutated hands", "bad anatomy", 
                       "disfigured", "malformed limbs", "missing arms", "extra limbs"],
            "artifacts": ["text", "watermark", "logo", "signature", "username"],
            "style": ["cartoon", "anime", "drawing", "painting", "illustration"]
        }
        
        # Style templates
        self.style_templates = {
            "photorealistic": {
                "en": "photorealistic, highly detailed, professional photography, sharp focus",
                "tr": "fotogerçekçi, yüksek detay, profesyonel fotoğrafçılık, keskin odak"
            },
            "cinematic": {
                "en": "cinematic, movie still, dramatic lighting, film grain",
                "tr": "sinematik, film karesi, dramatik aydınlatma, film grenli"
            },
            "portrait": {
                "en": "portrait photography, shallow depth of field, bokeh background",
                "tr": "portre fotoğrafçılığı, sığ alan derinliği, bokeh arka plan"
            },
            "street": {
                "en": "street photography, candid, urban, natural moment",
                "tr": "sokak fotoğrafçılığı, doğal, şehir, anlık"
            },
            "editorial": {
                "en": "editorial photography, fashion, high-end, magazine quality",
                "tr": "editöryal fotoğrafçılık, moda, üst düzey, dergi kalitesi"
            }
        }
        
        # Camera/lens templates
        self.camera_templates = {
            "shallow": {
                "en": "85mm lens, f/1.8, shallow depth of field",
                "tr": "85mm lens, f/1.8, sığ alan derinliği"
            },
            "wide": {
                "en": "24mm wide angle, environmental portrait",
                "tr": "24mm geniş açı, çevresel portre"
            },
            "standard": {
                "en": "50mm lens, natural perspective",
                "tr": "50mm lens, doğal perspektif"
            },
            "telephoto": {
                "en": "135mm telephoto, compressed background",
                "tr": "135mm telefoto, sıkıştırılmış arka plan"
            }
        }
        
        # Lighting templates
        self.lighting_templates = {
            "natural": {
                "en": "natural daylight, soft shadows",
                "tr": "doğal gün ışığı, yumuşak gölgeler"
            },
            "dramatic": {
                "en": "dramatic lighting, high contrast, chiaroscuro",
                "tr": "dramatik aydınlatma, yüksek kontrast"
            },
            "soft": {
                "en": "soft diffused light, even illumination",
                "tr": "yumuşak dağınık ışık, eşit aydınlatma"
            },
            "golden_hour": {
                "en": "golden hour, warm sunlight, long shadows",
                "tr": "altın saat, sıcak güneş ışığı, uzun gölgeler"
            }
        }
    
    def generate(self,
                 image: Image.Image,
                 content_understanding: Dict[str, Any],
                 content_type: Dict[str, Any],
                 pathway: Dict[str, Any],
                 generative_score: float,
                 camera_score: float,
                 ai_probability: float,
                 diffusion_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate content-rich prompt hypothesis.
        
        Args:
            image: PIL Image
            content_understanding: Result from ContentUnderstanding module
            content_type: Content type classification
            pathway: Pathway classification result
            generative_score: Computed generative evidence score
            camera_score: Computed camera evidence score
            ai_probability: AI probability from ensemble
            diffusion_result: Diffusion fingerprint result
        
        Returns:
            Dict with hypothesis, confidence, slots, and explanations
        """
        result = PromptHypothesisV2Result()
        
        # Check enabling conditions
        # UPDATED: Enable when pathway=T2I (any confidence) OR camera_score very low with generative signals
        pathway_pred = pathway.get("pred", "unknown") if pathway else "unknown"
        pathway_conf = pathway.get("confidence", "low") if pathway else "low"
        diffusion_score = diffusion_result.get("diffusion_score", 0) if diffusion_result else 0
        
        should_enable = (
            generative_score >= 0.45 or  # Lowered threshold
            pathway_pred in ["t2i", "i2i"] or  # Any T2I/I2I pathway
            (camera_score < 0.15 and diffusion_score >= 0.25) or  # Low camera + some diffusion
            (ai_probability >= 55 and camera_score < 0.30)  # Lowered AI threshold
        )
        
        if not should_enable:
            result.why_reasons.append("Generative evidence insufficient for prompt hypothesis")
            return result.__dict__
        
        result.enabled = True
        
        try:
            # Get content slots
            slots = content_understanding.get("slots", {})
            result.slots = slots
            
            # Determine confidence tier
            confidence, conf_score, conf_reasons = self._compute_confidence(
                generative_score, camera_score, pathway_pred, pathway_conf,
                content_understanding.get("confidence", 0)
            )
            result.confidence = confidence
            result.confidence_score = conf_score
            result.why_reasons.extend(conf_reasons)
            
            # Build prompts
            en_short, en_long = self._build_english_prompts(slots, content_type)
            tr_short, tr_long = self._build_turkish_prompts(slots, content_type)
            
            result.en_short = en_short
            result.en_long = en_long
            result.tr_short = tr_short
            result.tr_long = tr_long
            
            # Generate negative suggestions
            result.negative_suggestions = self._generate_negatives(slots)
            
            logger.info(f"Prompt hypothesis V2: confidence={confidence}, "
                       f"generative_score={generative_score:.2f}")
            
        except Exception as e:
            logger.error(f"Prompt hypothesis V2 error: {e}")
            result.enabled = False
            result.why_reasons.append(f"Error: {str(e)}")
        
        return result.__dict__
    
    def _compute_confidence(self, generative_score: float, camera_score: float,
                            pathway_pred: str, pathway_conf: str,
                            content_conf: float) -> Tuple[str, float, List[str]]:
        """
        Compute confidence tier with calibrated thresholds.
        
        Returns:
            Tuple of (confidence_tier, confidence_score, reasons)
        """
        reasons = []
        score = 0.0
        
        # Generative score contribution
        if generative_score >= 0.80:
            score += 0.40
            reasons.append(f"Strong generative evidence ({generative_score:.0%})")
        elif generative_score >= 0.60:
            score += 0.25
            reasons.append(f"Moderate generative evidence ({generative_score:.0%})")
        elif generative_score >= 0.45:
            score += 0.10
            reasons.append(f"Weak generative evidence ({generative_score:.0%})")
        
        # Camera score penalty
        if camera_score >= 0.50:
            score -= 0.20
            reasons.append(f"Camera evidence present ({camera_score:.0%})")
        elif camera_score >= 0.30:
            score -= 0.10
            reasons.append(f"Some camera evidence ({camera_score:.0%})")
        
        # Pathway contribution
        if pathway_pred == "t2i":
            if pathway_conf == "high":
                score += 0.30
                reasons.append("T2I pathway (high confidence)")
            elif pathway_conf == "medium":
                score += 0.20
                reasons.append("T2I pathway (medium confidence)")
            else:
                score += 0.10
                reasons.append("T2I pathway (low confidence)")
        elif pathway_pred == "i2i":
            score += 0.15
            reasons.append("I2I pathway detected")
        
        # Content understanding contribution
        if content_conf >= 0.7:
            score += 0.10
            reasons.append("Good content extraction")
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        # Determine tier
        if score >= 0.65:
            tier = "high"
        elif score >= 0.40:
            tier = "medium"
        else:
            tier = "low"
        
        return tier, score, reasons
    
    def _build_english_prompts(self, slots: Dict, 
                                content_type: Dict) -> Tuple[str, str]:
        """Build English short and long prompts"""
        parts = []
        
        # Subject
        subject_types = slots.get("subject_types", [])
        if subject_types:
            if "woman" in subject_types:
                parts.append("young woman")
            elif "man" in subject_types:
                parts.append("young man")
            elif "person" in subject_types:
                parts.append("person")
        
        # Pose/action
        poses = slots.get("poses", [])
        if poses:
            parts.append(poses[0])
        
        # Environment
        environment = slots.get("environment", "")
        location = slots.get("location_type", "")
        if location:
            parts.append(f"in {location}")
        elif environment:
            parts.append(f"{environment} setting")
        
        # Clothing
        clothing = slots.get("clothing", [])
        if clothing:
            parts.append(", ".join(clothing[:2]))
        
        # Build short prompt
        short_parts = parts[:4]
        short_prompt = ", ".join(short_parts) if short_parts else "photorealistic image"
        
        # Add style for short
        style = slots.get("style", "photorealistic")
        if style and style != "photorealistic":
            short_prompt = f"{style} {short_prompt}"
        else:
            short_prompt = f"photorealistic {short_prompt}"
        
        # Build long prompt
        long_parts = parts.copy()
        
        # Add lighting
        lighting = slots.get("lighting_type", "")
        if lighting:
            template = self.lighting_templates.get(lighting, {})
            if template:
                long_parts.append(template.get("en", ""))
        
        # Add camera/composition
        dof = slots.get("depth_of_field", "")
        if dof:
            template = self.camera_templates.get(dof, {})
            if template:
                long_parts.append(template.get("en", ""))
        
        # Add style template
        style_template = self.style_templates.get(style, self.style_templates["photorealistic"])
        long_parts.append(style_template.get("en", ""))
        
        # Add aesthetic
        aesthetic = slots.get("aesthetic", "")
        if aesthetic:
            long_parts.append(aesthetic)
        
        long_prompt = ", ".join([p for p in long_parts if p])
        
        return short_prompt, long_prompt
    
    def _build_turkish_prompts(self, slots: Dict,
                                content_type: Dict) -> Tuple[str, str]:
        """Build Turkish short and long prompts"""
        parts = []
        
        # Subject translations
        subject_tr = {
            "woman": "genç kadın",
            "man": "genç erkek",
            "person": "kişi",
            "child": "çocuk"
        }
        
        subject_types = slots.get("subject_types", [])
        if subject_types:
            for st in subject_types:
                if st in subject_tr:
                    parts.append(subject_tr[st])
                    break
        
        # Environment translations
        env_tr = {
            "indoor": "iç mekan",
            "outdoor": "dış mekan",
            "studio": "stüdyo",
            "street": "sokak",
            "park": "park"
        }
        
        environment = slots.get("environment", "")
        location = slots.get("location_type", "")
        if location and location in env_tr:
            parts.append(f"{env_tr[location]} ortamında")
        elif environment and environment in env_tr:
            parts.append(f"{env_tr[environment]} ortamında")
        
        # Build short prompt
        short_parts = parts[:3]
        short_prompt = ", ".join(short_parts) if short_parts else "fotogerçekçi görsel"
        short_prompt = f"fotogerçekçi {short_prompt}"
        
        # Build long prompt
        long_parts = parts.copy()
        
        # Add lighting
        lighting = slots.get("lighting_type", "")
        if lighting:
            template = self.lighting_templates.get(lighting, {})
            if template:
                long_parts.append(template.get("tr", ""))
        
        # Add camera
        dof = slots.get("depth_of_field", "")
        if dof:
            template = self.camera_templates.get(dof, {})
            if template:
                long_parts.append(template.get("tr", ""))
        
        # Add style
        style = slots.get("style", "photorealistic")
        style_template = self.style_templates.get(style, self.style_templates["photorealistic"])
        long_parts.append(style_template.get("tr", ""))
        
        long_prompt = ", ".join([p for p in long_parts if p])
        
        return short_prompt, long_prompt
    
    def _generate_negatives(self, slots: Dict) -> List[str]:
        """Generate negative prompt suggestions"""
        negatives = []
        
        # Always include quality negatives
        negatives.extend(self.negative_prompts["quality"][:3])
        
        # Add anatomy negatives for portraits
        subject_types = slots.get("subject_types", [])
        if subject_types:
            negatives.extend(self.negative_prompts["anatomy"][:3])
        
        # Add artifact negatives
        negatives.extend(self.negative_prompts["artifacts"][:2])
        
        # Deduplicate and limit
        return list(dict.fromkeys(negatives))[:10]


# Global instance
_prompt_hypothesis_v2: Optional[PromptHypothesisV2Generator] = None

def get_prompt_hypothesis_v2() -> PromptHypothesisV2Generator:
    global _prompt_hypothesis_v2
    if _prompt_hypothesis_v2 is None:
        _prompt_hypothesis_v2 = PromptHypothesisV2Generator()
    return _prompt_hypothesis_v2
