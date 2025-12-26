"""
Content Type Classifier - Photo vs Non-Photo (CGI/Illustration) Gate
=====================================================================
Uses CLIP prompt ensemble to classify content type before AI detection.
This helps avoid "AI vs real" ambiguity for CGI/illustration content.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Prompt lists for CLIP ensemble
PHOTO_POSITIVE_PROMPTS = [
    "a real photograph",
    "a smartphone photo",
    "a DSLR photo",
    "a candid photo",
    "a photo taken with a camera",
    "a documentary photo",
    "a natural photo with realistic skin texture",
    "a portrait photo with real camera bokeh",
    "a photojournalism image",
    "a raw photo"
]

NON_PHOTO_POSITIVE_PROMPTS = [
    "a 3D render",
    "CGI",
    "a computer-generated image",
    "a digital character render",
    "a realistic 3D character",
    "a video game character render",
    "a cinematic render",
    "a digital illustration",
    "concept art",
    "anime style illustration",
    "cartoon illustration",
    "a stylized digital portrait",
    "a synthetic image"
]

# NEW: Photoreal AI prompts (for detecting AI-generated photorealistic images)
PHOTOREAL_AI_PROMPTS = [
    "a photorealistic AI-generated portrait",
    "a synthetic fashion portrait with bokeh",
    "a studio-like AI portrait",
    "a popular AI generator photoreal image",
    "an AI-generated headshot",
    "a diffusion model generated photo",
    "a hyperrealistic AI portrait",
    "a synthetic person photo",
    "an AI-generated professional portrait",
    "a machine learning generated face"
]

STYLE_TRANSFER_PROMPTS = [
    "a photo with a heavy filter",
    "style transfer",
    "a filtered photo",
    "beauty retouching",
    "skin smoothing filter",
    "instagram filter"
]

INPAINTING_PROMPTS = [
    "inpainting",
    "content aware fill",
    "edited photo",
    "photoshopped image"
]

# NEW: Composite/Edited photo prompts
COMPOSITE_PROMPTS = [
    "a composite image",
    "a photo collage",
    "a photomontage",
    "an image with pasted elements",
    "a photo with added objects",
    "a manipulated photograph",
    "a photo with face swap",
    "a photo with background replacement",
    "a photo with object removal",
    "a digitally altered photograph"
]


@dataclass
class ContentTypeResult:
    """Result of content type classification"""
    content_type: str  # photo_like, non_photo_like, uncertain
    confidence: str    # high, medium, low
    non_photo_score: float
    photo_score: float
    margin: float
    top_matches: Dict[str, List[Dict]]


class ContentTypeClassifier:
    """
    CLIP-based Photo vs Non-Photo classifier.
    Uses prompt ensemble for robust classification.
    Now includes composite/edited photo detection and photoreal AI detection.
    """
    
    def __init__(self, clip_model=None, clip_preprocess=None, device: str = "cpu"):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # Cached text embeddings
        self._photo_embeddings = None
        self._non_photo_embeddings = None
        self._style_embeddings = None
        self._inpaint_embeddings = None
        self._composite_embeddings = None
        self._photoreal_ai_embeddings = None  # NEW
        
        # Thresholds
        self.alpha = 10.0  # Sigmoid scaling factor
        self.high_confidence_threshold = 0.75
        self.low_confidence_threshold = 0.25
        self.margin_threshold = 0.10
        self.composite_threshold = 0.60
        self.photoreal_ai_threshold = 0.55  # NEW: threshold for photoreal AI detection
    
    def set_clip_model(self, clip_model, clip_preprocess, device: str):
        """Set CLIP model (called after model loading)"""
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self._cache_text_embeddings()
    
    def _cache_text_embeddings(self):
        """Pre-compute and cache text embeddings for all prompts"""
        if not self.clip_model:
            return
        
        try:
            import clip
            
            with torch.no_grad():
                # Photo prompts
                photo_tokens = clip.tokenize(PHOTO_POSITIVE_PROMPTS).to(self.device)
                self._photo_embeddings = self.clip_model.encode_text(photo_tokens)
                self._photo_embeddings = self._photo_embeddings / self._photo_embeddings.norm(dim=-1, keepdim=True)
                
                # Non-photo prompts
                non_photo_tokens = clip.tokenize(NON_PHOTO_POSITIVE_PROMPTS).to(self.device)
                self._non_photo_embeddings = self.clip_model.encode_text(non_photo_tokens)
                self._non_photo_embeddings = self._non_photo_embeddings / self._non_photo_embeddings.norm(dim=-1, keepdim=True)
                
                # Style transfer prompts
                style_tokens = clip.tokenize(STYLE_TRANSFER_PROMPTS).to(self.device)
                self._style_embeddings = self.clip_model.encode_text(style_tokens)
                self._style_embeddings = self._style_embeddings / self._style_embeddings.norm(dim=-1, keepdim=True)
                
                # Inpainting prompts
                inpaint_tokens = clip.tokenize(INPAINTING_PROMPTS).to(self.device)
                self._inpaint_embeddings = self.clip_model.encode_text(inpaint_tokens)
                self._inpaint_embeddings = self._inpaint_embeddings / self._inpaint_embeddings.norm(dim=-1, keepdim=True)
                
                # Composite prompts
                composite_tokens = clip.tokenize(COMPOSITE_PROMPTS).to(self.device)
                self._composite_embeddings = self.clip_model.encode_text(composite_tokens)
                self._composite_embeddings = self._composite_embeddings / self._composite_embeddings.norm(dim=-1, keepdim=True)
                
                # NEW: Photoreal AI prompts
                photoreal_ai_tokens = clip.tokenize(PHOTOREAL_AI_PROMPTS).to(self.device)
                self._photoreal_ai_embeddings = self.clip_model.encode_text(photoreal_ai_tokens)
                self._photoreal_ai_embeddings = self._photoreal_ai_embeddings / self._photoreal_ai_embeddings.norm(dim=-1, keepdim=True)
            
            logger.info("✅ Content type text embeddings cached (including composite and photoreal AI)")
            
        except Exception as e:
            logger.warning(f"Failed to cache text embeddings: {e}")

    
    def classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify image content type using CLIP prompt ensemble.
        
        Returns:
            Dict with content_type, confidence, scores, and top matches
            Now includes composite_score and photo_like_generated_hint for photoreal AI detection
        """
        result = {
            "type": "uncertain",
            "confidence": "low",
            "non_photo_score": 0.5,
            "photo_score": 0.5,
            "composite_score": 0.0,
            "photo_like_generated_hint": 0.0,  # NEW: hint for photoreal AI content
            "margin": 0.0,
            "style_score": 0.0,
            "inpaint_score": 0.0,
            "top_matches": {
                "photo": [],
                "non_photo": [],
                "style": [],
                "inpaint": [],
                "composite": [],
                "photoreal_ai": []  # NEW
            },
            "enabled": False
        }
        
        if not self.clip_model or self._photo_embeddings is None:
            result["error"] = "CLIP model not loaded or embeddings not cached"
            return result
        
        result["enabled"] = True
        
        try:
            # Get image embedding
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            photo_sims = self._calculate_similarities(
                image_features, self._photo_embeddings, PHOTO_POSITIVE_PROMPTS
            )
            non_photo_sims = self._calculate_similarities(
                image_features, self._non_photo_embeddings, NON_PHOTO_POSITIVE_PROMPTS
            )
            style_sims = self._calculate_similarities(
                image_features, self._style_embeddings, STYLE_TRANSFER_PROMPTS
            )
            inpaint_sims = self._calculate_similarities(
                image_features, self._inpaint_embeddings, INPAINTING_PROMPTS
            )
            
            # Composite similarities
            composite_sims = []
            if self._composite_embeddings is not None:
                composite_sims = self._calculate_similarities(
                    image_features, self._composite_embeddings, COMPOSITE_PROMPTS
                )
            
            # NEW: Photoreal AI similarities
            photoreal_ai_sims = []
            if self._photoreal_ai_embeddings is not None:
                photoreal_ai_sims = self._calculate_similarities(
                    image_features, self._photoreal_ai_embeddings, PHOTOREAL_AI_PROMPTS
                )
            
            # Calculate mean scores
            s_photo = np.mean([s["sim"] for s in photo_sims])
            s_nonphoto = np.mean([s["sim"] for s in non_photo_sims])
            s_style = np.mean([s["sim"] for s in style_sims])
            s_inpaint = np.mean([s["sim"] for s in inpaint_sims])
            s_composite = np.mean([s["sim"] for s in composite_sims]) if composite_sims else 0.0
            s_photoreal_ai = np.mean([s["sim"] for s in photoreal_ai_sims]) if photoreal_ai_sims else 0.0
            
            # Calculate margin and non_photo_score
            margin = s_nonphoto - s_photo
            non_photo_score = self._sigmoid(self.alpha * margin)
            
            # Composite score (normalized)
            composite_score = self._sigmoid(self.alpha * (s_composite - s_photo * 0.8))
            
            # NEW: Photo-like generated hint
            # High when image looks like a photo but matches photoreal AI prompts
            # This helps detect AI-generated photorealistic portraits
            photoreal_ai_margin = s_photoreal_ai - s_photo
            photo_like_generated_hint = self._sigmoid(self.alpha * (s_photoreal_ai - s_nonphoto * 0.7))
            
            # Boost hint if image is photo-like but has high photoreal AI similarity
            if non_photo_score < 0.5 and s_photoreal_ai > s_photo * 0.9:
                photo_like_generated_hint = min(1.0, photo_like_generated_hint * 1.3)
            
            result["photo_score"] = round(float(s_photo), 4)
            result["non_photo_score"] = round(float(non_photo_score), 4)
            result["composite_score"] = round(float(composite_score), 4)
            result["photo_like_generated_hint"] = round(float(photo_like_generated_hint), 4)
            result["margin"] = round(float(margin), 4)
            result["style_score"] = round(float(s_style), 4)
            result["inpaint_score"] = round(float(s_inpaint), 4)
            
            # Top matches
            result["top_matches"]["photo"] = sorted(photo_sims, key=lambda x: -x["sim"])[:3]
            result["top_matches"]["non_photo"] = sorted(non_photo_sims, key=lambda x: -x["sim"])[:3]
            result["top_matches"]["style"] = sorted(style_sims, key=lambda x: -x["sim"])[:2]
            result["top_matches"]["inpaint"] = sorted(inpaint_sims, key=lambda x: -x["sim"])[:2]
            if composite_sims:
                result["top_matches"]["composite"] = sorted(composite_sims, key=lambda x: -x["sim"])[:3]
            if photoreal_ai_sims:
                result["top_matches"]["photoreal_ai"] = sorted(photoreal_ai_sims, key=lambda x: -x["sim"])[:3]
            
            # Classify (now includes composite and photoreal AI checks)
            result["type"], result["confidence"] = self._classify_from_scores(
                non_photo_score, margin, composite_score, photo_like_generated_hint
            )
            
            logger.info(f"Content type: {result['type']} ({result['confidence']}) - "
                       f"margin={margin:.3f}, non_photo={non_photo_score:.3f}, "
                       f"composite={composite_score:.3f}, photoreal_ai_hint={photo_like_generated_hint:.3f}")
            
        except Exception as e:
            logger.error(f"Content type classification error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _calculate_similarities(self, image_features, text_embeddings, prompts: List[str]) -> List[Dict]:
        """Calculate cosine similarities between image and text embeddings"""
        similarities = (image_features @ text_embeddings.T).squeeze(0).cpu().numpy()
        
        return [
            {"prompt": prompt, "sim": round(float(sim), 4)}
            for prompt, sim in zip(prompts, similarities)
        ]
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for score normalization"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _classify_from_scores(self, non_photo_score: float, margin: float, 
                               composite_score: float = 0.0,
                               photo_like_generated_hint: float = 0.0) -> Tuple[str, str]:
        """Classify content type from scores, including composite and photoreal AI detection"""
        
        # Check for possible composite first
        if composite_score >= self.composite_threshold:
            # High composite score suggests edited/manipulated photo
            if non_photo_score < 0.5:  # Still looks like a photo base
                return "possible_composite", "medium"
        
        # NEW: Check for photoreal AI (looks like photo but AI-generated)
        if photo_like_generated_hint >= self.photoreal_ai_threshold:
            if non_photo_score < 0.5:  # Looks like a photo
                return "photo_like_generated", "medium"
        
        # Non-photo (CGI/Illustration)
        if non_photo_score >= self.high_confidence_threshold and margin >= self.margin_threshold:
            return "non_photo_like", "high"
        
        # Photo-like
        if non_photo_score <= self.low_confidence_threshold and margin <= -self.margin_threshold:
            return "photo_like", "high"
        
        # Medium confidence zones
        if non_photo_score >= 0.65 and margin >= 0.05:
            return "non_photo_like", "medium"
        
        if non_photo_score <= 0.35 and margin <= -0.05:
            return "photo_like", "medium"
        
        # Uncertain
        return "uncertain", "low"


# Global instance
_content_type_classifier: Optional[ContentTypeClassifier] = None

def get_content_type_classifier() -> ContentTypeClassifier:
    """Get or create global content type classifier"""
    global _content_type_classifier
    if _content_type_classifier is None:
        _content_type_classifier = ContentTypeClassifier()
    return _content_type_classifier

def classify_content_type(image: Image.Image, clip_model=None, 
                         clip_preprocess=None, device: str = "cpu") -> Dict[str, Any]:
    """Convenience function for content type classification"""
    classifier = get_content_type_classifier()
    if clip_model and classifier.clip_model is None:
        classifier.set_clip_model(clip_model, clip_preprocess, device)
    return classifier.classify(image)
