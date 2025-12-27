"""
Content Extraction Module - BLIP Captioning & Tagging
======================================================
Extracts structured content from images using vision-language models.

Models (lazy-loaded, optional):
- BLIP-base: Image captioning (~500MB, CPU-friendly)
- RAM/Tag2Text: Multi-label tagging (optional, heavier)
- CLIP: Style/aesthetic scoring (already loaded in detector)

CRITICAL: All models are OPTIONAL. System works without them.
If models not available, falls back to CLIP-only analysis.

Output:
- caption: Natural language description (EN)
- tags: Multi-label tags with confidence
- scene_graph: Structured extraction (subjects, setting, objects)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports - only load when needed
_blip_model = None
_blip_processor = None
_blip_device = None
_blip_available = None


def _check_blip_available() -> bool:
    """Check if BLIP dependencies are available"""
    global _blip_available
    if _blip_available is not None:
        return _blip_available
    
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_available = True
        logger.info("BLIP dependencies available")
    except ImportError:
        _blip_available = False
        logger.warning("BLIP dependencies not available (transformers not installed)")
    
    return _blip_available


def _load_blip_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Lazy load BLIP model"""
    global _blip_model, _blip_processor, _blip_device
    
    if _blip_model is not None:
        return _blip_model, _blip_processor, _blip_device
    
    if not _check_blip_available():
        return None, None, None
    
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        logger.info(f"Loading BLIP model: {model_name}")
        
        # Determine device
        if torch.cuda.is_available():
            _blip_device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _blip_device = torch.device("mps")
        else:
            _blip_device = torch.device("cpu")
        
        # Load model and processor
        _blip_processor = BlipProcessor.from_pretrained(model_name)
        _blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        _blip_model = _blip_model.to(_blip_device)
        _blip_model.eval()
        
        logger.info(f"BLIP model loaded on {_blip_device}")
        return _blip_model, _blip_processor, _blip_device
        
    except Exception as e:
        logger.error(f"Failed to load BLIP model: {e}")
        return None, None, None


@dataclass
class SceneGraph:
    """Structured scene representation"""
    # Subjects
    subjects: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"type": "woman", "count": 1, "attributes": ["young", "sitting"]}]
    
    # Setting/Environment
    setting: str = ""  # indoor, outdoor, studio
    location: str = ""  # street, park, office, bedroom
    time_of_day: str = ""  # day, night, golden_hour
    weather: str = ""  # sunny, cloudy, rainy
    
    # Objects
    objects: List[str] = field(default_factory=list)
    
    # Actions/Poses
    actions: List[str] = field(default_factory=list)
    
    # Clothing/Accessories (for people)
    clothing: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)
    
    # Style hints
    style: str = ""  # photorealistic, cinematic, editorial
    mood: str = ""  # dramatic, soft, vibrant
    
    # Confidence
    confidence: float = 0.0


@dataclass
class ContentExtractionResult:
    """Result of content extraction"""
    # Caption
    caption: str = ""
    caption_confidence: float = 0.0
    
    # Tags
    tags: List[Dict[str, float]] = field(default_factory=list)
    # Format: [{"tag": "woman", "confidence": 0.95}, ...]
    
    # Scene graph
    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    
    # Method used
    method: str = ""  # "blip", "clip_only", "fallback"
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


class ContentExtractor:
    """
    Extracts structured content from images.
    
    Pipeline:
    1. Generate caption with BLIP (if available)
    2. Extract tags (from caption + CLIP)
    3. Parse into scene graph
    """
    
    def __init__(self, config=None):
        from forensic.config import get_config
        self.config = config or get_config()
        
        # Tag categories for extraction
        self.subject_tags = {
            "person": ["person", "people", "human", "figure"],
            "woman": ["woman", "female", "girl", "lady"],
            "man": ["man", "male", "boy", "guy"],
            "child": ["child", "kid", "baby", "infant"],
            "animal": ["dog", "cat", "bird", "animal", "pet"],
            "group": ["group", "crowd", "couple", "family"]
        }
        
        self.setting_tags = {
            "indoor": ["indoor", "inside", "room", "interior"],
            "outdoor": ["outdoor", "outside", "exterior"],
            "studio": ["studio", "backdrop", "professional"],
            "street": ["street", "road", "sidewalk", "urban", "city"],
            "nature": ["nature", "forest", "mountain", "beach", "park"],
            "office": ["office", "desk", "workplace"],
            "home": ["home", "bedroom", "living room", "kitchen", "bathroom"]
        }
        
        self.action_tags = {
            "sitting": ["sitting", "seated", "sit"],
            "standing": ["standing", "stand"],
            "walking": ["walking", "walk", "strolling"],
            "lying": ["lying", "laying", "reclined"],
            "posing": ["posing", "pose", "model"],
            "selfie": ["selfie", "self-portrait"],
            "looking": ["looking", "gazing", "staring"]
        }
        
        self.style_tags = {
            "photorealistic": ["photo", "photograph", "realistic", "real"],
            "cinematic": ["cinematic", "movie", "film", "dramatic"],
            "portrait": ["portrait", "headshot", "close-up"],
            "fashion": ["fashion", "editorial", "vogue", "glamour"],
            "candid": ["candid", "natural", "spontaneous"],
            "artistic": ["artistic", "art", "creative"]
        }
        
        self.clothing_tags = [
            "dress", "shirt", "pants", "jeans", "skirt", "suit", "jacket",
            "coat", "sweater", "blouse", "shorts", "bikini", "swimsuit",
            "uniform", "casual", "formal", "elegant", "black dress", "white shirt"
        ]
        
        self.object_tags = [
            "car", "phone", "camera", "computer", "book", "bag", "chair",
            "table", "bed", "window", "door", "tree", "flower", "building",
            "food", "drink", "glasses", "hat", "watch", "jewelry"
        ]
    
    def extract(self, image: Image.Image, 
                clip_features: Dict[str, Any] = None) -> ContentExtractionResult:
        """
        Extract content from image.
        
        Args:
            image: PIL Image
            clip_features: Optional CLIP features from content_type classifier
        
        Returns:
            ContentExtractionResult with caption, tags, and scene graph
        """
        result = ContentExtractionResult()
        
        # Try BLIP captioning first
        if self.config.ENABLE_BLIP_CAPTIONING:
            caption, confidence = self._generate_caption_blip(image)
            if caption:
                result.caption = caption
                result.caption_confidence = confidence
                result.method = "blip"
                logger.info(f"BLIP caption: {caption[:100]}...")
        
        # If no BLIP caption, try CLIP-based description
        if not result.caption and clip_features:
            caption = self._generate_caption_clip(clip_features)
            if caption:
                result.caption = caption
                result.caption_confidence = 0.5  # Lower confidence for CLIP-only
                result.method = "clip_only"
        
        # If still no caption, use fallback
        if not result.caption:
            result.caption = "image content"
            result.caption_confidence = 0.1
            result.method = "fallback"
            result.warnings.append("No captioning model available")
        
        # Extract tags from caption
        result.tags = self._extract_tags_from_caption(result.caption)
        
        # Add CLIP-based tags if available
        if clip_features:
            clip_tags = self._extract_tags_from_clip(clip_features)
            result.tags = self._merge_tags(result.tags, clip_tags)
        
        # Build scene graph
        result.scene_graph = self._build_scene_graph(result.caption, result.tags)
        
        return result
    
    def _generate_caption_blip(self, image: Image.Image) -> Tuple[str, float]:
        """Generate caption using BLIP model"""
        model, processor, device = _load_blip_model(self.config.BLIP_MODEL_NAME)
        
        if model is None:
            return "", 0.0
        
        try:
            import torch
            
            # Prepare image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process
            inputs = processor(image, return_tensors="pt").to(device)
            
            # Generate caption
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=self.config.BLIP_MAX_LENGTH,
                    min_length=self.config.BLIP_MIN_LENGTH,
                    num_beams=3,
                    early_stopping=True
                )
            
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # Estimate confidence based on caption quality
            confidence = self._estimate_caption_confidence(caption)
            
            return caption, confidence
            
        except Exception as e:
            logger.error(f"BLIP captioning error: {e}")
            return "", 0.0
    
    def _generate_caption_clip(self, clip_features: Dict[str, Any]) -> str:
        """Generate simple caption from CLIP features"""
        # Use top matches to build a simple description
        top_matches = clip_features.get("top_matches", {})
        
        parts = []
        
        # Get photo matches
        photo_matches = top_matches.get("photo", [])
        if photo_matches:
            # Extract key terms from top match
            top_prompt = photo_matches[0].get("prompt", "")
            if top_prompt:
                parts.append(top_prompt)
        
        # Get AI photoreal matches
        ai_matches = top_matches.get("ai_photoreal", [])
        if ai_matches and not parts:
            top_prompt = ai_matches[0].get("prompt", "")
            if top_prompt:
                parts.append(top_prompt)
        
        return " ".join(parts) if parts else ""
    
    def _estimate_caption_confidence(self, caption: str) -> float:
        """Estimate confidence based on caption quality"""
        if not caption:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Longer captions tend to be more detailed
        word_count = len(caption.split())
        if word_count >= 10:
            confidence += 0.2
        elif word_count >= 5:
            confidence += 0.1
        
        # Check for specific content indicators
        specific_indicators = ["woman", "man", "person", "sitting", "standing", 
                             "wearing", "holding", "looking", "street", "room"]
        matches = sum(1 for ind in specific_indicators if ind in caption.lower())
        confidence += min(0.2, matches * 0.05)
        
        return min(1.0, confidence)
    
    def _extract_tags_from_caption(self, caption: str) -> List[Dict[str, float]]:
        """Extract tags from caption text"""
        tags = []
        caption_lower = caption.lower()
        
        # Check subject tags
        for category, keywords in self.subject_tags.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    tags.append({"tag": category, "confidence": 0.8, "source": "caption"})
                    break
        
        # Check setting tags
        for category, keywords in self.setting_tags.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    tags.append({"tag": category, "confidence": 0.7, "source": "caption"})
                    break
        
        # Check action tags
        for category, keywords in self.action_tags.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    tags.append({"tag": category, "confidence": 0.7, "source": "caption"})
                    break
        
        # Check style tags
        for category, keywords in self.style_tags.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    tags.append({"tag": category, "confidence": 0.6, "source": "caption"})
                    break
        
        # Check clothing tags
        for clothing in self.clothing_tags:
            if clothing in caption_lower:
                tags.append({"tag": clothing, "confidence": 0.7, "source": "caption"})
        
        # Check object tags
        for obj in self.object_tags:
            if obj in caption_lower:
                tags.append({"tag": obj, "confidence": 0.6, "source": "caption"})
        
        return tags
    
    def _extract_tags_from_clip(self, clip_features: Dict[str, Any]) -> List[Dict[str, float]]:
        """Extract tags from CLIP features"""
        tags = []
        
        top_matches = clip_features.get("top_matches", {})
        
        # Process all match categories
        for category, matches in top_matches.items():
            for match in matches[:3]:  # Top 3 from each category
                prompt = match.get("prompt", "").lower()
                score = match.get("score", 0)
                
                # Extract keywords from prompt
                for word in prompt.split():
                    if len(word) > 3:  # Skip short words
                        # Check if it's a meaningful tag
                        for tag_category, keywords in {**self.subject_tags, **self.setting_tags}.items():
                            if word in keywords:
                                tags.append({
                                    "tag": tag_category, 
                                    "confidence": score * 0.8,
                                    "source": "clip"
                                })
        
        return tags
    
    def _merge_tags(self, tags1: List[Dict], tags2: List[Dict]) -> List[Dict]:
        """Merge two tag lists, combining confidence for duplicates"""
        tag_dict = {}
        
        for tag_info in tags1 + tags2:
            tag = tag_info["tag"]
            conf = tag_info["confidence"]
            
            if tag in tag_dict:
                # Average confidence for duplicates
                tag_dict[tag]["confidence"] = (tag_dict[tag]["confidence"] + conf) / 2
                tag_dict[tag]["sources"].append(tag_info.get("source", "unknown"))
            else:
                tag_dict[tag] = {
                    "tag": tag,
                    "confidence": conf,
                    "sources": [tag_info.get("source", "unknown")]
                }
        
        # Convert back to list and sort by confidence
        result = list(tag_dict.values())
        result.sort(key=lambda x: -x["confidence"])
        
        return result
    
    def _build_scene_graph(self, caption: str, tags: List[Dict]) -> SceneGraph:
        """Build structured scene graph from caption and tags"""
        graph = SceneGraph()
        caption_lower = caption.lower()
        
        # Extract tag names for easy lookup
        tag_names = {t["tag"] for t in tags}
        
        # Subjects
        for subject_type in ["woman", "man", "person", "child", "group"]:
            if subject_type in tag_names or subject_type in caption_lower:
                subject = {"type": subject_type, "count": 1, "attributes": []}
                
                # Add attributes
                for action in ["sitting", "standing", "walking", "lying", "posing"]:
                    if action in tag_names or action in caption_lower:
                        subject["attributes"].append(action)
                
                graph.subjects.append(subject)
        
        # Setting
        for setting in ["indoor", "outdoor", "studio"]:
            if setting in tag_names or setting in caption_lower:
                graph.setting = setting
                break
        
        # Location
        for location in ["street", "nature", "office", "home"]:
            if location in tag_names or location in caption_lower:
                graph.location = location
                break
        
        # More specific location from caption
        location_keywords = {
            "street": ["street", "sidewalk", "road", "urban"],
            "park": ["park", "garden", "grass"],
            "beach": ["beach", "ocean", "sea", "sand"],
            "office": ["office", "desk", "workplace"],
            "bedroom": ["bedroom", "bed"],
            "kitchen": ["kitchen"],
            "bathroom": ["bathroom"],
            "studio": ["studio", "backdrop"]
        }
        
        for loc, keywords in location_keywords.items():
            if any(kw in caption_lower for kw in keywords):
                graph.location = loc
                break
        
        # Actions
        for action in ["sitting", "standing", "walking", "lying", "posing", "selfie", "looking"]:
            if action in tag_names or action in caption_lower:
                graph.actions.append(action)
        
        # Clothing
        for clothing in self.clothing_tags:
            if clothing in caption_lower:
                graph.clothing.append(clothing)
        
        # Objects
        for obj in self.object_tags:
            if obj in caption_lower:
                graph.objects.append(obj)
        
        # Style
        for style in ["photorealistic", "cinematic", "portrait", "fashion", "candid", "artistic"]:
            if style in tag_names:
                graph.style = style
                break
        
        # Default style based on content
        if not graph.style:
            if graph.subjects:
                graph.style = "portrait" if len(graph.subjects) == 1 else "group"
            else:
                graph.style = "photorealistic"
        
        # Compute confidence
        filled_fields = sum([
            len(graph.subjects) > 0,
            bool(graph.setting),
            bool(graph.location),
            len(graph.actions) > 0,
            len(graph.clothing) > 0,
            len(graph.objects) > 0
        ])
        graph.confidence = filled_fields / 6.0
        
        return graph


# Global instance
_content_extractor: Optional[ContentExtractor] = None

def get_content_extractor() -> ContentExtractor:
    global _content_extractor
    if _content_extractor is None:
        _content_extractor = ContentExtractor()
    return _content_extractor
