"""
Content Understanding Module - Visual Content Analysis
=======================================================
Extracts structured content information from images for prompt hypothesis.

Uses lightweight, deterministic methods:
- Scene classification (indoor/outdoor, environment type)
- Subject detection (people count, poses, clothing)
- Composition analysis (framing, depth of field)
- Lighting analysis (natural/studio, time of day)
- Style inference (photorealistic, cinematic, etc.)

CRITICAL: Does NOT claim to recover original prompts.
All outputs are "hypotheses from visual content".
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContentSlots:
    """Structured content slots extracted from image"""
    # Subject
    subject_count: int = 0
    subject_types: List[str] = field(default_factory=list)  # woman, man, child, animal
    age_buckets: List[str] = field(default_factory=list)  # young_adult, adult, elderly
    poses: List[str] = field(default_factory=list)  # sitting, standing, walking
    clothing: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)
    
    # Environment
    environment: str = ""  # indoor, outdoor, studio
    location_type: str = ""  # street, park, office, bedroom, etc.
    time_of_day: str = ""  # day, night, golden_hour, blue_hour
    weather: str = ""  # sunny, cloudy, rainy
    
    # Camera/Composition
    shot_type: str = ""  # portrait, full_body, close_up, wide
    camera_angle: str = ""  # eye_level, low_angle, high_angle
    depth_of_field: str = ""  # shallow, deep, medium
    focal_length_hint: str = ""  # wide, standard, telephoto
    
    # Lighting
    lighting_type: str = ""  # natural, studio, dramatic, soft
    lighting_direction: str = ""  # front, side, back, ambient
    
    # Style
    style: str = ""  # photorealistic, cinematic, editorial, candid
    aesthetic: str = ""  # high_contrast, soft, vibrant, muted
    
    # Confidence
    extraction_confidence: float = 0.0


class ContentUnderstanding:
    """
    Extracts structured content understanding from images.
    
    Uses image analysis to infer:
    - What is in the image (subjects, objects, scene)
    - How it was shot (composition, lighting, camera)
    - What style it represents (aesthetic, mood)
    """
    
    def __init__(self):
        # Scene type keywords for classification
        self.indoor_keywords = ["room", "office", "bedroom", "kitchen", "bathroom", 
                               "studio", "interior", "indoor", "inside"]
        self.outdoor_keywords = ["street", "park", "beach", "mountain", "city",
                                "outdoor", "outside", "nature", "sky"]
        
        # Lighting analysis thresholds
        self.brightness_thresholds = {
            "dark": 60,
            "dim": 100,
            "normal": 160,
            "bright": 200
        }
    
    def analyze(self, image: Image.Image, 
                clip_features: Dict[str, Any] = None,
                content_type: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis entry point.
        
        Args:
            image: PIL Image
            clip_features: Optional CLIP features from content_type classifier
            content_type: Optional content type classification result
        
        Returns:
            Dict with slots, confidence, and raw analysis data
        """
        result = {
            "slots": {},
            "confidence": 0.0,
            "raw_analysis": {},
            "enabled": True
        }
        
        try:
            # Convert to RGB array
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            
            # 1. Analyze composition
            composition = self._analyze_composition(image, img_array)
            result["raw_analysis"]["composition"] = composition
            
            # 2. Analyze lighting
            lighting = self._analyze_lighting(img_array)
            result["raw_analysis"]["lighting"] = lighting
            
            # 3. Analyze color/style
            style = self._analyze_style(img_array)
            result["raw_analysis"]["style"] = style
            
            # 4. Infer scene from CLIP features if available
            scene = self._infer_scene(clip_features, content_type)
            result["raw_analysis"]["scene"] = scene
            
            # 5. Build content slots
            slots = self._build_slots(composition, lighting, style, scene)
            result["slots"] = slots.__dict__
            result["confidence"] = slots.extraction_confidence
            
            logger.info(f"Content understanding: confidence={slots.extraction_confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Content understanding error: {e}")
            result["enabled"] = False
            result["error"] = str(e)
        
        return result
    
    def _analyze_composition(self, image: Image.Image, 
                             img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition"""
        h, w = img_array.shape[:2]
        aspect_ratio = w / h
        
        # Determine shot type from aspect ratio and content distribution
        shot_type = "medium"
        if aspect_ratio > 1.5:
            shot_type = "wide"
        elif aspect_ratio < 0.75:
            shot_type = "portrait"
        elif abs(aspect_ratio - 1.0) < 0.1:
            shot_type = "square"
        
        # Analyze blur distribution for depth of field
        dof = self._estimate_depth_of_field(img_array)
        
        # Analyze content distribution (center-weighted vs spread)
        center_weight = self._analyze_center_weight(img_array)
        
        return {
            "aspect_ratio": round(aspect_ratio, 3),
            "shot_type": shot_type,
            "depth_of_field": dof,
            "center_weighted": center_weight > 0.6,
            "dimensions": {"width": w, "height": h}
        }
    
    def _estimate_depth_of_field(self, img_array: np.ndarray) -> str:
        """Estimate depth of field from blur distribution"""
        from scipy.ndimage import laplace, gaussian_filter
        
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)
        
        # Compute local sharpness using Laplacian
        lap = np.abs(laplace(gray))
        
        # Divide into regions
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        margin_h, margin_w = h // 4, w // 4
        
        # Center region sharpness
        center_region = lap[center_h-margin_h:center_h+margin_h, 
                           center_w-margin_w:center_w+margin_w]
        center_sharpness = np.mean(center_region)
        
        # Edge regions sharpness
        top_sharpness = np.mean(lap[:margin_h, :])
        bottom_sharpness = np.mean(lap[-margin_h:, :])
        left_sharpness = np.mean(lap[:, :margin_w])
        right_sharpness = np.mean(lap[:, -margin_w:])
        edge_sharpness = np.mean([top_sharpness, bottom_sharpness, 
                                  left_sharpness, right_sharpness])
        
        # Compare center vs edge sharpness
        if center_sharpness > 0:
            ratio = edge_sharpness / (center_sharpness + 1e-6)
            
            if ratio < 0.5:
                return "shallow"  # Bokeh effect
            elif ratio < 0.8:
                return "medium"
            else:
                return "deep"
        
        return "unknown"
    
    def _analyze_center_weight(self, img_array: np.ndarray) -> float:
        """Analyze how center-weighted the content is"""
        from scipy.ndimage import laplace
        
        gray = np.mean(img_array, axis=2)
        h, w = gray.shape
        
        # Compute edge magnitude as proxy for content
        edges = np.abs(laplace(gray))
        
        # Create center mask
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Weight by distance from center (closer = higher weight)
        weights = 1 - (dist / max_dist)
        
        # Compute weighted vs unweighted edge sum
        weighted_sum = np.sum(edges * weights)
        total_sum = np.sum(edges)
        
        if total_sum > 0:
            return weighted_sum / total_sum
        return 0.5
    
    def _analyze_lighting(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting characteristics"""
        # Overall brightness
        brightness = np.mean(img_array)
        
        # Contrast
        contrast = np.std(img_array)
        
        # Determine lighting level
        if brightness < self.brightness_thresholds["dark"]:
            level = "dark"
        elif brightness < self.brightness_thresholds["dim"]:
            level = "dim"
        elif brightness < self.brightness_thresholds["normal"]:
            level = "normal"
        elif brightness < self.brightness_thresholds["bright"]:
            level = "bright"
        else:
            level = "very_bright"
        
        # Analyze lighting direction from brightness gradient
        h, w = img_array.shape[:2]
        gray = np.mean(img_array, axis=2)
        
        # Compare quadrants
        top_brightness = np.mean(gray[:h//2, :])
        bottom_brightness = np.mean(gray[h//2:, :])
        left_brightness = np.mean(gray[:, :w//2])
        right_brightness = np.mean(gray[:, w//2:])
        
        # Determine dominant light direction
        direction = "ambient"
        max_diff = 0
        
        if abs(top_brightness - bottom_brightness) > max_diff:
            max_diff = abs(top_brightness - bottom_brightness)
            direction = "top" if top_brightness > bottom_brightness else "bottom"
        
        if abs(left_brightness - right_brightness) > max_diff:
            direction = "left" if left_brightness > right_brightness else "right"
        
        # Determine lighting type
        if contrast > 70:
            lighting_type = "dramatic"
        elif contrast < 35:
            lighting_type = "soft"
        else:
            lighting_type = "natural"
        
        return {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "level": level,
            "direction": direction,
            "type": lighting_type
        }
    
    def _analyze_style(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze visual style and aesthetics"""
        # Color saturation
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Simple saturation estimate
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-6))
        
        # Color temperature (warm vs cool)
        avg_r = np.mean(r)
        avg_b = np.mean(b)
        temperature = "neutral"
        if avg_r > avg_b * 1.15:
            temperature = "warm"
        elif avg_b > avg_r * 1.15:
            temperature = "cool"
        
        # Determine aesthetic
        contrast = np.std(img_array)
        
        if saturation > 0.5 and contrast > 60:
            aesthetic = "vibrant"
        elif saturation < 0.25:
            aesthetic = "muted"
        elif contrast > 70:
            aesthetic = "high_contrast"
        elif contrast < 40:
            aesthetic = "soft"
        else:
            aesthetic = "balanced"
        
        # Determine style hint
        style = "photorealistic"
        if saturation > 0.6:
            style = "vivid"
        elif contrast > 75:
            style = "dramatic"
        elif contrast < 35 and saturation < 0.3:
            style = "dreamy"
        
        return {
            "saturation": round(saturation, 3),
            "temperature": temperature,
            "aesthetic": aesthetic,
            "style_hint": style
        }
    
    def _infer_scene(self, clip_features: Dict[str, Any],
                     content_type: Dict[str, Any]) -> Dict[str, Any]:
        """Infer scene information from CLIP features"""
        result = {
            "environment": "unknown",
            "location_type": "unknown",
            "subject_hints": []
        }
        
        if not content_type:
            return result
        
        # Extract from top matches
        top_matches = content_type.get("top_matches", {})
        
        # Check photo matches for scene hints
        photo_matches = top_matches.get("photo", [])
        ai_photoreal_matches = top_matches.get("ai_photoreal", [])
        
        all_matches = photo_matches + ai_photoreal_matches
        
        for match in all_matches:
            prompt = match.get("prompt", "").lower()
            
            # Environment detection
            if any(kw in prompt for kw in self.indoor_keywords):
                result["environment"] = "indoor"
            elif any(kw in prompt for kw in self.outdoor_keywords):
                result["environment"] = "outdoor"
            
            # Subject detection
            if "portrait" in prompt or "person" in prompt or "face" in prompt:
                result["subject_hints"].append("person")
            if "woman" in prompt or "female" in prompt:
                result["subject_hints"].append("woman")
            if "man" in prompt or "male" in prompt:
                result["subject_hints"].append("man")
            if "landscape" in prompt:
                result["subject_hints"].append("landscape")
            if "street" in prompt or "urban" in prompt:
                result["location_type"] = "street"
            if "studio" in prompt:
                result["location_type"] = "studio"
        
        # Deduplicate
        result["subject_hints"] = list(set(result["subject_hints"]))
        
        return result
    
    def _build_slots(self, composition: Dict, lighting: Dict,
                     style: Dict, scene: Dict) -> ContentSlots:
        """Build structured content slots from analysis"""
        slots = ContentSlots()
        
        # Composition
        slots.shot_type = composition.get("shot_type", "")
        slots.depth_of_field = composition.get("depth_of_field", "")
        
        if composition.get("depth_of_field") == "shallow":
            slots.focal_length_hint = "telephoto"
        elif composition.get("shot_type") == "wide":
            slots.focal_length_hint = "wide"
        else:
            slots.focal_length_hint = "standard"
        
        # Lighting
        slots.lighting_type = lighting.get("type", "")
        slots.lighting_direction = lighting.get("direction", "")
        
        # Infer time of day from brightness
        level = lighting.get("level", "normal")
        if level in ["dark", "dim"]:
            slots.time_of_day = "night"
        elif level == "very_bright":
            slots.time_of_day = "midday"
        else:
            slots.time_of_day = "day"
        
        # Style
        slots.style = style.get("style_hint", "")
        slots.aesthetic = style.get("aesthetic", "")
        
        # Scene
        slots.environment = scene.get("environment", "")
        slots.location_type = scene.get("location_type", "")
        
        # Subject hints
        subject_hints = scene.get("subject_hints", [])
        if "woman" in subject_hints:
            slots.subject_types.append("woman")
            slots.subject_count = max(slots.subject_count, 1)
        if "man" in subject_hints:
            slots.subject_types.append("man")
            slots.subject_count = max(slots.subject_count, 1)
        if "person" in subject_hints and not slots.subject_types:
            slots.subject_types.append("person")
            slots.subject_count = max(slots.subject_count, 1)
        
        # Compute confidence
        filled_slots = sum([
            bool(slots.shot_type),
            bool(slots.depth_of_field),
            bool(slots.lighting_type),
            bool(slots.style),
            bool(slots.environment),
            len(slots.subject_types) > 0
        ])
        slots.extraction_confidence = filled_slots / 6.0
        
        return slots


# Global instance
_content_understanding: Optional[ContentUnderstanding] = None

def get_content_understanding() -> ContentUnderstanding:
    global _content_understanding
    if _content_understanding is None:
        _content_understanding = ContentUnderstanding()
    return _content_understanding
