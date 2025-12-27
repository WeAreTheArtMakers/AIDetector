"""
Prompt Compiler Module - Scene Graph to Prompt Conversion
==========================================================
Compiles structured scene graph into SD/MJ compatible prompts.

Input: SceneGraph from content_extraction.py
Output: Compiled prompts (TR/EN) with confidence

CRITICAL: This produces RECONSTRUCTED prompts (hypothesis), NOT recovered prompts.
The distinction must be clear in all outputs.

Prompt structure follows SD/MJ best practices:
1. Subject (who/what)
2. Action/Pose
3. Setting/Environment
4. Clothing/Attributes
5. Lighting/Mood
6. Camera/Composition
7. Style modifiers
8. Quality tags
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompiledPrompt:
    """Result of prompt compilation"""
    # Short prompt (1 line, essential elements)
    short_en: str = ""
    short_tr: str = ""
    
    # Long prompt (detailed, all elements)
    long_en: str = ""
    long_tr: str = ""
    
    # Negative prompt suggestions
    negative_en: str = ""
    negative_tr: str = ""
    
    # Structured breakdown
    breakdown: Dict[str, str] = field(default_factory=dict)
    
    # Confidence
    confidence: str = "low"  # low, medium, high
    confidence_score: float = 0.0
    
    # Compilation notes
    notes: List[str] = field(default_factory=list)


class PromptCompiler:
    """
    Compiles scene graph into SD/MJ compatible prompts.
    
    Supports:
    - English prompts (primary)
    - Turkish prompts (translated)
    - Negative prompt suggestions
    - Multiple detail levels (short/long)
    """
    
    def __init__(self):
        # Translation dictionaries
        self.subject_tr = {
            "woman": "kadın",
            "man": "erkek",
            "person": "kişi",
            "child": "çocuk",
            "group": "grup",
            "young": "genç",
            "adult": "yetişkin",
            "elderly": "yaşlı"
        }
        
        self.action_tr = {
            "sitting": "oturan",
            "standing": "ayakta duran",
            "walking": "yürüyen",
            "lying": "uzanan",
            "posing": "poz veren",
            "looking": "bakan",
            "selfie": "selfie çeken"
        }
        
        self.setting_tr = {
            "indoor": "iç mekan",
            "outdoor": "dış mekan",
            "studio": "stüdyo",
            "street": "sokak",
            "park": "park",
            "beach": "plaj",
            "office": "ofis",
            "bedroom": "yatak odası",
            "nature": "doğa"
        }
        
        self.style_tr = {
            "photorealistic": "fotogerçekçi",
            "cinematic": "sinematik",
            "portrait": "portre",
            "fashion": "moda",
            "candid": "doğal",
            "artistic": "sanatsal",
            "dramatic": "dramatik",
            "soft": "yumuşak"
        }
        
        self.lighting_tr = {
            "natural light": "doğal ışık",
            "studio lighting": "stüdyo aydınlatması",
            "dramatic lighting": "dramatik aydınlatma",
            "soft light": "yumuşak ışık",
            "golden hour": "altın saat",
            "backlight": "arka ışık"
        }
        
        self.camera_tr = {
            "shallow depth of field": "sığ alan derinliği",
            "bokeh": "bokeh",
            "wide angle": "geniş açı",
            "close-up": "yakın çekim",
            "full body": "tam boy",
            "portrait shot": "portre çekimi"
        }
        
        # Quality tags
        self.quality_tags_en = [
            "highly detailed", "professional photography", 
            "sharp focus", "8k", "high resolution"
        ]
        
        self.quality_tags_tr = [
            "yüksek detay", "profesyonel fotoğrafçılık",
            "keskin odak", "8k", "yüksek çözünürlük"
        ]
        
        # Common negative prompts
        self.negative_base_en = [
            "lowres", "jpeg artifacts", "blurry", "worst quality", "low quality",
            "deformed", "ugly", "bad anatomy", "bad proportions",
            "text", "watermark", "logo", "signature"
        ]
        
        self.negative_base_tr = [
            "düşük çözünürlük", "jpeg artefaktları", "bulanık", "kötü kalite",
            "deforme", "çirkin", "kötü anatomi", "kötü oranlar",
            "metin", "filigran", "logo", "imza"
        ]
        
        self.negative_portrait_en = [
            "deformed hands", "extra fingers", "mutated hands",
            "poorly drawn hands", "poorly drawn face", "disfigured",
            "extra limbs", "missing arms", "fused fingers"
        ]
        
        self.negative_portrait_tr = [
            "deforme eller", "fazla parmak", "mutasyonlu eller",
            "kötü çizilmiş eller", "kötü çizilmiş yüz", "şekil bozukluğu",
            "fazla uzuv", "eksik kol", "yapışık parmaklar"
        ]
    
    def compile(self, 
                scene_graph,  # SceneGraph from content_extraction
                content_result=None,  # ContentExtractionResult
                style_hints: Dict[str, Any] = None) -> CompiledPrompt:
        """
        Compile scene graph into prompts.
        
        Args:
            scene_graph: SceneGraph object
            content_result: Full ContentExtractionResult (optional)
            style_hints: Additional style hints from analysis
        
        Returns:
            CompiledPrompt with short/long versions in EN/TR
        """
        result = CompiledPrompt()
        style_hints = style_hints or {}
        
        # Build prompt components
        components = self._extract_components(scene_graph, style_hints)
        result.breakdown = components
        
        # Compile English prompts
        result.short_en = self._compile_short_en(components)
        result.long_en = self._compile_long_en(components)
        
        # Compile Turkish prompts
        result.short_tr = self._compile_short_tr(components)
        result.long_tr = self._compile_long_tr(components)
        
        # Generate negative prompts
        result.negative_en = self._compile_negative_en(components)
        result.negative_tr = self._compile_negative_tr(components)
        
        # Compute confidence
        result.confidence, result.confidence_score = self._compute_confidence(
            scene_graph, components, content_result
        )
        
        # Add notes
        result.notes = self._generate_notes(components, scene_graph)
        
        logger.info(f"Compiled prompt: confidence={result.confidence}, "
                   f"short_en={len(result.short_en)} chars")
        
        return result
    
    def _extract_components(self, scene_graph, style_hints: Dict) -> Dict[str, str]:
        """Extract prompt components from scene graph"""
        components = {
            "subject": "",
            "action": "",
            "setting": "",
            "location": "",
            "clothing": "",
            "objects": "",
            "lighting": "",
            "camera": "",
            "style": "",
            "mood": "",
            "quality": ""
        }
        
        # Subject
        if scene_graph.subjects:
            subject_parts = []
            for subj in scene_graph.subjects:
                subj_type = subj.get("type", "person")
                count = subj.get("count", 1)
                attrs = subj.get("attributes", [])
                
                # Build subject string
                if count > 1:
                    subj_str = f"{count} {subj_type}s"
                else:
                    # Add age hint if available
                    age = "young" if "young" in attrs else ""
                    subj_str = f"{age} {subj_type}".strip()
                
                subject_parts.append(subj_str)
            
            components["subject"] = ", ".join(subject_parts)
        
        # Action/Pose
        if scene_graph.actions:
            components["action"] = scene_graph.actions[0]  # Primary action
        
        # Setting
        if scene_graph.setting:
            components["setting"] = scene_graph.setting
        
        # Location
        if scene_graph.location:
            components["location"] = scene_graph.location
        
        # Clothing
        if scene_graph.clothing:
            components["clothing"] = ", ".join(scene_graph.clothing[:3])
        
        # Objects
        if scene_graph.objects:
            components["objects"] = ", ".join(scene_graph.objects[:3])
        
        # Style
        if scene_graph.style:
            components["style"] = scene_graph.style
        elif style_hints.get("style"):
            components["style"] = style_hints["style"]
        else:
            components["style"] = "photorealistic"
        
        # Mood
        if scene_graph.mood:
            components["mood"] = scene_graph.mood
        elif style_hints.get("mood"):
            components["mood"] = style_hints["mood"]
        
        # Lighting (from style hints or infer)
        if style_hints.get("lighting"):
            components["lighting"] = style_hints["lighting"]
        elif components["setting"] == "outdoor":
            components["lighting"] = "natural light"
        elif components["setting"] == "studio":
            components["lighting"] = "studio lighting"
        else:
            components["lighting"] = "soft light"
        
        # Camera (from style hints or infer)
        if style_hints.get("camera"):
            components["camera"] = style_hints["camera"]
        elif components["subject"] and "portrait" in components["style"]:
            components["camera"] = "85mm lens, shallow depth of field, bokeh"
        elif components["setting"] == "outdoor":
            components["camera"] = "natural perspective"
        
        # Quality tags
        components["quality"] = ", ".join(self.quality_tags_en[:3])
        
        return components
    
    def _compile_short_en(self, components: Dict[str, str]) -> str:
        """Compile short English prompt"""
        parts = []
        
        # Subject + action
        if components["subject"]:
            if components["action"]:
                parts.append(f"{components['subject']} {components['action']}")
            else:
                parts.append(components["subject"])
        
        # Location
        if components["location"]:
            parts.append(f"in {components['location']}")
        elif components["setting"]:
            parts.append(f"{components['setting']} setting")
        
        # Style
        if components["style"]:
            parts.append(components["style"])
        
        return ", ".join(parts) if parts else "photorealistic image"
    
    def _compile_long_en(self, components: Dict[str, str]) -> str:
        """Compile long English prompt"""
        parts = []
        
        # Subject + action
        if components["subject"]:
            subj_part = components["subject"]
            if components["action"]:
                subj_part += f" {components['action']}"
            parts.append(subj_part)
        
        # Clothing
        if components["clothing"]:
            parts.append(f"wearing {components['clothing']}")
        
        # Location/Setting
        if components["location"]:
            parts.append(f"in {components['location']}")
        elif components["setting"]:
            parts.append(f"{components['setting']} environment")
        
        # Objects
        if components["objects"]:
            parts.append(f"with {components['objects']}")
        
        # Lighting
        if components["lighting"]:
            parts.append(components["lighting"])
        
        # Camera
        if components["camera"]:
            parts.append(components["camera"])
        
        # Style
        if components["style"]:
            parts.append(components["style"])
        
        # Mood
        if components["mood"]:
            parts.append(components["mood"])
        
        # Quality
        parts.append(components["quality"])
        
        return ", ".join(parts) if parts else "photorealistic, highly detailed"
    
    def _compile_short_tr(self, components: Dict[str, str]) -> str:
        """Compile short Turkish prompt"""
        parts = []
        
        # Subject
        if components["subject"]:
            subject_tr = self._translate_subject(components["subject"])
            if components["action"]:
                action_tr = self.action_tr.get(components["action"], components["action"])
                parts.append(f"{action_tr} {subject_tr}")
            else:
                parts.append(subject_tr)
        
        # Location
        if components["location"]:
            loc_tr = self.setting_tr.get(components["location"], components["location"])
            parts.append(f"{loc_tr} ortamında")
        elif components["setting"]:
            setting_tr = self.setting_tr.get(components["setting"], components["setting"])
            parts.append(setting_tr)
        
        # Style
        if components["style"]:
            style_tr = self.style_tr.get(components["style"], components["style"])
            parts.append(style_tr)
        
        return ", ".join(parts) if parts else "fotogerçekçi görsel"
    
    def _compile_long_tr(self, components: Dict[str, str]) -> str:
        """Compile long Turkish prompt"""
        parts = []
        
        # Subject + action
        if components["subject"]:
            subject_tr = self._translate_subject(components["subject"])
            if components["action"]:
                action_tr = self.action_tr.get(components["action"], components["action"])
                parts.append(f"{action_tr} {subject_tr}")
            else:
                parts.append(subject_tr)
        
        # Clothing
        if components["clothing"]:
            parts.append(f"{components['clothing']} giymiş")
        
        # Location
        if components["location"]:
            loc_tr = self.setting_tr.get(components["location"], components["location"])
            parts.append(f"{loc_tr} ortamında")
        elif components["setting"]:
            setting_tr = self.setting_tr.get(components["setting"], components["setting"])
            parts.append(f"{setting_tr} ortam")
        
        # Lighting
        if components["lighting"]:
            light_tr = self.lighting_tr.get(components["lighting"], components["lighting"])
            parts.append(light_tr)
        
        # Camera
        if components["camera"]:
            camera_tr = self._translate_camera(components["camera"])
            parts.append(camera_tr)
        
        # Style
        if components["style"]:
            style_tr = self.style_tr.get(components["style"], components["style"])
            parts.append(style_tr)
        
        # Quality
        parts.append(", ".join(self.quality_tags_tr[:2]))
        
        return ", ".join(parts) if parts else "fotogerçekçi, yüksek detay"
    
    def _translate_subject(self, subject: str) -> str:
        """Translate subject to Turkish"""
        subject_lower = subject.lower()
        
        for en, tr in self.subject_tr.items():
            if en in subject_lower:
                subject_lower = subject_lower.replace(en, tr)
        
        return subject_lower
    
    def _translate_camera(self, camera: str) -> str:
        """Translate camera terms to Turkish"""
        camera_lower = camera.lower()
        
        for en, tr in self.camera_tr.items():
            if en in camera_lower:
                camera_lower = camera_lower.replace(en, tr)
        
        return camera_lower
    
    def _compile_negative_en(self, components: Dict[str, str]) -> str:
        """Compile English negative prompt"""
        negatives = self.negative_base_en.copy()
        
        # Add portrait-specific negatives if subject is person
        if components["subject"]:
            negatives.extend(self.negative_portrait_en[:5])
        
        return ", ".join(negatives)
    
    def _compile_negative_tr(self, components: Dict[str, str]) -> str:
        """Compile Turkish negative prompt"""
        negatives = self.negative_base_tr.copy()
        
        if components["subject"]:
            negatives.extend(self.negative_portrait_tr[:5])
        
        return ", ".join(negatives)
    
    def _compute_confidence(self, scene_graph, components: Dict, 
                           content_result) -> Tuple[str, float]:
        """Compute confidence level for compiled prompt"""
        score = 0.0
        
        # Scene graph confidence
        if hasattr(scene_graph, 'confidence'):
            score += scene_graph.confidence * 0.4
        
        # Component completeness
        filled = sum(1 for v in components.values() if v)
        score += (filled / len(components)) * 0.3
        
        # Content extraction confidence
        if content_result and hasattr(content_result, 'caption_confidence'):
            score += content_result.caption_confidence * 0.3
        
        # Determine tier
        if score >= 0.65:
            return "high", score
        elif score >= 0.40:
            return "medium", score
        else:
            return "low", score
    
    def _generate_notes(self, components: Dict, scene_graph) -> List[str]:
        """Generate compilation notes"""
        notes = []
        
        if not components["subject"]:
            notes.append("No clear subject detected")
        
        if not components["location"] and not components["setting"]:
            notes.append("Setting/location unclear")
        
        if not components["clothing"] and components["subject"]:
            notes.append("Clothing details not extracted")
        
        if scene_graph.confidence < 0.5:
            notes.append("Low scene graph confidence - prompt may be incomplete")
        
        return notes


# Global instance
_prompt_compiler: Optional[PromptCompiler] = None

def get_prompt_compiler() -> PromptCompiler:
    global _prompt_compiler
    if _prompt_compiler is None:
        _prompt_compiler = PromptCompiler()
    return _prompt_compiler
