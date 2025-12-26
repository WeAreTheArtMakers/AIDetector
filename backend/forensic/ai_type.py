"""
AI Generation Type Classifier
==============================
Heuristic classification of AI generation type:
- text_to_image
- img2img
- inpainting
- unknown
"""

import logging
import numpy as np
from typing import Dict, Any, List
from PIL import Image

logger = logging.getLogger(__name__)

class AITypeClassifier:
    """Classify AI generation type based on heuristics"""
    
    def classify(self, image: Image.Image, metadata: Dict, 
                model_scores: List[Dict] = None) -> Dict[str, Any]:
        """
        Classify AI generation type
        
        This is HEURISTIC only - not definitive evidence.
        """
        result = {
            "predicted_type": "unknown",
            "guess": "unknown",  # Backward compat
            "confidence": 0.0,
            "indicators": [],
            "evidence": [],  # Backward compat
            "disclaimer": "This is a heuristic guess based on image characteristics, not definitive proof."
        }
        
        try:
            # Collect evidence
            evidence = []
            scores = {
                "text_to_image": 0.0,
                "img2img": 0.0,
                "inpainting": 0.0
            }
            
            # Check dimensions (text-to-image often has specific sizes)
            dim_evidence = self._analyze_dimensions(image)
            evidence.extend(dim_evidence["evidence"])
            scores["text_to_image"] += dim_evidence.get("t2i_score", 0)
            
            # Check for partial editing patterns (inpainting)
            inpaint_evidence = self._detect_inpainting_patterns(image)
            evidence.extend(inpaint_evidence["evidence"])
            scores["inpainting"] += inpaint_evidence.get("score", 0)
            
            # Check for style transfer patterns (img2img)
            style_evidence = self._detect_style_patterns(image, metadata)
            evidence.extend(style_evidence["evidence"])
            scores["img2img"] += style_evidence.get("score", 0)
            
            # Check model disagreement (high disagreement might indicate img2img)
            if model_scores:
                disagree_evidence = self._analyze_model_disagreement(model_scores)
                evidence.extend(disagree_evidence["evidence"])
                scores["img2img"] += disagree_evidence.get("img2img_score", 0)
            
            # Determine best guess
            max_type = max(scores, key=scores.get)
            max_score = scores[max_type]
            
            if max_score > 0.3:
                result["predicted_type"] = max_type
                result["guess"] = max_type  # Backward compat
                result["confidence"] = min(0.7, max_score)  # Cap confidence
            
            result["indicators"] = evidence[:5]  # For frontend
            result["evidence"] = evidence[:5]  # Backward compat
            
        except Exception as e:
            logger.warning(f"AI type classification error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _analyze_dimensions(self, image: Image.Image) -> Dict:
        """Analyze dimensions for text-to-image patterns"""
        evidence = []
        t2i_score = 0.0
        
        width, height = image.size
        
        # Common text-to-image sizes
        t2i_sizes = {
            (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
            (512, 768), (768, 512), (1024, 768), (768, 1024),
            (1024, 1792), (1792, 1024),  # DALL-E 3
            (1216, 832), (832, 1216),  # SDXL
        }
        
        if (width, height) in t2i_sizes:
            evidence.append(f"Dimensions {width}x{height} match common text-to-image sizes")
            t2i_score += 0.4
        
        # Perfect square is very common in t2i
        if width == height:
            evidence.append("Perfect square aspect ratio (common in text-to-image)")
            t2i_score += 0.2
        
        return {"evidence": evidence, "t2i_score": t2i_score}
    
    def _detect_inpainting_patterns(self, image: Image.Image) -> Dict:
        """Detect patterns suggesting inpainting"""
        evidence = []
        score = 0.0
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            gray = np.mean(img_array, axis=2)
            
            # Look for localized texture inconsistencies
            from scipy.ndimage import uniform_filter
            
            # Calculate local variance
            local_var = uniform_filter(gray**2, size=32) - uniform_filter(gray, size=32)**2
            
            # Check for regions with very different variance
            var_std = np.std(local_var)
            var_mean = np.mean(local_var)
            
            if var_std > var_mean * 0.8:
                # High variance in variance suggests mixed regions
                evidence.append("Localized texture inconsistencies detected")
                score += 0.25
            
            # Check for sharp boundaries in texture
            var_gradient = np.abs(np.diff(local_var, axis=0)).mean() + \
                          np.abs(np.diff(local_var, axis=1)).mean()
            
            if var_gradient > var_mean * 0.3:
                evidence.append("Sharp texture boundaries (possible inpainting edges)")
                score += 0.2
                
        except Exception as e:
            logger.warning(f"Inpainting detection error: {e}")
        
        return {"evidence": evidence, "score": score}
    
    def _detect_style_patterns(self, image: Image.Image, metadata: Dict) -> Dict:
        """Detect patterns suggesting img2img / style transfer"""
        evidence = []
        score = 0.0
        
        # Check if there's camera info but also AI-like characteristics
        camera = metadata.get("camera", {})
        if camera.get("has_camera_info"):
            # Has camera info but might be processed
            evidence.append("Camera metadata present - could be img2img from real photo")
            score += 0.15
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            
            # Check for stylization patterns
            # Stylized images often have reduced color palette
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_ratio = unique_colors / total_pixels
            
            if color_ratio < 0.1:
                evidence.append("Reduced color palette (possible stylization)")
                score += 0.2
            
            # Check for edge enhancement (common in style transfer)
            from scipy.ndimage import sobel
            gray = np.mean(img_array, axis=2)
            edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
            
            edge_ratio = np.sum(edges > 50) / edges.size
            if edge_ratio > 0.15:
                evidence.append("Enhanced edges (possible style transfer)")
                score += 0.15
                
        except Exception as e:
            logger.warning(f"Style detection error: {e}")
        
        return {"evidence": evidence, "score": score}
    
    def _analyze_model_disagreement(self, model_scores: List[Dict]) -> Dict:
        """Analyze model disagreement for img2img hints"""
        evidence = []
        img2img_score = 0.0
        
        if len(model_scores) < 2:
            return {"evidence": evidence, "img2img_score": img2img_score}
        
        scores = [s.get("raw_score", s.get("raw", 0.5)) for s in model_scores]
        disagreement = max(scores) - min(scores)
        
        # High disagreement might indicate img2img
        # (some models see "real" base, others see "AI" modifications)
        if disagreement > 0.4:
            evidence.append(f"High model disagreement ({disagreement:.2f}) - possible img2img")
            img2img_score += 0.25
        
        return {"evidence": evidence, "img2img_score": img2img_score}


def classify_ai_type(image: Image.Image, metadata: Dict, 
                    model_scores: List[Dict] = None) -> Dict[str, Any]:
    """Convenience function"""
    classifier = AITypeClassifier()
    return classifier.classify(image, metadata, model_scores)
