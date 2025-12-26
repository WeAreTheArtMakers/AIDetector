"""
Generation Pathway Classifier
==============================
Classifies image generation pathway:
- real_photo: Camera-captured photograph
- t2i: Text-to-Image generation (DALL-E, Midjourney, Stable Diffusion, Grok, etc.)
- i2i: Image-to-Image transformation (requires positive evidence of input image)
- inpainting: Localized region editing/generation
- stylization: Style transfer / artistic filter
- unknown: Insufficient evidence

Key principles:
- T2I is default for high diffusion score without transformation evidence
- I2I requires POSITIVE evidence of input image transformation
- Real photo requires camera/CFA evidence or low diffusion score
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PathwayType:
    """Generation pathway constants"""
    REAL_PHOTO = "real_photo"
    T2I = "t2i"  # Text-to-Image
    I2I = "i2i"  # Image-to-Image
    INPAINTING = "inpainting"
    STYLIZATION = "stylization"
    UNKNOWN = "unknown"


class GeneratorFamily:
    """Generator family constants (vendor-agnostic)"""
    DIFFUSION_T2I_MODERN = "diffusion_t2i_modern"  # Modern T2I (SDXL-class, closed-model photoreal)
    DIFFUSION_T2I_LEGACY = "diffusion_t2i_legacy"  # Legacy T2I (SD 1.x class)
    DIFFUSION_I2I = "diffusion_i2i"  # Image-to-Image transformations
    VENDOR_PHOTOREAL_STYLE = "vendor_photoreal_style"  # NEW: Popular closed-model photoreal family
    GAN_LEGACY = "gan_legacy"  # StyleGAN, etc.
    RENDER_3D = "render_3d"  # 3D renders, CGI
    CAMERA_PHOTO = "camera_photo"
    UNKNOWN = "unknown"


@dataclass
class PathwayResult:
    """Result of pathway classification"""
    pathway: str
    confidence: str
    probs: Dict[str, float]
    evidence: Dict[str, Any]
    generator_family: str
    generator_confidence: str


class PathwayClassifier:
    """
    Multi-signal classifier for generation pathway detection.
    
    Fuses signals from:
    - Diffusion fingerprint (frequency, residuals, patches)
    - Content type (photo vs non-photo)
    - Transformation evidence (for I2I)
    - Camera evidence (CFA, PRNU proxy)
    - Metadata signals
    """
    
    def __init__(self):
        # Thresholds
        self.diffusion_high_threshold = 0.70
        self.diffusion_medium_threshold = 0.50
        self.i2i_evidence_threshold = 0.60
        self.camera_evidence_threshold = 0.50
        
    def classify(self, 
                 image: Image.Image,
                 diffusion_result: Dict[str, Any],
                 content_type: Dict[str, Any],
                 manipulation_result: Dict[str, Any],
                 edit_assessment: Dict[str, Any],
                 metadata: Dict[str, Any],
                 domain: Dict[str, Any],
                 model_scores: List[Dict] = None) -> Dict[str, Any]:
        """
        Main classification entry point.
        
        Returns:
            Dict with pathway prediction, probabilities, and evidence
        """
        result = {
            "pred": PathwayType.UNKNOWN,
            "probs": {
                PathwayType.REAL_PHOTO: 0.0,
                PathwayType.T2I: 0.0,
                PathwayType.I2I: 0.0,
                PathwayType.INPAINTING: 0.0,
                PathwayType.STYLIZATION: 0.0,
                PathwayType.UNKNOWN: 1.0
            },
            "confidence": "low",
            "evidence": {
                "signals": [],
                "notes": [],
                "diffusion_score": 0.0,
                "i2i_evidence_score": 0.0,
                "camera_evidence_score": 0.0,
                "transformation_evidence": []
            },
            "generator_family": {
                "pred": GeneratorFamily.UNKNOWN,
                "confidence": "low",
                "top_candidates": []
            }
        }
        
        try:
            # Extract key signals
            diffusion_score = diffusion_result.get("diffusion_score", 0.0)
            result["evidence"]["diffusion_score"] = diffusion_score
            
            # 1. Compute I2I evidence score (transformation evidence)
            i2i_evidence = self._compute_i2i_evidence(
                image, manipulation_result, edit_assessment, 
                content_type, metadata, diffusion_result
            )
            result["evidence"]["i2i_evidence_score"] = i2i_evidence["score"]
            result["evidence"]["transformation_evidence"] = i2i_evidence["evidence"]
            
            # 2. Compute camera evidence score
            camera_evidence = self._compute_camera_evidence(
                image, metadata, diffusion_result
            )
            result["evidence"]["camera_evidence_score"] = camera_evidence["score"]
            
            # 3. Check for non-photo content (CGI/3D)
            non_photo_score = content_type.get("non_photo_score", 0.5)
            is_non_photo = (content_type.get("type") == "non_photo_like" and 
                          content_type.get("confidence") in ["high", "medium"])
            
            # 4. Check for inpainting evidence
            inpainting_score = self._compute_inpainting_score(
                manipulation_result, edit_assessment
            )
            
            # 5. Compute pathway probabilities
            probs = self._compute_pathway_probs(
                diffusion_score=diffusion_score,
                i2i_evidence_score=i2i_evidence["score"],
                camera_evidence_score=camera_evidence["score"],
                non_photo_score=non_photo_score,
                inpainting_score=inpainting_score,
                is_non_photo=is_non_photo,
                metadata=metadata
            )
            result["probs"] = probs
            
            # 6. Determine final prediction
            pred, confidence = self._determine_prediction(probs, result["evidence"])
            result["pred"] = pred
            result["confidence"] = confidence
            
            # 7. Classify generator family
            generator = self._classify_generator_family(
                pred, diffusion_score, diffusion_result, 
                content_type, metadata
            )
            result["generator_family"] = generator
            
            # 8. Build signal explanations
            result["evidence"]["signals"] = self._build_signals(
                diffusion_score, i2i_evidence, camera_evidence,
                inpainting_score, is_non_photo, pred
            )
            
            logger.info(f"Pathway: {pred} ({confidence}), "
                       f"diffusion={diffusion_score:.2f}, "
                       f"i2i_evidence={i2i_evidence['score']:.2f}, "
                       f"camera={camera_evidence['score']:.2f}")
            
        except Exception as e:
            logger.error(f"Pathway classification error: {e}")
            result["evidence"]["notes"].append(f"Error: {str(e)}")
        
        return result
    
    def _compute_i2i_evidence(self, image: Image.Image,
                              manipulation_result: Dict,
                              edit_assessment: Dict,
                              content_type: Dict,
                              metadata: Dict,
                              diffusion_result: Dict) -> Dict[str, Any]:
        """
        Compute I2I evidence score.
        
        I2I requires POSITIVE evidence of input image transformation:
        - Stylization edges along contours + photo base
        - Structure-preserving transformation patterns
        - Inpainting boundaries
        - Software hints (Photoshop + local manipulation)
        """
        evidence = []
        score = 0.0
        
        # 1. Check for stylization edge enhancement along contours
        # (I2I often enhances edges while preserving structure)
        if manipulation_result:
            edge_matte_score = manipulation_result.get("edge_matte", {}).get("score", 0)
            if edge_matte_score > 0.5:
                # Check if it's along semantic contours (not random)
                regions = manipulation_result.get("edge_matte", {}).get("regions", [])
                if len(regions) > 0 and len(regions) < 10:  # Localized, not everywhere
                    evidence.append("Edge enhancement along contours (possible I2I)")
                    score += 0.25
        
        # 2. Check for structure-preserving transformation
        # (Photo noise in background + stylized foreground)
        if edit_assessment:
            edit_type = edit_assessment.get("edit_type", "none_detected")
            if edit_type == "local_manipulation":
                local_score = edit_assessment.get("local_manipulation_score", 0)
                if local_score > 0.5:
                    evidence.append("Localized transformation detected")
                    score += 0.30
        
        # 3. Check for software hints suggesting I2I workflow
        software = metadata.get("software", {})
        software_name = software.get("name", "").lower()
        
        i2i_software = ["photoshop", "lightroom", "capture one", "affinity", 
                       "gimp", "pixelmator", "luminar"]
        if any(sw in software_name for sw in i2i_software):
            evidence.append(f"Editing software detected: {software_name}")
            score += 0.20
        
        # 4. Check for mixed photo/AI characteristics
        # (High diffusion score but also camera-like features)
        diffusion_score = diffusion_result.get("diffusion_score", 0)
        cfa_score = diffusion_result.get("channel_features", {}).get("cfa_artifact_score", 0)
        
        if diffusion_score > 0.5 and cfa_score > 0.2:
            evidence.append("Mixed AI/camera characteristics (possible I2I from photo)")
            score += 0.25
        
        # 5. Check content type for composite hints
        composite_score = content_type.get("composite_score", 0)
        if composite_score > 0.5:
            evidence.append("Composite/edited content detected")
            score += 0.15
        
        # Cap score
        score = min(1.0, score)
        
        return {"score": float(score), "evidence": evidence}
    
    def _compute_camera_evidence(self, image: Image.Image,
                                  metadata: Dict,
                                  diffusion_result: Dict) -> Dict[str, Any]:
        """
        Compute camera/real photo evidence score.
        
        Signals:
        - CFA/demosaicing artifacts
        - Camera metadata (make, model, settings)
        - GPS data
        - Natural noise patterns
        """
        evidence = []
        score = 0.0
        
        # 1. CFA artifacts (from diffusion fingerprint)
        cfa_score = diffusion_result.get("channel_features", {}).get("cfa_artifact_score", 0)
        if cfa_score > 0.3:
            evidence.append("CFA demosaicing artifacts detected")
            score += 0.30
        elif cfa_score > 0.15:
            evidence.append("Weak CFA patterns")
            score += 0.15
        
        # 2. Camera metadata
        camera = metadata.get("camera", {})
        if camera.get("has_camera_info"):
            evidence.append(f"Camera: {camera.get('make', '')} {camera.get('model', '')}")
            score += 0.25
            
            if camera.get("is_known_maker"):
                score += 0.10
        
        # 3. GPS data
        gps = metadata.get("gps", {})
        if gps.get("present") and gps.get("latitude") is not None:
            evidence.append("GPS coordinates present")
            score += 0.15
        
        # 4. Exposure settings
        technical = metadata.get("technical", {})
        tech_fields = ["exposure_time", "f_number", "iso", "focal_length"]
        tech_count = sum(1 for f in tech_fields if technical.get(f))
        
        if tech_count >= 3:
            evidence.append(f"Camera settings present ({tech_count}/4)")
            score += 0.15
        
        # 5. Low diffusion score supports camera origin
        diffusion_score = diffusion_result.get("diffusion_score", 0)
        if diffusion_score < 0.3:
            evidence.append("Low diffusion fingerprint")
            score += 0.20
        
        # Cap score
        score = min(1.0, score)
        
        return {"score": float(score), "evidence": evidence}
    
    def _compute_inpainting_score(self, manipulation_result: Dict,
                                   edit_assessment: Dict) -> float:
        """Compute inpainting evidence score"""
        score = 0.0
        
        if not manipulation_result:
            return score
        
        # Check for localized manipulation with clear boundaries
        edit_type = edit_assessment.get("edit_type", "none_detected") if edit_assessment else "none_detected"
        
        if edit_type == "local_manipulation":
            local_score = edit_assessment.get("local_manipulation_score", 0)
            corroborated = len(edit_assessment.get("corroborated_methods", [])) >= 2
            
            if local_score > 0.7 and corroborated:
                score += 0.5
            elif local_score > 0.5:
                score += 0.25
        
        # Check for splice/noise boundaries
        splice_score = manipulation_result.get("splice", {}).get("score", 0)
        splice_regions = manipulation_result.get("splice", {}).get("regions", [])
        
        if splice_score > 0.6 and len(splice_regions) > 0 and len(splice_regions) < 5:
            # Localized splice regions suggest inpainting
            score += 0.3
        
        return min(1.0, score)
    
    def _compute_pathway_probs(self, diffusion_score: float,
                                i2i_evidence_score: float,
                                camera_evidence_score: float,
                                non_photo_score: float,
                                inpainting_score: float,
                                is_non_photo: bool,
                                metadata: Dict) -> Dict[str, float]:
        """
        Compute pathway probabilities using signal fusion.
        
        CRITICAL I2I GATING:
        - I2I requires i2i_evidence_score >= 0.60 with positive transform evidence
        - If diffusion_score >= 0.60 and I2I gate fails => T2I
        """
        probs = {
            PathwayType.REAL_PHOTO: 0.0,
            PathwayType.T2I: 0.0,
            PathwayType.I2I: 0.0,
            PathwayType.INPAINTING: 0.0,
            PathwayType.STYLIZATION: 0.0,
            PathwayType.UNKNOWN: 0.0
        }
        
        # === REAL PHOTO ===
        # High camera evidence + low diffusion
        if camera_evidence_score > 0.5 and diffusion_score < 0.4:
            probs[PathwayType.REAL_PHOTO] = 0.7 + camera_evidence_score * 0.2
        elif camera_evidence_score > 0.3 and diffusion_score < 0.5:
            probs[PathwayType.REAL_PHOTO] = 0.4 + camera_evidence_score * 0.3
        elif diffusion_score < 0.3:
            probs[PathwayType.REAL_PHOTO] = 0.3 + (1 - diffusion_score) * 0.3
        
        # === T2I (Text-to-Image) ===
        # High diffusion + NO I2I evidence (or I2I evidence below threshold)
        if diffusion_score >= self.diffusion_high_threshold:
            if i2i_evidence_score < self.i2i_evidence_threshold:
                # Strong T2I signal - I2I gate failed
                probs[PathwayType.T2I] = 0.7 + diffusion_score * 0.25
            else:
                # Some T2I but I2I evidence present
                probs[PathwayType.T2I] = 0.3 + diffusion_score * 0.2
        elif diffusion_score >= self.diffusion_medium_threshold:
            if i2i_evidence_score < 0.4:
                probs[PathwayType.T2I] = 0.5 + diffusion_score * 0.3
            elif i2i_evidence_score < self.i2i_evidence_threshold:
                # Medium diffusion, some I2I hints but not enough
                probs[PathwayType.T2I] = 0.4 + diffusion_score * 0.2
        elif diffusion_score >= 0.40:
            # Lower diffusion but still possible T2I
            if i2i_evidence_score < 0.3:
                probs[PathwayType.T2I] = 0.3 + diffusion_score * 0.2
        
        # === I2I (Image-to-Image) ===
        # CRITICAL: Requires i2i_evidence_score >= 0.60 (positive transform evidence)
        if i2i_evidence_score >= self.i2i_evidence_threshold:
            if diffusion_score > 0.4:
                probs[PathwayType.I2I] = 0.5 + i2i_evidence_score * 0.3
            else:
                probs[PathwayType.I2I] = 0.3 + i2i_evidence_score * 0.2
        # If I2I evidence is below threshold, do NOT assign I2I probability
        # This prevents false I2I labels on T2I content
        
        # === INPAINTING ===
        if inpainting_score > 0.5:
            probs[PathwayType.INPAINTING] = 0.4 + inpainting_score * 0.4
        
        # === STYLIZATION ===
        # I2I-like but with style transfer characteristics
        # Only if I2I evidence is present
        if i2i_evidence_score >= 0.4 and diffusion_score > 0.3:
            style_hints = i2i_evidence_score * 0.5 + diffusion_score * 0.3
            if style_hints > 0.5:
                probs[PathwayType.STYLIZATION] = style_hints * 0.5
        
        # === NON-PHOTO ADJUSTMENT ===
        # If clearly non-photo (CGI/3D), boost T2I or reduce real_photo
        if is_non_photo:
            probs[PathwayType.REAL_PHOTO] *= 0.3
            if probs[PathwayType.T2I] > 0:
                probs[PathwayType.T2I] *= 1.2
        
        # === NORMALIZE ===
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs[PathwayType.UNKNOWN] = 1.0
        
        # Round for cleaner output
        probs = {k: round(v, 4) for k, v in probs.items()}
        
        return probs
    
    def _determine_prediction(self, probs: Dict[str, float],
                               evidence: Dict) -> Tuple[str, str]:
        """
        Determine final prediction and confidence.
        
        CRITICAL I2I GATING:
        - I2I cannot be top-1 unless i2i_evidence_score >= 0.60
        - If diffusion_score >= 0.60 and I2I gate fails => pathway = T2I
        """
        
        # Find top prediction
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        top_pred, top_prob = sorted_probs[0]
        second_pred, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
        
        # Determine confidence
        margin = top_prob - second_prob
        
        if top_prob >= 0.7 and margin >= 0.3:
            confidence = "high"
        elif top_prob >= 0.5 and margin >= 0.15:
            confidence = "medium"
        elif top_prob >= 0.35:
            confidence = "low"
        else:
            top_pred = PathwayType.UNKNOWN
            confidence = "low"
        
        # === CRITICAL FIX: I2I requires positive evidence ===
        # I2I cannot be top-1 unless i2i_evidence_score >= 0.60
        if top_pred == PathwayType.I2I:
            i2i_evidence_score = evidence.get("i2i_evidence_score", 0)
            diffusion_score = evidence.get("diffusion_score", 0)
            
            if i2i_evidence_score < self.i2i_evidence_threshold:
                # Not enough I2I evidence - fall back to T2I or unknown
                # If diffusion_score >= 0.60, use T2I (not I2I)
                if diffusion_score >= 0.60:
                    top_pred = PathwayType.T2I
                    confidence = "high" if diffusion_score >= 0.75 else "medium"
                elif diffusion_score >= self.diffusion_medium_threshold:
                    top_pred = PathwayType.T2I
                    confidence = "low"
                else:
                    top_pred = PathwayType.UNKNOWN
                    confidence = "low"
        
        # === ADDITIONAL CHECK: High diffusion without I2I evidence => T2I ===
        # Even if I2I wasn't top-1, ensure we don't miss T2I
        if top_pred == PathwayType.UNKNOWN:
            diffusion_score = evidence.get("diffusion_score", 0)
            i2i_evidence_score = evidence.get("i2i_evidence_score", 0)
            
            if diffusion_score >= 0.60 and i2i_evidence_score < self.i2i_evidence_threshold:
                top_pred = PathwayType.T2I
                confidence = "medium" if diffusion_score >= 0.70 else "low"
        
        return top_pred, confidence
    
    def _classify_generator_family(self, pathway: str,
                                    diffusion_score: float,
                                    diffusion_result: Dict,
                                    content_type: Dict,
                                    metadata: Dict) -> Dict[str, Any]:
        """
        Classify generator family (vendor-agnostic).
        Uses photo_like_generated_hint for photoreal AI detection.
        """
        result = {
            "pred": GeneratorFamily.UNKNOWN,
            "confidence": "low",
            "top_candidates": []
        }
        
        # Real photo
        if pathway == PathwayType.REAL_PHOTO:
            result["pred"] = GeneratorFamily.CAMERA_PHOTO
            result["confidence"] = "high"
            result["top_candidates"] = [GeneratorFamily.CAMERA_PHOTO]
            return result
        
        # Non-photo (CGI/3D)
        if content_type.get("type") == "non_photo_like":
            non_photo_score = content_type.get("non_photo_score", 0)
            if non_photo_score > 0.7:
                result["pred"] = GeneratorFamily.RENDER_3D
                result["confidence"] = "medium"
                result["top_candidates"] = [GeneratorFamily.RENDER_3D, 
                                           GeneratorFamily.DIFFUSION_T2I_MODERN]
                return result
        
        # NEW: Check for photoreal AI (popular closed-model family)
        photo_like_generated_hint = content_type.get("photo_like_generated_hint", 0)
        if photo_like_generated_hint >= 0.55 and pathway == PathwayType.T2I:
            # High photoreal AI hint + T2I pathway = vendor photoreal style
            result["pred"] = GeneratorFamily.VENDOR_PHOTOREAL_STYLE
            result["confidence"] = "medium" if photo_like_generated_hint >= 0.65 else "low"
            result["top_candidates"] = [GeneratorFamily.VENDOR_PHOTOREAL_STYLE,
                                       GeneratorFamily.DIFFUSION_T2I_MODERN]
            return result
        
        # T2I classification
        if pathway == PathwayType.T2I:
            # Check for modern vs legacy diffusion
            spectrum_features = diffusion_result.get("spectrum_features", {})
            roll_off = spectrum_features.get("roll_off_slope", 0)
            
            # Modern diffusion has steeper roll-off
            if diffusion_score >= 0.7:
                if roll_off < -2.8:
                    result["pred"] = GeneratorFamily.DIFFUSION_T2I_MODERN
                    result["confidence"] = "high"
                else:
                    result["pred"] = GeneratorFamily.DIFFUSION_T2I_MODERN
                    result["confidence"] = "medium"
                result["top_candidates"] = [GeneratorFamily.DIFFUSION_T2I_MODERN]
                
                # If also has photoreal hint, add vendor_photoreal_style as candidate
                if photo_like_generated_hint >= 0.40:
                    result["top_candidates"].insert(0, GeneratorFamily.VENDOR_PHOTOREAL_STYLE)
                    
            elif diffusion_score >= 0.5:
                result["pred"] = GeneratorFamily.DIFFUSION_T2I_MODERN
                result["confidence"] = "low"
                result["top_candidates"] = [GeneratorFamily.DIFFUSION_T2I_MODERN,
                                           GeneratorFamily.DIFFUSION_T2I_LEGACY]
        
        # I2I classification
        elif pathway == PathwayType.I2I:
            result["pred"] = GeneratorFamily.DIFFUSION_I2I
            result["confidence"] = "medium"
            result["top_candidates"] = [GeneratorFamily.DIFFUSION_I2I]
        
        return result
    
    def _build_signals(self, diffusion_score: float,
                       i2i_evidence: Dict,
                       camera_evidence: Dict,
                       inpainting_score: float,
                       is_non_photo: bool,
                       pred: str) -> List[str]:
        """Build human-readable signal explanations"""
        signals = []
        
        if diffusion_score >= 0.7:
            signals.append(f"Strong diffusion fingerprint ({diffusion_score:.0%})")
        elif diffusion_score >= 0.5:
            signals.append(f"Moderate diffusion fingerprint ({diffusion_score:.0%})")
        elif diffusion_score < 0.3:
            signals.append(f"Low diffusion fingerprint ({diffusion_score:.0%})")
        
        if i2i_evidence["score"] >= 0.6:
            signals.append(f"I2I transformation evidence ({i2i_evidence['score']:.0%})")
            signals.extend(i2i_evidence["evidence"][:2])
        elif i2i_evidence["score"] < 0.3:
            signals.append("No significant I2I transformation evidence")
        
        if camera_evidence["score"] >= 0.5:
            signals.append(f"Camera origin evidence ({camera_evidence['score']:.0%})")
            signals.extend(camera_evidence["evidence"][:2])
        
        if inpainting_score >= 0.5:
            signals.append(f"Inpainting evidence ({inpainting_score:.0%})")
        
        if is_non_photo:
            signals.append("Non-photo content (CGI/3D/Illustration)")
        
        return signals[:8]  # Limit to 8 signals


# Global instance
_pathway_classifier: Optional[PathwayClassifier] = None

def get_pathway_classifier() -> PathwayClassifier:
    global _pathway_classifier
    if _pathway_classifier is None:
        _pathway_classifier = PathwayClassifier()
    return _pathway_classifier
