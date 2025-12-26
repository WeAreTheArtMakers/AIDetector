"""
Edit Assessment Module - Global vs Local Edit Taxonomy
=======================================================
Separates:
- True composite/splice edits (region-based manipulation)
- Global post-processing (Instagram filters, color grading, denoise/sharpen)
- Platform re-encoding / resizing / screenshot artifacts
- Non-photo / 3D render / digital artwork (NOT composite)

Key principles:
- "local_manipulation" requires multi-method corroboration
- Border/corner artifacts are suppressed
- Globality test prevents false splice detection from global filters
- Non-photo content overrides composite labels
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class EditType:
    """Edit type classification constants"""
    NONE_DETECTED = "none_detected"
    GLOBAL_POSTPROCESS = "global_postprocess"
    LOCAL_MANIPULATION = "local_manipulation"
    GENERATOR_ARTIFACTS = "generator_artifacts"  # AI generation artifacts (not manipulation)


class EditAssessor:
    """
    Assesses edit type by analyzing manipulation detection results
    with globality tests, border suppression, and corroboration requirements.
    """
    
    def __init__(self):
        # Thresholds
        self.globality_threshold = 0.30  # If >30% of image is "suspicious", it's global (lowered from 0.35)
        self.border_margin_ratio = 0.04  # 4% border margin to suppress (3-5% range)
        self.local_manipulation_threshold = 0.65  # Score threshold for local manipulation
        self.corroboration_threshold = 0.50  # Minimum score for corroborating method
        self.stability_tolerance = 0.30  # Max shift ratio for stable detection
        
        # Boundary-type corroboration thresholds (for local_manipulation)
        # CRITICAL: Only edge_matte, copy_move, or inpainting_boundary count
        # noise_inconsistency and blur_mismatch are NOT boundary evidence
        self.boundary_corroboration_threshold = 0.60  # edge_matte, copy_move, or inpainting_boundary
        
    def assess(self, 
               image: Image.Image,
               manipulation_result: Dict[str, Any],
               domain: Dict[str, Any],
               content_type: Dict[str, Any],
               metadata: Dict[str, Any],
               pathway_result: Dict[str, Any] = None,
               diffusion_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main assessment entry point.
        
        Returns:
            Dict with edit_type, confidence, scores, and reasons
        """
        result = {
            "edit_type": EditType.NONE_DETECTED,
            "confidence": "low",
            "reasons": [],
            "global_adjustment_score": 0.0,
            "local_manipulation_score": 0.0,
            "generator_artifacts_score": 0.0,  # NEW: AI generation artifacts
            "globality_ratio": 0.0,
            "border_suppressed_count": 0,
            "corroborated_methods": [],
            "boundary_corroborated": False,  # NEW: boundary-type corroboration
            "suppressed_regions": [],
            "filtered_regions": [],  # Regions after filtering
            "raw_regions": [],  # Original regions before filtering
        }
        
        if not manipulation_result or not manipulation_result.get("enabled"):
            return result
        
        try:
            img_array = np.array(image.convert('RGB'))
            h, w = img_array.shape[:2]
            
            # Step 1: Collect all raw regions
            raw_regions = self._collect_all_regions(manipulation_result)
            result["raw_regions"] = raw_regions
            
            # Step 2: Apply border suppression
            filtered_regions, border_suppressed = self._apply_border_suppression(
                raw_regions, w, h
            )
            result["border_suppressed_count"] = border_suppressed
            
            # Step 3: Compute globality ratio
            globality = self._compute_globality(filtered_regions, w, h)
            result["globality_ratio"] = round(float(globality), 3)
            
            # Step 4: Check for platform re-encoding artifacts
            is_reencoded = self._check_reencoding(domain, metadata)
            
            # Step 5: Check for non-photo content (CGI/3D/illustration)
            is_non_photo = self._check_non_photo(content_type)
            
            # Step 5.5: Check for AI-generated content (T2I pathway)
            is_ai_generated = self._check_ai_generated(pathway_result, diffusion_result)
            
            # Step 6: Apply re-encoding aware thresholds
            effective_threshold = self.local_manipulation_threshold
            if is_reencoded:
                effective_threshold += 0.15  # Raise threshold for recompressed images
                result["reasons"].append("reencoding_threshold_raised")
            
            # Step 7: Check multi-method corroboration
            corroborated, corroborated_methods = self._check_corroboration(
                manipulation_result, effective_threshold
            )
            result["corroborated_methods"] = corroborated_methods
            
            # Step 7.5: Check boundary-type corroboration (CRITICAL for local_manipulation)
            boundary_corroborated = self._check_boundary_corroboration(manipulation_result)
            result["boundary_corroborated"] = boundary_corroborated
            
            # Step 8: Stability test (if we have enough regions)
            is_stable = True
            if len(filtered_regions) >= 3:
                is_stable = self._check_stability(image, manipulation_result)
                if not is_stable:
                    result["reasons"].append("unstable_detection")
            
            # Step 9: Compute scores
            global_score = self._compute_global_score(
                manipulation_result, globality, is_reencoded, metadata
            )
            local_score = self._compute_local_score(
                filtered_regions, corroborated, is_stable, effective_threshold,
                boundary_corroborated  # NEW: require boundary corroboration
            )
            generator_artifacts_score = self._compute_generator_artifacts_score(
                manipulation_result, is_ai_generated, globality, 
                boundary_corroborated, diffusion_result
            )
            
            result["global_adjustment_score"] = round(float(global_score), 3)
            result["local_manipulation_score"] = round(float(local_score), 3)
            result["generator_artifacts_score"] = round(float(generator_artifacts_score), 3)
            result["filtered_regions"] = filtered_regions
            
            # Step 10: Determine edit type
            edit_type, confidence, reasons = self._determine_edit_type(
                global_score=global_score,
                local_score=local_score,
                generator_artifacts_score=generator_artifacts_score,
                globality=globality,
                corroborated=corroborated,
                boundary_corroborated=boundary_corroborated,
                is_stable=is_stable,
                is_non_photo=is_non_photo,
                is_ai_generated=is_ai_generated,
                is_reencoded=is_reencoded,
                filtered_regions=filtered_regions
            )
            
            result["edit_type"] = edit_type
            result["confidence"] = confidence
            result["reasons"].extend(reasons)
            
            logger.info(f"Edit assessment: {edit_type} ({confidence}) - "
                       f"global={global_score:.2f}, local={local_score:.2f}, "
                       f"gen_artifacts={generator_artifacts_score:.2f}, "
                       f"globality={globality:.2f}, corroborated={corroborated}, "
                       f"boundary_corroborated={boundary_corroborated}")
            
        except Exception as e:
            logger.error(f"Edit assessment error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _collect_all_regions(self, manipulation_result: Dict) -> List[Dict]:
        """Collect all detected regions from all methods"""
        all_regions = []
        for method in ["splice", "copy_move", "edge_matte", "blur_noise_mismatch"]:
            method_data = manipulation_result.get(method, {})
            regions = method_data.get("regions", [])
            for r in regions:
                all_regions.append({
                    **r,
                    "source_method": method
                })
        return all_regions
    
    def _apply_border_suppression(self, regions: List[Dict], 
                                   width: int, height: int) -> Tuple[List[Dict], int]:
        """
        Suppress regions that are primarily in border/corner areas.
        Many false positives occur at (0,0), borders, and uniform backgrounds.
        """
        margin_x = int(width * self.border_margin_ratio)
        margin_y = int(height * self.border_margin_ratio)
        
        filtered = []
        suppressed = 0
        
        for r in regions:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            
            # Check if region is primarily in border area
            in_left_border = x < margin_x
            in_right_border = (x + w) > (width - margin_x)
            in_top_border = y < margin_y
            in_bottom_border = (y + h) > (height - margin_y)
            
            # Calculate overlap with border
            border_overlap = 0
            region_area = w * h
            
            if in_left_border:
                border_overlap += min(margin_x - x, w) * h
            if in_right_border:
                border_overlap += min((x + w) - (width - margin_x), w) * h
            if in_top_border:
                border_overlap += min(margin_y - y, h) * w
            if in_bottom_border:
                border_overlap += min((y + h) - (height - margin_y), h) * w
            
            border_ratio = border_overlap / (region_area + 1e-6)
            
            # Suppress if >60% of region is in border
            if border_ratio > 0.60:
                suppressed += 1
                r["suppressed"] = True
                r["suppression_reason"] = "border_artifact"
            else:
                r["suppressed"] = False
                filtered.append(r)
        
        return filtered, suppressed
    
    def _compute_globality(self, regions: List[Dict], 
                           width: int, height: int) -> float:
        """
        Compute globality ratio - what fraction of image has suspicious regions.
        High globality suggests global processing, not local splice.
        """
        if not regions:
            return 0.0
        
        # Create coverage map
        coverage = np.zeros((height, width), dtype=np.float32)
        
        for r in regions:
            x1 = max(0, r["x"])
            y1 = max(0, r["y"])
            x2 = min(width, r["x"] + r["w"])
            y2 = min(height, r["y"] + r["h"])
            
            score = r.get("score", 0.5)
            coverage[y1:y2, x1:x2] = np.maximum(coverage[y1:y2, x1:x2], score)
        
        # Compute ratio of covered pixels
        threshold = 0.3
        covered_pixels = np.sum(coverage > threshold)
        total_pixels = width * height
        
        globality = covered_pixels / total_pixels
        
        return globality
    
    def _check_reencoding(self, domain: Dict, metadata: Dict) -> bool:
        """Check if image has been re-encoded (social media, screenshot, etc.)"""
        characteristics = domain.get("characteristics", [])
        domain_type = domain.get("type", "unknown")
        
        reencoding_signals = [
            "recompressed" in characteristics,
            "resized" in characteristics,
            "screenshot" in characteristics,
            "social_media" in characteristics,
            domain_type in ["recompressed", "screenshot", "social_media"],
        ]
        
        # Check software metadata for social media
        software = metadata.get("software", {})
        software_name = software.get("name", "").lower()
        social_apps = ["instagram", "whatsapp", "facebook", "twitter", "tiktok", "snapchat"]
        if any(app in software_name for app in social_apps):
            reencoding_signals.append(True)
        
        return any(reencoding_signals)
    
    def _check_non_photo(self, content_type: Dict) -> bool:
        """Check if content is non-photo (CGI/3D/illustration)"""
        ct_type = content_type.get("type", "unknown")
        ct_confidence = content_type.get("confidence", "low")
        non_photo_score = content_type.get("non_photo_score", 0.5)
        
        return (
            (ct_type == "non_photo_like" and ct_confidence in ["high", "medium"]) or
            non_photo_score >= 0.70
        )
    
    def _check_ai_generated(self, pathway_result: Dict, diffusion_result: Dict) -> bool:
        """
        Check if content is AI-generated (T2I pathway with high diffusion score).
        
        This is used to distinguish AI generation artifacts from true manipulation.
        """
        if not pathway_result and not diffusion_result:
            return False
        
        # Check pathway classification - T2I with medium+ confidence
        if pathway_result:
            pathway_pred = pathway_result.get("pred", "unknown")
            pathway_conf = pathway_result.get("confidence", "low")
            
            # T2I pathway is strong signal
            if pathway_pred == "t2i" and pathway_conf in ["high", "medium"]:
                return True
            
            # Also check if pathway is T2I with low confidence but high diffusion
            if pathway_pred == "t2i" and pathway_conf == "low":
                diffusion_score = pathway_result.get("evidence", {}).get("diffusion_score", 0)
                if diffusion_score >= 0.60:
                    return True
        
        # Check diffusion fingerprint directly
        if diffusion_result:
            diffusion_score = diffusion_result.get("diffusion_score", 0)
            # High diffusion score = likely AI-generated
            if diffusion_score >= 0.60:
                return True
        
        return False
    
    def _check_boundary_corroboration(self, manipulation_result: Dict) -> bool:
        """
        Check for boundary-type corroboration.
        
        CRITICAL: local_manipulation requires at least one of:
        - edge_matte >= 0.60 (matting/compositing boundary)
        - copy_move >= 0.60 (duplicated region boundary)
        - inpainting_boundary >= 0.60 (localized splice with clear edges AND edge_matte support)
        
        EXPLICITLY NOT boundary evidence:
        - noise_inconsistency (splice score from noise analysis alone)
        - blur_mismatch (blur_noise_mismatch score)
        
        These are texture-level signals that can be triggered by:
        - AI generation artifacts
        - Recompression
        - Global filters
        
        Without boundary evidence, use generator_artifacts or global_postprocess instead.
        """
        if not manipulation_result:
            return False
        
        # BOUNDARY-TYPE METHODS ONLY
        edge_matte_score = manipulation_result.get("edge_matte", {}).get("score", 0)
        copy_move_score = manipulation_result.get("copy_move", {}).get("score", 0)
        
        # Direct boundary methods
        if edge_matte_score >= self.boundary_corroboration_threshold:
            return True
        
        if copy_move_score >= self.boundary_corroboration_threshold:
            return True
        
        # Inpainting boundary: requires BOTH splice evidence AND edge_matte support
        # Splice alone is NOT boundary evidence (it's noise-based)
        splice_data = manipulation_result.get("splice", {})
        splice_score = splice_data.get("score", 0)
        splice_regions = splice_data.get("regions", [])
        
        # Inpainting boundary requires:
        # 1. High splice score with localized regions
        # 2. AND some edge_matte support (>= 0.40)
        # This prevents false positives from noise-only detection
        if splice_score >= 0.70 and 0 < len(splice_regions) <= 3:
            # Check if there's supporting edge_matte evidence
            if edge_matte_score >= 0.40:
                # Check if regions are localized and high-scoring
                high_score_regions = [r for r in splice_regions if r.get("score", 0) >= 0.65]
                if len(high_score_regions) >= 1:
                    return True
        
        # No boundary evidence found
        return False
    
    def _check_corroboration(self, manipulation_result: Dict, 
                              threshold: float) -> Tuple[bool, List[str]]:
        """
        Check if multiple independent methods agree on manipulation.
        Single-method high score is "suspicious" but not "likely composite".
        """
        methods_above_threshold = []
        
        method_scores = {
            "splice": manipulation_result.get("splice", {}).get("score", 0),
            "copy_move": manipulation_result.get("copy_move", {}).get("score", 0),
            "edge_matte": manipulation_result.get("edge_matte", {}).get("score", 0),
            "blur_noise_mismatch": manipulation_result.get("blur_noise_mismatch", {}).get("score", 0)
        }
        
        for method, score in method_scores.items():
            if score >= threshold:
                methods_above_threshold.append(method)
        
        # Copy-move alone is strong evidence (it's very specific)
        if "copy_move" in methods_above_threshold and method_scores["copy_move"] >= 0.70:
            return True, methods_above_threshold
        
        # Otherwise require at least 2 methods
        corroborated = len(methods_above_threshold) >= 2
        
        return corroborated, methods_above_threshold
    
    def _check_stability(self, image: Image.Image, 
                         manipulation_result: Dict) -> bool:
        """
        Check if detection is stable across scales.
        If hotspots move wildly when downsampled, detection is unreliable.
        
        Multi-scale stability test:
        - Regions should be clustered (not scattered)
        - High spread = unstable = likely global artifacts, not local manipulation
        """
        regions = self._collect_all_regions(manipulation_result)
        
        if len(regions) < 2:
            return True
        
        # Check if regions are clustered (stable) or scattered (unstable)
        xs = [r["x"] + r["w"]/2 for r in regions]
        ys = [r["y"] + r["h"]/2 for r in regions]
        
        img_w, img_h = image.size
        
        # Compute normalized spread
        x_spread = np.std(xs) / (img_w + 1e-6)
        y_spread = np.std(ys) / (img_h + 1e-6)
        avg_spread = (x_spread + y_spread) / 2
        
        # Also check if regions cover too much of the image (globality proxy)
        total_region_area = sum(r["w"] * r["h"] for r in regions)
        image_area = img_w * img_h
        coverage_ratio = total_region_area / (image_area + 1e-6)
        
        # Unstable if:
        # 1. Regions are very scattered (high spread)
        # 2. Regions cover too much of the image (suggests global, not local)
        if avg_spread >= 0.35:
            return False  # Too scattered
        
        if coverage_ratio >= 0.40:
            return False  # Too much coverage = likely global
        
        # Check for "noise-like" distribution (many small scattered regions)
        small_regions = [r for r in regions if r["w"] * r["h"] < (img_w * img_h * 0.01)]
        if len(small_regions) > 5 and len(small_regions) / len(regions) > 0.7:
            return False  # Many small scattered regions = unstable
        
        return True
    
    def _compute_global_score(self, manipulation_result: Dict,
                               globality: float, is_reencoded: bool,
                               metadata: Dict) -> float:
        """
        Compute global adjustment score.
        High when evidence suggests global processing (filters, color grading).
        """
        score = 0.0
        
        # High globality suggests global processing
        if globality > self.globality_threshold:
            score += 0.4
        elif globality > 0.20:
            score += 0.2
        
        # Re-encoding suggests platform processing
        if is_reencoded:
            score += 0.3
        
        # Check for software that applies global edits
        software = metadata.get("software", {})
        software_name = software.get("name", "").lower()
        
        global_edit_apps = ["instagram", "vsco", "snapseed", "lightroom", 
                           "photoshop", "camera raw", "capture one"]
        if any(app in software_name for app in global_edit_apps):
            score += 0.2
        
        # High noise variance across image suggests global processing
        splice_notes = manipulation_result.get("splice", {}).get("notes", [])
        if any("High noise variance" in n for n in splice_notes):
            score += 0.1
        
        return min(1.0, score)
    
    def _compute_local_score(self, filtered_regions: List[Dict],
                              corroborated: bool, is_stable: bool,
                              threshold: float,
                              boundary_corroborated: bool = False) -> float:
        """
        Compute local manipulation score.
        Only high if evidence is strong, corroborated, stable, AND boundary-corroborated.
        
        CRITICAL: local_manipulation requires boundary-type corroboration
        (edge_matte, copy_move, or inpainting_boundary >= 0.60)
        """
        if not filtered_regions:
            return 0.0
        
        # Base score from max region score
        max_score = max(r.get("score", 0) for r in filtered_regions)
        
        # Penalize if not corroborated
        if not corroborated:
            max_score *= 0.5
        
        # Penalize if unstable
        if not is_stable:
            max_score *= 0.6
        
        # CRITICAL: Penalize heavily if no boundary-type corroboration
        # Noise inconsistency + blur mismatch alone should NOT trigger local_manipulation
        if not boundary_corroborated:
            max_score *= 0.4  # Strong penalty
        
        # Penalize if only border-adjacent regions remain
        non_edge_regions = [r for r in filtered_regions 
                           if r.get("x", 0) > 50 and r.get("y", 0) > 50]
        if len(non_edge_regions) == 0 and len(filtered_regions) > 0:
            max_score *= 0.4
        
        return max_score
    
    def _compute_generator_artifacts_score(self, manipulation_result: Dict,
                                            is_ai_generated: bool,
                                            globality: float,
                                            boundary_corroborated: bool,
                                            diffusion_result: Dict) -> float:
        """
        Compute generator artifacts score.
        
        High when:
        - AI-generated content (T2I pathway or high diffusion score)
        - Noise/texture inconsistencies WITHOUT boundary corroboration
        - High globality (artifacts spread across image)
        
        This distinguishes AI generation artifacts from true local manipulation.
        
        CRITICAL: If AI-generated and no boundary corroboration, this should be HIGH
        to prevent false "composite" labels on T2I portraits.
        """
        score = 0.0
        
        if not manipulation_result:
            return score
        
        splice_score = manipulation_result.get("splice", {}).get("score", 0)
        blur_score = manipulation_result.get("blur_noise_mismatch", {}).get("score", 0)
        overall_manip = manipulation_result.get("overall_score", 0)
        
        # Get diffusion score
        diffusion_score = 0.0
        if diffusion_result:
            diffusion_score = diffusion_result.get("diffusion_score", 0)
        
        # CASE 1: AI-generated content with manipulation signals but NO boundary corroboration
        # This is the key case for T2I portraits with false-positive manipulation detection
        if is_ai_generated and not boundary_corroborated:
            # Noise/blur inconsistencies in AI content = generator artifacts
            if splice_score >= 0.30 or blur_score >= 0.30:
                score += 0.5
            
            # High overall manipulation score without boundary = generator artifacts
            if overall_manip >= 0.40:
                score += 0.3
            
            # High globality in AI content = generator artifacts (not splice)
            if globality > 0.25:
                score += 0.2
        
        # CASE 2: High diffusion score + manipulation signals without boundary
        if diffusion_score >= 0.55 and not boundary_corroborated:
            if overall_manip >= 0.30:
                score += 0.3
            
            # Specifically: noise inconsistency + blur mismatch = generator artifacts
            if splice_score >= 0.40 and blur_score >= 0.40:
                score += 0.2
        
        # CASE 3: High globality alone suggests generator artifacts (for AI content)
        if is_ai_generated and globality > 0.30:
            score += 0.2
        
        return min(1.0, score)
    
    def _determine_edit_type(self, global_score: float, local_score: float,
                              generator_artifacts_score: float,
                              globality: float, corroborated: bool,
                              boundary_corroborated: bool,
                              is_stable: bool, is_non_photo: bool,
                              is_ai_generated: bool,
                              is_reencoded: bool,
                              filtered_regions: List[Dict]) -> Tuple[str, str, List[str]]:
        """
        Determine final edit type classification.
        
        Priority:
        1. AI-generated content => generator_artifacts (unless boundary evidence)
        2. Non-photo content suppresses local manipulation claims
        3. High globality => global postprocess or generator_artifacts
        4. Local manipulation REQUIRES boundary corroboration (edge_matte/copy_move/inpainting_boundary)
        5. Without boundary evidence, NEVER output local_manipulation
        
        CRITICAL RULES:
        - noise_inconsistency + blur_mismatch alone => generator_artifacts or global_postprocess
        - local_manipulation requires boundary_corroborated == true
        - If boundary not present, never output "Likely Composite"
        """
        reasons = []
        
        # RULE 0: AI-generated content without boundary evidence => generator_artifacts
        # This is the FIRST check to prevent false composite on T2I portraits
        if is_ai_generated and not boundary_corroborated:
            if generator_artifacts_score >= 0.40 or globality > 0.25:
                reasons.append("ai_generated_no_boundary_evidence")
                confidence = "high" if generator_artifacts_score >= 0.60 else "medium"
                return EditType.GENERATOR_ARTIFACTS, confidence, reasons
            elif local_score >= 0.40:
                # Even with high local score, without boundary = generator artifacts
                reasons.append("ai_generated_local_signals_no_boundary")
                return EditType.GENERATOR_ARTIFACTS, "medium", reasons
        
        # RULE 1: Non-photo content suppresses local manipulation claims
        if is_non_photo:
            reasons.append("non_photo_content")
            if local_score < 0.85 or not corroborated or not boundary_corroborated:
                # Only allow local manipulation for non-photo if very strong evidence
                if generator_artifacts_score >= 0.40:
                    return EditType.GENERATOR_ARTIFACTS, "medium", reasons
                return EditType.NONE_DETECTED, "medium", reasons
        
        # RULE 2: High globality => global postprocess or generator_artifacts
        # Globality > 30% means artifacts are spread across image = not local splice
        if globality > self.globality_threshold:
            reasons.append(f"high_globality_{globality:.2f}")
            if is_ai_generated:
                return EditType.GENERATOR_ARTIFACTS, "medium", reasons
            if local_score < 0.80 or not boundary_corroborated:
                return EditType.GLOBAL_POSTPROCESS, "medium", reasons
        
        # RULE 3: Local manipulation REQUIRES boundary corroboration
        # This is the CRITICAL gate - without boundary evidence, no local_manipulation
        if local_score >= 0.70 and corroborated and is_stable and boundary_corroborated:
            reasons.append("corroborated_local_manipulation_with_boundary")
            confidence = "high" if local_score >= 0.85 else "medium"
            return EditType.LOCAL_MANIPULATION, confidence, reasons
        
        # RULE 4: High local score but MISSING boundary corroboration
        # This is the key fix - without boundary, use generator_artifacts or global
        if local_score >= 0.50 and not boundary_corroborated:
            reasons.append("no_boundary_corroboration")
            if is_ai_generated:
                return EditType.GENERATOR_ARTIFACTS, "medium", reasons
            if is_reencoded or global_score >= 0.40:
                return EditType.GLOBAL_POSTPROCESS, "low", reasons
            # Even without reencoding, don't claim local manipulation
            return EditType.NONE_DETECTED, "low", reasons
        
        # RULE 5: High local score but not corroborated => suspicious but not confirmed
        if local_score >= 0.60 and not corroborated:
            reasons.append("single_method_suspicious")
            if global_score >= 0.50:
                return EditType.GLOBAL_POSTPROCESS, "low", reasons
            return EditType.NONE_DETECTED, "low", reasons
        
        # RULE 6: Re-encoded with some signals => likely global postprocess
        if is_reencoded and (global_score >= 0.35 or len(filtered_regions) > 0):
            reasons.append("reencoded_with_artifacts")
            return EditType.GLOBAL_POSTPROCESS, "medium", reasons
        
        # RULE 7: Some global signals but no strong local evidence
        if global_score >= 0.45 and local_score < 0.50:
            reasons.append("global_signals_only")
            return EditType.GLOBAL_POSTPROCESS, "low", reasons
        
        # RULE 8: No significant evidence
        if global_score < 0.30 and local_score < 0.30 and generator_artifacts_score < 0.30:
            return EditType.NONE_DETECTED, "medium", reasons
        
        # Default: uncertain - but never local_manipulation without boundary
        reasons.append("uncertain_signals")
        return EditType.NONE_DETECTED, "low", reasons


# Global instance
_edit_assessor: Optional[EditAssessor] = None

def get_edit_assessor() -> EditAssessor:
    global _edit_assessor
    if _edit_assessor is None:
        _edit_assessor = EditAssessor()
    return _edit_assessor
