"""
Verdict Text Generator - UX Text Layer for Forensic AI Analysis
================================================================
Generates consistent, non-redundant verdict text based on analysis results.

Key principles:
- Two axes: content_type_axis + evidential_quality
- "BELİRSİZ/Inconclusive" only when truly_inconclusive
- Composite labels require evidence-backed triggers
- Single global footer + optional conditional banner (no ⚠️ spam)
- PROVENANCE (EXIF/GPS) must affect camera_score and verdict
- T2I pathway is evidence, NOT a verdict switch
- Consistency checker prevents contradictory headlines

CRITICAL RULES:
R1) Headline must never contradict the highest-weight evidence blocks
R2) Provenance signals (EXIF/GPS/time/software) must affect camera_score and verdict
R3) T2I/I2I pathway is NOT allowed to override strong camera provenance + camera pipeline evidence
R4) When evidence is mixed, verdict must be "Inconclusive" not "Likely AI" or "Likely Photo"
R5) "Belirsiz" is used ONLY when neither side reaches threshold after weighting
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerdictResult:
    """Result of verdict text generation"""
    verdict_key: str
    title_tr: str
    title_en: str
    subtitle_tr: str
    subtitle_en: str
    banner_key: Optional[str]
    banner_tr: Optional[str]
    banner_en: Optional[str]
    footer_key: str
    footer_tr: str
    footer_en: str
    tags: List[str]
    summary_axes: Dict[str, Any]


# Verdict key constants
class VerdictKey:
    NON_PHOTO_HIGH = "NON_PHOTO_HIGH"
    COMPOSITE_LIKELY = "COMPOSITE_LIKELY"
    COMPOSITE_POSSIBLE = "COMPOSITE_POSSIBLE"
    PHOTO_GLOBAL_EDIT = "PHOTO_GLOBAL_EDIT"  # Global edit (filter/grade)
    AI_T2I_HIGH = "AI_T2I_HIGH"  # Text-to-Image high confidence
    AI_T2I_MEDIUM = "AI_T2I_MEDIUM"  # Text-to-Image medium confidence
    AI_PHOTOREAL_HIGH = "AI_PHOTOREAL_HIGH"  # Photoreal AI (popular closed-model family)
    AI_PHOTOREAL_MEDIUM = "AI_PHOTOREAL_MEDIUM"  # Photoreal AI medium confidence
    AI_GENERATIVE_UNKNOWN = "AI_GENERATIVE_UNKNOWN"  # NEW: AI-generated but pathway unknown
    AI_HIGH = "AI_HIGH"
    AI_MEDIUM = "AI_MEDIUM"
    REAL_LIKELY = "REAL_LIKELY"
    REAL_MEDIUM = "REAL_MEDIUM"
    GENERATOR_ARTIFACTS = "GENERATOR_ARTIFACTS"  # AI generation artifacts (not manipulation)
    INCONCLUSIVE = "INCONCLUSIVE"


# Banner keys
class BannerKey:
    LOW_EVIDENCE = "BANNER_LOW_EVIDENCE"
    INCONCLUSIVE = "BANNER_INCONCLUSIVE"


# Footer key
FOOTER_KEY = "FOOTER_PROBABILISTIC"

# Footer text (always present) - UPDATED: Forensic-grade disclaimer
FOOTER_TR = "Bu rapor teknik sinyallere dayalı olasılıksal bir değerlendirmedir; adli bağlamda ek doğrulama (orijinal dosya, zincirleme delil, C2PA vb.) önerilir."
FOOTER_EN = "This report is a probabilistic assessment based on technical signals; for forensic use, additional verification (original file, chain-of-custody, C2PA, etc.) is recommended."

# Banner texts - UPDATED: Only show when truly needed
BANNERS = {
    BannerKey.LOW_EVIDENCE: {
        "tr": "Kanıt kalitesi düşük. Daha güvenilir sonuç için orijinal dosyayı yükleyin.",
        "en": "Evidence quality is low. Upload the original file for more reliable results."
    },
    BannerKey.INCONCLUSIVE: {
        "tr": "Sonuç belirsiz — karışık sinyaller. Daha net sonuç için orijinal dosya önerilir.",
        "en": "Result is inconclusive — mixed signals. Upload the original file for a clearer result."
    }
}


def _safe_get(d: Dict, *keys, default=None):
    """Safely get nested dict value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def _check_gps_consistency(report: Dict) -> Dict[str, Any]:
    """
    Check GPS data consistency and generate appropriate messaging.
    
    Returns:
        Dict with:
        - present: bool
        - consistency: "strong" | "weak" | "none"
        - message_tr: str
        - message_en: str
        - reasons: List[str]
    """
    result = {
        "present": False,
        "consistency": "none",
        "message_tr": "",
        "message_en": "",
        "reasons": []
    }
    
    gps = report.get("gps", {})
    metadata = report.get("metadata", {}) or report.get("metadata_extended", {})
    domain = report.get("domain", {})
    
    # Check if GPS is present
    if not gps.get("present", False) or gps.get("latitude") is None:
        result["message_tr"] = "GPS verisi yok"
        result["message_en"] = "No GPS data"
        return result
    
    result["present"] = True
    
    # Collect consistency signals
    consistency_score = 0
    
    # 1. Check if domain is original (not recompressed)
    if domain.get("type") == "original":
        consistency_score += 2
        result["reasons"].append("original_domain")
    elif domain.get("type") == "recompressed":
        consistency_score -= 1
        result["reasons"].append("recompressed_domain")
    
    # 2. Check for camera metadata
    camera = metadata.get("camera", {}) if isinstance(metadata, dict) else {}
    if camera.get("has_camera_info", False):
        consistency_score += 2
        result["reasons"].append("camera_info_present")
    
    # 3. Check for timestamps
    timestamps = metadata.get("timestamps", {}) if isinstance(metadata, dict) else {}
    if timestamps.get("datetime_original"):
        consistency_score += 1
        result["reasons"].append("datetime_original_present")
    
    # 4. Check for GPS altitude (more complete GPS data)
    if gps.get("altitude") is not None:
        consistency_score += 1
        result["reasons"].append("altitude_present")
    
    # 5. Check for EXIF completeness
    exif_count = metadata.get("exif_count", 0) if isinstance(metadata, dict) else 0
    if exif_count >= 20:
        consistency_score += 1
        result["reasons"].append("rich_exif")
    
    # 6. Check domain characteristics for social media (reduces trust)
    characteristics = domain.get("characteristics", [])
    if "social_media" in characteristics or "exif_stripped" in characteristics:
        consistency_score -= 2
        result["reasons"].append("social_media_or_stripped")
    
    # Determine consistency level
    if consistency_score >= 4:
        result["consistency"] = "strong"
        result["message_tr"] = "GPS (EXIF) bulundu ve kamera metadata ile tutarlı."
        result["message_en"] = "GPS (EXIF) found and consistent with camera metadata."
    elif consistency_score >= 1:
        result["consistency"] = "weak"
        result["message_tr"] = "GPS (EXIF) bulundu. EXIF alanları değiştirilebilir; GPS tek başına nihai kanıt değildir, ancak güçlü bir bağlam ipucudur."
        result["message_en"] = "GPS (EXIF) found. EXIF fields can be modified; GPS alone is not definitive proof, but provides strong contextual evidence."
    else:
        result["consistency"] = "weak"
        result["message_tr"] = "GPS (EXIF) bulundu ancak doğrulama sınırlı (metadata eksik veya yeniden işlenmiş)."
        result["message_en"] = "GPS (EXIF) found but verification limited (metadata missing or reprocessed)."
    
    return result


def _compute_evidential_quality(report: Dict) -> Dict[str, Any]:
    """Compute evidential quality score and level"""
    # Check if already computed
    existing = _safe_get(report, "summary_axes", "evidential_quality")
    if existing and existing.get("score") is not None:
        return existing
    
    score = 100
    reasons = []
    
    # Domain penalties
    domain = report.get("domain", {})
    characteristics = domain.get("characteristics", [])
    
    if "recompressed" in characteristics or domain.get("type") == "recompressed":
        score -= 20
        reasons.append("recompressed")
    
    if "exif_stripped" in characteristics:
        score -= 20
        reasons.append("exif_stripped")
    
    if "screenshot" in characteristics or domain.get("type") == "screenshot":
        score -= 15
        reasons.append("screenshot")
    
    if "resized" in characteristics:
        score -= 10
        reasons.append("resized")
    
    # JPEG forensics
    jpeg = report.get("jpeg_forensics", {})
    if jpeg.get("double_compression_probability", 0) >= 0.40:
        score -= 10
        reasons.append("double_compression")
    
    # Uncertainty metrics
    uncertainty = report.get("uncertainty", {})
    ci = uncertainty.get("confidence_interval", [0.3, 0.7])
    ci_width = ci[1] - ci[0] if len(ci) >= 2 else 0.4
    
    if ci_width >= 0.40:
        score -= 10
        reasons.append("wide_ci")
    
    disagreement = uncertainty.get("disagreement", 0)
    if disagreement >= 0.45:
        score -= 10
        reasons.append("high_disagreement")
    
    # Clamp
    score = max(0, min(100, score))
    
    # Level
    if score >= 75:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"
    
    return {
        "score": score,
        "level": level,
        "reasons": reasons
    }


def _compute_ai_likelihood(report: Dict) -> Dict[str, Any]:
    """Compute AI likelihood axis"""
    existing = _safe_get(report, "summary_axes", "ai_likelihood")
    if existing and existing.get("probability") is not None:
        return existing
    
    p = report.get("ai_probability", 50) / 100.0
    uncertainty = report.get("uncertainty", {})
    ci = uncertainty.get("confidence_interval", [0.3, 0.7])
    
    if p >= 0.70:
        level = "high"
    elif p <= 0.30:
        level = "low"
    else:
        level = "medium"
    
    return {
        "probability": p,
        "confidence_interval": ci,
        "level": level
    }


def _compute_content_type_axis(report: Dict) -> Dict[str, Any]:
    """Compute content type axis with proper gating"""
    content_type = report.get("content_type", {})
    
    non_photo_score = content_type.get("non_photo_score", 0.5)
    photo_score = content_type.get("photo_score", 0.5)
    composite_score = content_type.get("composite_score", 0)
    margin = content_type.get("margin", 0)
    ct_type = content_type.get("type", "uncertain")
    ct_confidence = content_type.get("confidence", "low")
    
    # Check for GPS and domain original (strong real photo indicators)
    gps = report.get("gps", {})
    gps_present = gps.get("present", False) and gps.get("latitude") is not None
    domain = report.get("domain", {})
    domain_original = domain.get("type") == "original"
    
    # Determine axis value
    if ct_type == "non_photo_like" and ct_confidence == "high":
        return {"value": "non_photo_like", "confidence": "high"}
    
    # CRITICAL FIX: Don't label as composite without evidence
    # If GPS present + domain original, strongly favor photo_like
    if gps_present and domain_original:
        if photo_score >= non_photo_score * 0.8:  # Photo score is reasonably high
            return {"value": "photo_like", "confidence": "medium"}
    
    if ct_type == "photo_like":
        return {"value": "photo_like", "confidence": ct_confidence}
    
    if ct_type == "possible_composite":
        # Downgrade confidence if no manipulation evidence
        return {"value": "composite_possible", "confidence": "low"}
    
    if ct_type == "non_photo_like":
        return {"value": "non_photo_like", "confidence": ct_confidence}
    
    return {"value": "unknown", "confidence": "low"}


def _check_composite_evidence_strong(report: Dict) -> bool:
    """
    Check if composite evidence is backed by edit assessment.
    
    CRITICAL RULES:
    - generator_artifacts edit type should NOT trigger composite labels
    - local_manipulation requires boundary_corroborated == true
    - noise_inconsistency + blur_mismatch alone => NOT composite
    - Without boundary evidence, NEVER claim composite
    """
    # First check edit_assessment if available (new system)
    edit_assessment = report.get("edit_assessment", {})
    if edit_assessment:
        edit_type = edit_assessment.get("edit_type", "none_detected")
        confidence = edit_assessment.get("confidence", "low")
        boundary_corroborated = edit_assessment.get("boundary_corroborated", False)
        
        # CRITICAL: Only return True if:
        # 1. edit_type is local_manipulation
        # 2. confidence is high or medium
        # 3. boundary_corroborated is True
        if edit_type == "local_manipulation" and confidence in ["high", "medium"] and boundary_corroborated:
            return True
        
        # If edit_assessment says global, generator_artifacts, or none, don't claim composite
        # Also don't claim composite if boundary is not corroborated
        if edit_type in ["global_postprocess", "none_detected", "generator_artifacts"]:
            return False
        
        # Even if edit_type is local_manipulation but no boundary corroboration
        if edit_type == "local_manipulation" and not boundary_corroborated:
            return False
    
    # Fallback to old manipulation check (for backward compatibility)
    # But still require boundary-type evidence
    manipulation = report.get("manipulation", {})
    
    # Check each manipulation module
    splice = _safe_get(manipulation, "splice", "score", default=0)
    copy_move = _safe_get(manipulation, "copy_move", "score", default=0)
    edge_matte = _safe_get(manipulation, "edge_matte", "score", default=0)
    blur_mismatch = _safe_get(manipulation, "blur_noise_mismatch", "score", default=0)
    
    # CRITICAL: Require boundary-type corroboration
    # Boundary methods: copy_move, edge_matte
    # NOT boundary: splice (noise), blur_mismatch
    threshold = 0.60
    
    # Copy-move alone is strong (very specific detection)
    if copy_move >= 0.75:
        return True
    
    # Edge matte alone with high score
    if edge_matte >= 0.75:
        return True
    
    # Require at least one boundary method above threshold
    boundary_methods_high = (copy_move >= threshold) or (edge_matte >= threshold)
    
    if not boundary_methods_high:
        # No boundary evidence - cannot claim composite
        return False
    
    # With boundary evidence, also require corroboration from another method
    other_methods_high = (splice >= threshold) or (blur_mismatch >= threshold)
    
    return boundary_methods_high and other_methods_high


def _check_non_photo_high_conf(report: Dict, content_axis: Dict) -> bool:
    """Check if non-photo classification is high confidence"""
    if content_axis.get("value") == "non_photo_like" and content_axis.get("confidence") == "high":
        return True
    
    # Also check CLIP semantic score
    models = report.get("models", [])
    for m in models:
        if m.get("name") == "clip_semantic" and m.get("raw_score", 0) >= 0.95:
            content_type = report.get("content_type", {})
            if content_type.get("non_photo_score", 0) >= 0.65:
                return True
    
    return False


def _check_global_edit(report: Dict) -> Tuple[bool, str]:
    """
    Check if image has global post-processing (filters, color grading).
    Returns (is_global_edit, confidence)
    
    CRITICAL FIX: Only output global_postprocess if:
    1. Platform signatures present (Instagram software) OR
    2. Clear global transform evidence AND camera_evidence > 0.20
    
    If camera_evidence == 0 and domain is recompressed/exif_stripped,
    treat as generator_artifacts instead (for AI content).
    """
    edit_assessment = report.get("edit_assessment", {})
    pathway = report.get("pathway", {})
    
    # Get camera evidence score
    camera_evidence_score = pathway.get("evidence", {}).get("camera_evidence_score", 0)
    diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0)
    
    # Check domain characteristics
    domain = report.get("domain", {})
    characteristics = domain.get("characteristics", [])
    is_recompressed = "recompressed" in characteristics or domain.get("type") == "recompressed"
    is_exif_stripped = "exif_stripped" in characteristics
    
    # Check for platform software signatures
    metadata = report.get("metadata_extended", {}) or report.get("metadata", {})
    software = metadata.get("software", {}) if isinstance(metadata, dict) else {}
    software_name = software.get("name", "").lower() if isinstance(software, dict) else ""
    
    platform_apps = ["instagram", "whatsapp", "facebook", "twitter", "snapchat", "tiktok"]
    has_platform_signature = any(app in software_name for app in platform_apps)
    
    if edit_assessment:
        edit_type = edit_assessment.get("edit_type", "none_detected")
        confidence = edit_assessment.get("confidence", "low")
        global_score = edit_assessment.get("global_adjustment_score", 0)
        
        if edit_type == "global_postprocess":
            # CRITICAL FIX: Only allow global_postprocess if camera evidence exists
            # OR platform signature is present
            if camera_evidence_score >= 0.20 or has_platform_signature:
                return True, confidence
            
            # If no camera evidence and high diffusion, this is likely AI content
            # Don't label as "Photo (global edit)"
            if camera_evidence_score < 0.20 and diffusion_score >= 0.55:
                return False, "low"  # Will be handled as AI/generator_artifacts
            
            # If recompressed/exif_stripped with no camera evidence, be cautious
            if (is_recompressed or is_exif_stripped) and camera_evidence_score < 0.20:
                # Only allow if platform signature present
                if has_platform_signature:
                    return True, "low"
                return False, "low"
            
            return True, confidence
        
        # Also check if global score is high even if not classified as global
        if global_score >= 0.65:
            # Same gating: require camera evidence or platform signature
            if camera_evidence_score >= 0.20 or has_platform_signature:
                return True, "low"
    
    # Fallback: check domain characteristics
    if "social_media" in characteristics or is_recompressed:
        # Only if platform signature present
        if has_platform_signature:
            manipulation = report.get("manipulation", {})
            splice = _safe_get(manipulation, "splice", "score", default=0)
            blur = _safe_get(manipulation, "blur_noise_mismatch", "score", default=0)
            
            if (splice >= 0.50 or blur >= 0.50) and not _check_composite_evidence_strong(report):
                return True, "medium"
    
    return False, "low"


def _check_generator_artifacts(report: Dict) -> Tuple[bool, str]:
    """
    Check if image has AI generator artifacts (not manipulation).
    Returns (has_generator_artifacts, confidence)
    """
    edit_assessment = report.get("edit_assessment", {})
    
    if edit_assessment:
        edit_type = edit_assessment.get("edit_type", "none_detected")
        confidence = edit_assessment.get("confidence", "low")
        gen_artifacts_score = edit_assessment.get("generator_artifacts_score", 0)
        
        if edit_type == "generator_artifacts":
            return True, confidence
        
        # Also check if generator artifacts score is high
        if gen_artifacts_score >= 0.50:
            return True, "low"
    
    return False, "low"


def _check_photoreal_ai(report: Dict) -> Tuple[bool, str, float]:
    """
    Check if image is photoreal AI-generated (popular closed-model family).
    Returns (is_photoreal_ai, confidence, hint_score)
    """
    content_type = report.get("content_type", {})
    pathway = report.get("pathway", {})
    
    photo_like_generated_hint = content_type.get("photo_like_generated_hint", 0)
    ct_type = content_type.get("type", "unknown")
    
    pathway_pred = pathway.get("pred", "unknown")
    pathway_conf = pathway.get("confidence", "low")
    generator_family = pathway.get("generator_family", {}).get("pred", "unknown")
    
    # Check for photoreal AI indicators
    is_photoreal = False
    confidence = "low"
    
    # Content type indicates photo-like generated
    if ct_type == "photo_like_generated":
        is_photoreal = True
        confidence = "medium"
    
    # Pathway indicates T2I with vendor photoreal style
    if pathway_pred == "t2i" and generator_family == "vendor_photoreal_style":
        is_photoreal = True
        if pathway_conf in ["high", "medium"]:
            confidence = pathway_conf
    
    # High photo_like_generated_hint with T2I pathway
    if photo_like_generated_hint >= 0.55 and pathway_pred == "t2i":
        is_photoreal = True
        if photo_like_generated_hint >= 0.70:
            confidence = "high"
        elif photo_like_generated_hint >= 0.55:
            confidence = "medium"
    
    return is_photoreal, confidence, photo_like_generated_hint


def _check_ai_high_conf(ai_likelihood: Dict) -> bool:
    """Check if AI likelihood is high confidence"""
    p = ai_likelihood.get("probability", 0.5)
    ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
    ci_low = ci[0] if len(ci) >= 1 else 0.3
    
    return p >= 0.80 and ci_low >= 0.60


def _check_ai_low_conf(ai_likelihood: Dict) -> bool:
    """Check if AI likelihood is low (likely real)"""
    p = ai_likelihood.get("probability", 0.5)
    ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
    ci_high = ci[1] if len(ci) >= 2 else 0.7
    
    return p <= 0.20 and ci_high <= 0.45


def _check_truly_inconclusive(report: Dict, ai_likelihood: Dict, content_axis: Dict) -> bool:
    """
    Check if result is truly inconclusive (only case for BELİRSİZ)
    
    UPDATED RULES:
    Use BELİRSİZ only if:
    - pAI in [0.40, 0.60] AND model_disagreement > 0.35
    OR
    - evidence_quality < 30 AND no dominant class margin
    
    Otherwise choose the best class with a calibrated confidence band.
    """
    p = ai_likelihood.get("probability", 0.5)
    ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
    ci_width = ci[1] - ci[0] if len(ci) >= 2 else 0.4
    
    uncertainty = report.get("uncertainty", {})
    disagreement = uncertainty.get("disagreement", 0)
    
    # Get evidential quality
    eq = _compute_evidential_quality(report)
    eq_score = eq.get("score", 50)
    
    # RULE 1: Must be in uncertain probability band [0.40, 0.60]
    if not (0.40 <= p <= 0.60):
        return False
    
    # RULE 2: Must have high disagreement (> 0.35)
    if disagreement > 0.35:
        return True
    
    # RULE 3: Very low evidence quality (< 30) with no dominant class
    if eq_score < 30 and ci_width > 0.30:
        return True
    
    # Content type must be unknown/low confidence for true inconclusive
    if content_axis.get("value") not in ["unknown", "uncertain"] and content_axis.get("confidence") != "low":
        return False
    
    return False


def _check_photo_verdict_valid(report: Dict) -> Tuple[bool, str]:
    """
    Check if "Photo" verdict is valid.
    
    CRITICAL GATING:
    "Photo" verdict requires at least one of:
    1. Strong camera pipeline evidence (CFA/demosaic/sensor PRNU proxy) >= 0.35
    2. Strong authentic metadata bundle (DateTimeOriginal + camera make/model/lens + exposure)
       with consistency checks
    
    If not met -> do NOT output "Photo" as final label.
    """
    pathway = report.get("pathway", {})
    metadata = report.get("metadata", {}) or report.get("metadata_extended", {})
    
    # Get camera evidence score from pathway
    camera_evidence_score = pathway.get("evidence", {}).get("camera_evidence_score", 0)
    diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0)
    
    # GATE 1: Strong camera pipeline evidence
    if camera_evidence_score >= 0.35:
        return True, "camera_pipeline_evidence"
    
    # GATE 2: Authentic metadata bundle
    camera = metadata.get("camera", {}) if isinstance(metadata, dict) else {}
    timestamps = metadata.get("timestamps", {}) if isinstance(metadata, dict) else {}
    technical = metadata.get("technical", {}) if isinstance(metadata, dict) else {}
    gps = report.get("gps", {})
    
    # Check for authentic metadata bundle
    has_camera_info = camera.get("has_camera_info", False) if isinstance(camera, dict) else False
    has_datetime = bool(timestamps.get("datetime_original")) if isinstance(timestamps, dict) else False
    has_exposure = bool(technical.get("exposure_time")) if isinstance(technical, dict) else False
    has_gps = gps.get("present", False) and gps.get("latitude") is not None
    
    # Count authentic metadata signals
    auth_signals = sum([
        has_camera_info,
        has_datetime,
        has_exposure,
        has_gps
    ])
    
    # Require at least 2 authentic metadata signals for "Photo" verdict
    if auth_signals >= 2:
        # Additional check: diffusion score should be low for real photos
        if diffusion_score < 0.45:
            return True, "authentic_metadata_bundle"
    
    # If camera evidence is moderate (0.20-0.35) with some metadata
    if camera_evidence_score >= 0.20 and auth_signals >= 1 and diffusion_score < 0.50:
        return True, "moderate_camera_evidence"
    
    # Photo verdict NOT valid
    return False, "insufficient_camera_evidence"


def _format_percentage(value: float) -> str:
    """Format as percentage string"""
    return f"{int(value * 100)}"


def _compute_generative_candidate(report: Dict) -> Dict[str, Any]:
    """
    Compute generative_candidate flag for verdict generation.
    
    generative_candidate = (
        camera_evidence_score < 0.20
        AND (
            diffusion_fingerprint_score >= 0.25
            OR pAI >= 0.55
            OR (primary_model_score >= 0.70 AND model_disagreement <= 0.45)
            OR (clip_semantic_nonphoto_prob >= 0.55)
        )
    )
    """
    result = {
        "is_candidate": False,
        "camera_likelihood_low": False,
        "reasons": []
    }
    
    # Get pathway evidence
    pathway = report.get("pathway", {})
    camera_evidence_score = pathway.get("evidence", {}).get("camera_evidence_score", 0)
    diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0)
    
    # Fallback to diffusion_fingerprint if pathway doesn't have it
    if diffusion_score == 0:
        diffusion_fingerprint = report.get("diffusion_fingerprint", {})
        diffusion_score = diffusion_fingerprint.get("diffusion_score", 0)
    
    # Get AI probability
    ai_probability = report.get("ai_probability", 50)
    
    # Get model scores
    models = report.get("models", [])
    primary_model_score = 0.0
    model_disagreement = 0.0
    if models:
        scores = [m.get("raw_score", 0) for m in models if m.get("raw_score")]
        if scores:
            primary_model_score = max(scores)
            model_disagreement = max(scores) - min(scores) if len(scores) > 1 else 0
    
    # Get non-photo score
    content_type = report.get("content_type", {})
    non_photo_score = content_type.get("non_photo_score", 0.5)
    
    # Check camera likelihood
    result["camera_likelihood_low"] = camera_evidence_score < 0.20
    
    if not result["camera_likelihood_low"]:
        result["reasons"].append("camera_evidence_too_high")
        return result
    
    result["reasons"].append("low_camera_evidence")
    
    # Check generative signals
    generative_signals = []
    
    if diffusion_score >= 0.25:
        generative_signals.append(f"diffusion_{diffusion_score:.2f}")
    
    if ai_probability >= 55:
        generative_signals.append(f"pAI_{ai_probability:.0f}")
    
    if primary_model_score >= 0.70 and model_disagreement <= 0.45:
        generative_signals.append("primary_high_low_disagreement")
    
    if non_photo_score >= 0.55:
        generative_signals.append(f"non_photo_{non_photo_score:.2f}")
    
    if len(generative_signals) > 0:
        result["is_candidate"] = True
        result["reasons"].extend(generative_signals)
    else:
        result["reasons"].append("no_generative_signals")
    
    return result


def generate_verdict_text(report: Dict) -> Dict[str, Any]:
    """
    Generate verdict text based on report data.
    
    Returns dict with:
    - verdict_key, title_tr, title_en, subtitle_tr, subtitle_en
    - banner_key, banner_tr, banner_en (optional)
    - footer_key, footer_tr, footer_en
    - tags, summary_axes
    """
    # Compute axes
    eq = _compute_evidential_quality(report)
    ai_likelihood = _compute_ai_likelihood(report)
    content_axis = _compute_content_type_axis(report)
    
    # Check conditions
    composite_evidence_strong = _check_composite_evidence_strong(report)
    non_photo_high_conf = _check_non_photo_high_conf(report, content_axis)
    ai_high_conf = _check_ai_high_conf(ai_likelihood)
    ai_low_conf = _check_ai_low_conf(ai_likelihood)
    truly_inconclusive = _check_truly_inconclusive(report, ai_likelihood, content_axis)
    global_edit, global_edit_conf = _check_global_edit(report)
    generator_artifacts, gen_artifacts_conf = _check_generator_artifacts(report)
    photoreal_ai, photoreal_ai_conf, photoreal_hint = _check_photoreal_ai(report)
    photo_verdict_valid, photo_verdict_reason = _check_photo_verdict_valid(report)
    
    # Extract values for formatting
    p = ai_likelihood.get("probability", 0.5)
    ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
    p_pct = _format_percentage(p)
    ci_low_pct = _format_percentage(ci[0]) if len(ci) >= 1 else "30"
    ci_high_pct = _format_percentage(ci[1]) if len(ci) >= 2 else "70"
    eq_score = eq.get("score", 50)
    eq_level = eq.get("level", "medium")
    ai_level = ai_likelihood.get("level", "medium")
    
    # Additional checks for real photo
    gps = report.get("gps", {})
    gps_present = gps.get("present", False) and gps.get("latitude") is not None
    domain = report.get("domain", {})
    domain_original = domain.get("type") == "original"
    content_type = report.get("content_type", {})
    photo_score = content_type.get("photo_score", 0.5)
    non_photo_score = content_type.get("non_photo_score", 0.5)
    composite_score = content_type.get("composite_score", 0)
    
    # Edit assessment data
    edit_assessment = report.get("edit_assessment", {})
    edit_type = edit_assessment.get("edit_type", "none_detected")
    
    # Pathway data (T2I vs I2I vs Real)
    pathway = report.get("pathway", {})
    pathway_pred = pathway.get("pred", "unknown")
    pathway_conf = pathway.get("confidence", "low")
    generator_family = pathway.get("generator_family", {}).get("pred", "unknown")
    
    # Get diffusion score from pathway evidence or diffusion_fingerprint
    diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0)
    if diffusion_score == 0:
        diffusion_fingerprint = report.get("diffusion_fingerprint", {})
        diffusion_score = diffusion_fingerprint.get("diffusion_score", 0)
    
    # NEW: Compute generative_candidate
    generative_candidate = _compute_generative_candidate(report)
    is_generative_candidate = generative_candidate.get("is_candidate", False)
    camera_likelihood_low = generative_candidate.get("camera_likelihood_low", False)
    
    # ============================================================================
    # NEW: Text Forensics - Gibberish text as AI evidence
    # ============================================================================
    text_forensics = report.get("text_forensics", {})
    text_has_gibberish = False
    text_ai_score = 0.0
    gibberish_boost = 0.0
    
    if text_forensics and text_forensics.get("enabled", False) and text_forensics.get("has_text", False):
        text_ai_score = text_forensics.get("ai_text_score", 0)
        gibberish_ratio = text_forensics.get("gibberish_ratio", 0)
        text_verdict = text_forensics.get("verdict", "no_text")
        
        # If gibberish detected with high confidence, boost AI likelihood
        if text_verdict == "likely_ai" and gibberish_ratio >= 0.5:
            text_has_gibberish = True
            gibberish_boost = 0.15  # Significant boost
            logger.info(f"TEXT FORENSICS: Gibberish detected (ratio={gibberish_ratio:.2f}), AI boost +{gibberish_boost:.2f}")
        elif text_ai_score >= 0.6:
            text_has_gibberish = True
            gibberish_boost = 0.10  # Moderate boost
            logger.info(f"TEXT FORENSICS: High AI text score ({text_ai_score:.2f}), AI boost +{gibberish_boost:.2f}")
        elif gibberish_ratio >= 0.3:
            text_has_gibberish = True
            gibberish_boost = 0.05  # Small boost
            logger.info(f"TEXT FORENSICS: Some gibberish detected (ratio={gibberish_ratio:.2f}), AI boost +{gibberish_boost:.2f}")
    
    # Apply gibberish boost to AI probability for verdict calculation
    effective_p = min(1.0, p + gibberish_boost)
    if gibberish_boost > 0:
        logger.info(f"Effective pAI after text forensics: {p:.2f} -> {effective_p:.2f}")
    
    # Build summary axes
    summary_axes = {
        "content_type_axis": content_axis,
        "ai_likelihood": ai_likelihood,
        "evidential_quality": eq
    }
    
    # NEW: Add final_flags to summary_axes
    summary_axes["final_flags"] = {
        "generative_candidate": is_generative_candidate,
        "camera_likelihood_low": camera_likelihood_low,
        "generative_candidate_reasons": generative_candidate.get("reasons", []),
        "text_has_gibberish": text_has_gibberish,
        "gibberish_boost": gibberish_boost,
        "effective_p": effective_p
    }
    
    # Determine verdict using priority order
    verdict_key = None
    title_tr = ""
    title_en = ""
    subtitle_tr = ""
    subtitle_en = ""
    tags = []
    
    # Get camera evidence score for override rule
    camera_evidence_score = pathway.get("evidence", {}).get("camera_evidence_score", 0)
    
    # ============================================================================
    # NEW: Get provenance score from report (computed in main.py)
    # ============================================================================
    provenance_result = report.get("provenance", {})
    provenance_score = provenance_result.get("score", 0.0)
    is_platform_reencoded = provenance_result.get("is_platform_reencoded", False)
    
    # Get local tamper score
    tamper_score_local = edit_assessment.get("local_manipulation_score", 0.0)
    
    # ============================================================================
    # CRITICAL FIX: Boost camera_score with provenance
    # If provenance is strong, camera_score should be boosted
    # ============================================================================
    effective_camera_score = camera_evidence_score
    if provenance_score >= 0.45:
        camera_boost = provenance_score * 0.4
        effective_camera_score = min(1.0, camera_evidence_score + camera_boost)
        logger.info(f"Camera score boosted by provenance: {camera_evidence_score:.2f} -> {effective_camera_score:.2f}")
    
    # If platform software detected, camera_score should not collapse
    if is_platform_reencoded and effective_camera_score < 0.20:
        effective_camera_score = max(effective_camera_score, 0.20)
        logger.info(f"Camera score floor applied (platform re-encoding): {effective_camera_score:.2f}")
    
    # ============================================================================
    # PRIORITY 0: T2I OVERRIDE RULE (UPDATED - respects provenance)
    # ============================================================================
    # If pathway == T2I with high/medium confidence AND camera_evidence <= 0.15
    # AND provenance_score < 0.20:
    # - Force verdict to AI-generated regardless of pAI
    # 
    # CRITICAL: T2I pathway is NOT allowed to override strong provenance
    # If provenance_score >= 0.45 AND tamper_score_local < 0.25:
    # - Do NOT force AI verdict
    # ============================================================================
    
    # Check if provenance strongly supports photo
    provenance_blocks_ai = (provenance_score >= 0.45 and tamper_score_local < 0.25)
    
    t2i_override = (
        pathway_pred == "t2i" and 
        pathway_conf in ["high", "medium"] and 
        effective_camera_score <= 0.15 and
        provenance_score < 0.20  # NEW: Only override if provenance is weak
    )
    
    # Also trigger override if generative_score is high and camera_score is very low
    # BUT NOT if provenance strongly supports photo
    # CRITICAL FIX: Also require pAI >= 0.40 to avoid false positives on real photos
    # NEW: Gibberish text can also trigger override
    generative_score_override = (
        is_generative_candidate and 
        effective_camera_score <= 0.10 and
        diffusion_score >= 0.30 and
        effective_p >= 0.40 and  # Use effective_p (includes gibberish boost)
        not provenance_blocks_ai  # NEW: Respect provenance
    )
    
    # NEW: Gibberish text override - if strong gibberish detected, can trigger AI verdict
    gibberish_override = (
        text_has_gibberish and
        gibberish_boost >= 0.10 and  # Significant gibberish
        effective_camera_score <= 0.15 and
        effective_p >= 0.35 and  # Lower threshold when gibberish is strong evidence
        not provenance_blocks_ai
    )
    
    if t2i_override or generative_score_override or gibberish_override:
        # Determine confidence tier based on evidence quality
        override_reason = "t2i" if t2i_override else ("gibberish" if gibberish_override else "generative_score")
        
        if eq_score >= 50 and (pathway_conf == "high" or diffusion_score >= 0.50 or gibberish_boost >= 0.10):
            # High confidence
            if photoreal_ai or photoreal_hint >= 0.50:
                verdict_key = VerdictKey.AI_PHOTOREAL_HIGH
                title_tr = "Muhtemelen Yapay Zeka Üretimi (Fotogerçekçi)"
                title_en = "Likely AI-Generated (Photorealistic)"
                subtitle_tr = f"T2I pathway tespit edildi ({pathway_conf}). Kamera kanıtı: {int(effective_camera_score*100)}%. Diffusion: {int(diffusion_score*100)}%."
                subtitle_en = f"T2I pathway detected ({pathway_conf}). Camera evidence: {int(effective_camera_score*100)}%. Diffusion: {int(diffusion_score*100)}%."
                tags = ["T2I", "Photoreal AI", "High Confidence"]
            else:
                verdict_key = VerdictKey.AI_T2I_HIGH
                title_tr = "Muhtemelen Yapay Zeka Üretimi (Text-to-Image)"
                title_en = "Likely AI-Generated (Text-to-Image)"
                subtitle_tr = f"T2I pathway tespit edildi ({pathway_conf}). Kamera kanıtı: {int(effective_camera_score*100)}%. Diffusion: {int(diffusion_score*100)}%."
                subtitle_en = f"T2I pathway detected ({pathway_conf}). Camera evidence: {int(effective_camera_score*100)}%. Diffusion: {int(diffusion_score*100)}%."
                tags = ["T2I", "High Confidence"]
        else:
            # Medium confidence (evidence quality < 50 or pathway_conf != high)
            if photoreal_ai or photoreal_hint >= 0.50:
                verdict_key = VerdictKey.AI_PHOTOREAL_MEDIUM
                title_tr = "Muhtemelen Yapay Zeka Üretimi (Orta Güven)"
                title_en = "Likely AI-Generated (Medium Confidence)"
                subtitle_tr = f"T2I pathway tespit edildi. Kamera kanıtı yok. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
                subtitle_en = f"T2I pathway detected. No camera evidence. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
                tags = ["T2I", "Photoreal AI", "Medium Confidence"]
            else:
                verdict_key = VerdictKey.AI_T2I_MEDIUM
                title_tr = "Muhtemelen Yapay Zeka Üretimi (Orta Güven)"
                title_en = "Likely AI-Generated (Medium Confidence)"
                subtitle_tr = f"T2I pathway tespit edildi. Kamera kanıtı yok. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
                subtitle_en = f"T2I pathway detected. No camera evidence. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
                tags = ["T2I", "Medium Confidence"]
        
        # Add evidence quality tag
        if eq_level == "low":
            tags.append("Low Evidence")
        
        logger.info(f"T2I OVERRIDE triggered: pathway={pathway_pred}({pathway_conf}), "
                   f"camera={effective_camera_score:.2f}, diffusion={diffusion_score:.2f}, "
                   f"reason={override_reason}, gibberish_boost={gibberish_boost:.2f}")
    
    # ============================================================================
    # PRIORITY 0.5: STRONG PROVENANCE PHOTO RULE (NEW)
    # ============================================================================
    # If provenance_score >= 0.45 AND tamper_score_local < 0.25:
    # - Forbid headline "Likely AI"
    # - Headline must be "Muhtemelen Gerçek Fotoğraf" or "Belirsiz"
    # ============================================================================
    elif provenance_blocks_ai:
        # Strong provenance supports photo
        # Check if we should output PHOTO or INCONCLUSIVE
        
        # Compute posterior gap (simplified)
        # posterior_photo - posterior_ai should be >= 0.15 for PHOTO verdict
        photo_signals = effective_camera_score + provenance_score
        ai_signals = diffusion_score + (p - 0.5) * 0.5  # Normalize pAI contribution
        posterior_gap = photo_signals - ai_signals
        
        if posterior_gap >= 0.15 and eq_score >= 55:
            # Strong photo verdict
            if global_edit or is_platform_reencoded:
                verdict_key = VerdictKey.PHOTO_GLOBAL_EDIT
                title_tr = "Muhtemelen Gerçek Fotoğraf (Global düzenleme/Instagram işleme izi)"
                title_en = "Likely Real Photo (Global edit/Instagram processing detected)"
                subtitle_tr = f"EXIF/GPS fotoğraf lehine destekleyici ipuçları sağlıyor; yerel manipülasyon kanıtı yok. Provenance: {int(provenance_score*100)}%."
                subtitle_en = f"EXIF/GPS provides supporting clues for camera photo; no local manipulation evidence. Provenance: {int(provenance_score*100)}%."
                tags = ["Photo", "Global Edit", "Provenance"]
            else:
                verdict_key = VerdictKey.REAL_LIKELY
                title_tr = "Gerçek Fotoğraf (Yüksek Güven)"
                title_en = "Real Photo (High Confidence)"
                subtitle_tr = f"EXIF/GPS fotoğraf lehine güçlü ipuçları. Kamera kanıtı: {int(effective_camera_score*100)}%. Provenance: {int(provenance_score*100)}%."
                subtitle_en = f"Strong EXIF/GPS clues supporting camera photo. Camera evidence: {int(effective_camera_score*100)}%. Provenance: {int(provenance_score*100)}%."
                tags = ["Photo", "High Confidence", "Provenance"]
            
            if gps_present:
                tags.append("GPS")
        else:
            # Mixed signals - use INCONCLUSIVE
            verdict_key = VerdictKey.INCONCLUSIVE
            title_tr = "Belirsiz (Fotoğraf lehine güçlü metadata, ancak bazı üretken sinyaller var)"
            title_en = "Inconclusive (Strong metadata supporting photo, but some generative signals present)"
            subtitle_tr = f"Provenance: {int(provenance_score*100)}%. Diffusion: {int(diffusion_score*100)}%. Karışık sinyaller — orijinal dosya önerilir."
            subtitle_en = f"Provenance: {int(provenance_score*100)}%. Diffusion: {int(diffusion_score*100)}%. Mixed signals — original file recommended."
            tags = ["Inconclusive", "Mixed Signals", "Provenance"]
        
        logger.info(f"PROVENANCE PHOTO RULE triggered: provenance={provenance_score:.2f}, "
                   f"tamper_local={tamper_score_local:.2f}, posterior_gap={posterior_gap:.2f}")
    
    # PRIORITY 1: Non-photo digital artwork (CGI/3D/Illustration)
    # This OVERRIDES composite labels for non-photo content
    elif non_photo_high_conf:
        verdict_key = VerdictKey.NON_PHOTO_HIGH
        title_tr = "Dijital / Non-Photo içerik (Yüksek güven)"
        title_en = "Digital / Non-Photo content (High confidence)"
        subtitle_tr = f"Bu görsel fotoğraf değil; dijital çizim/CGI/AI sanatına benziyor. (pAI={p_pct}%, CI={ci_low_pct}–{ci_high_pct}%)"
        subtitle_en = f"This is not a camera photo; it matches digital/CGI/AI artwork. (pAI={p_pct}%, CI={ci_low_pct}–{ci_high_pct}%)"
        tags = ["Non-Photo", eq_level.capitalize(), f"AI: {ai_level.capitalize()}"]
    
    # PRIORITY 2: Composite/Edited (ONLY if evidence-backed AND boundary-corroborated)
    elif composite_evidence_strong and edit_type == "local_manipulation":
        verdict_key = VerdictKey.COMPOSITE_LIKELY
        title_tr = "Muhtemelen Kompozit / Düzenlenmiş"
        title_en = "Likely Composite / Edited"
        subtitle_tr = f"Yapıştırma/maskeleme izleri bulundu (çoklu yöntem doğrulaması). Kanıt kalitesi: {eq_score}/100."
        subtitle_en = f"Splice/matting signals detected (multi-method corroboration). Evidence quality: {eq_score}/100."
        tags = ["Composite", eq_level.capitalize(), f"AI: {ai_level.capitalize()}"]
    
    # PRIORITY 2.5: Generator Artifacts (AI generation artifacts, NOT manipulation)
    elif generator_artifacts and edit_type == "generator_artifacts":
        verdict_key = VerdictKey.GENERATOR_ARTIFACTS
        title_tr = "AI Üretim Artefaktları Tespit Edildi"
        title_en = "AI Generation Artifacts Detected"
        subtitle_tr = f"Doku/gürültü tutarsızlıkları AI üretim sürecinden kaynaklanıyor, manipülasyon değil. (pAI={p_pct}%)"
        subtitle_en = f"Texture/noise inconsistencies are from AI generation process, not manipulation. (pAI={p_pct}%)"
        tags = ["AI Artifacts", eq_level.capitalize()]
    
    # PRIORITY 3: Global Edit (filter/color grading)
    # CRITICAL FIX: Only if photo_verdict_valid AND not AI-generated
    elif global_edit and not composite_evidence_strong and not generator_artifacts and photo_verdict_valid:
        verdict_key = VerdictKey.PHOTO_GLOBAL_EDIT
        title_tr = "Fotoğraf (global düzenleme/filtre izi)"
        title_en = "Photo (global edit/filter detected)"
        subtitle_tr = f"Renk düzeltme, filtre veya platform işleme izleri var. Yerel manipülasyon kanıtı yok. (pAI={p_pct}%)"
        subtitle_en = f"Color grading, filter, or platform processing detected. No local manipulation evidence. (pAI={p_pct}%)"
        tags = ["Global Edit", eq_level.capitalize()]
        if gps_present:
            tags.append("GPS")
    
    # PRIORITY 3.3: AI with global edit signals but NO camera evidence
    # This catches T2I images that were being mislabeled as "Photo (global edit)"
    elif (global_edit or generator_artifacts) and not photo_verdict_valid and diffusion_score >= 0.55:
        # Force T2I classification when camera evidence is missing
        if photoreal_ai or photoreal_hint >= 0.50:
            verdict_key = VerdictKey.AI_PHOTOREAL_MEDIUM
            title_tr = "AI üretimi (Fotogerçekçi) olası"
            title_en = "Likely AI-generated (Photorealistic)"
            subtitle_tr = f"Kamera kanıtı yok, diffusion izi tespit edildi ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            subtitle_en = f"No camera evidence, diffusion fingerprint detected ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            tags = ["Photoreal AI", "No Camera Evidence", eq_level.capitalize()]
        else:
            verdict_key = VerdictKey.AI_T2I_MEDIUM
            title_tr = "AI üretimi (Text-to-Image) olası"
            title_en = "Likely AI-generated (Text-to-Image)"
            subtitle_tr = f"Kamera kanıtı yok, diffusion izi tespit edildi ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            subtitle_en = f"No camera evidence, diffusion fingerprint detected ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            tags = ["T2I", "No Camera Evidence", eq_level.capitalize()]
    
    # PRIORITY 3.5: Photoreal AI (popular closed-model family)
    elif photoreal_ai and pathway_pred == "t2i" and diffusion_score >= 0.55:
        if photoreal_ai_conf == "high" or (photoreal_hint >= 0.65 and diffusion_score >= 0.70):
            verdict_key = VerdictKey.AI_PHOTOREAL_HIGH
            title_tr = "AI üretimi (Fotogerçekçi) muhtemel"
            title_en = "Likely AI-generated (Photorealistic)"
            subtitle_tr = f"Popüler kapalı model fotogerçekçi ailesi. Diffusion izi: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
            subtitle_en = f"Popular closed-model photoreal family. Diffusion fingerprint: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
            tags = ["Photoreal AI", "T2I", eq_level.capitalize()]
        else:
            verdict_key = VerdictKey.AI_PHOTOREAL_MEDIUM
            title_tr = "AI üretimi (Fotogerçekçi) olası"
            title_en = "Possibly AI-generated (Photorealistic)"
            subtitle_tr = f"Fotogerçekçi AI üretim işaretleri. Diffusion izi: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
            subtitle_en = f"Photorealistic AI generation hints. Diffusion fingerprint: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
            tags = ["Photoreal AI?", eq_level.capitalize()]
    
    # PRIORITY 4: T2I (Text-to-Image) with high diffusion fingerprint
    elif pathway_pred == "t2i" and pathway_conf in ["high", "medium"] and diffusion_score >= 0.65:
        if pathway_conf == "high" or diffusion_score >= 0.75:
            verdict_key = VerdictKey.AI_T2I_HIGH
            title_tr = "AI üretimi (Text-to-Image) muhtemel"
            title_en = "Likely AI-generated (Text-to-Image)"
            subtitle_tr = f"Diffusion model izi güçlü ({int(diffusion_score*100)}%). Modern T2I ailesi. (pAI={p_pct}%)"
            subtitle_en = f"Strong diffusion fingerprint ({int(diffusion_score*100)}%). Modern T2I family. (pAI={p_pct}%)"
            tags = ["T2I", "Diffusion", eq_level.capitalize()]
        else:
            verdict_key = VerdictKey.AI_T2I_MEDIUM
            title_tr = "AI üretimi (Text-to-Image) olası"
            title_en = "Possibly AI-generated (Text-to-Image)"
            subtitle_tr = f"Diffusion model izi tespit edildi ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            subtitle_en = f"Diffusion fingerprint detected ({int(diffusion_score*100)}%). (pAI={p_pct}%)"
            tags = ["T2I?", eq_level.capitalize()]
    
    # PRIORITY 4.5: NEW - Generative candidate with unknown pathway
    # If generative_candidate == true but pathway is unknown, use this verdict
    elif is_generative_candidate and pathway_pred == "unknown" and not photo_verdict_valid:
        # Determine confidence based on signals
        gen_confidence = "medium"
        if diffusion_score >= 0.35 and p >= 0.60:
            gen_confidence = "medium"
        elif eq_score < 30:
            gen_confidence = "low"  # Cap confidence if evidence quality low
        
        verdict_key = VerdictKey.AI_GENERATIVE_UNKNOWN
        title_tr = "AI Üretimi (Muhtemel) — Pathway: Belirsiz"
        title_en = "Likely AI-Generated — Pathway: Unknown"
        subtitle_tr = f"Kamera kanıtı yok, üretim sinyalleri tespit edildi. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
        subtitle_en = f"No camera evidence, generative signals detected. Diffusion: {int(diffusion_score*100)}%. (pAI={p_pct}%)"
        tags = ["AI Generative", "Pathway Unknown", eq_level.capitalize()]
        if gen_confidence == "low":
            tags.append("Low Evidence")
    
    # PRIORITY 5: Likely AI (high confidence)
    elif ai_high_conf and not (gps_present and domain_original):
        verdict_key = VerdictKey.AI_HIGH
        title_tr = "AI üretimi muhtemel (Yüksek güven)"
        title_en = "Likely AI-generated (High confidence)"
        subtitle_tr = f"İstatistiksel skor yüksek: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Kanıt kalitesi: {eq_score}/100."
        subtitle_en = f"High statistical score: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Evidence quality: {eq_score}/100."
        tags = [eq_level.capitalize(), "AI High"]
    
    # PRIORITY 6: Likely Real Photo (REGRESSION FIX)
    # CRITICAL: Require camera_evidence_score >= 0.20 to output "Photo likely"
    elif (ai_low_conf or (p <= 0.40 and gps_present and domain_original and not composite_evidence_strong)) and camera_evidence_score >= 0.20:
        verdict_key = VerdictKey.REAL_LIKELY
        title_tr = "Gerçek fotoğraf olması muhtemel"
        title_en = "Likely a real photo"
        subtitle_tr = f"AI olasılığı düşük: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Kamera kanıtı: {int(camera_evidence_score*100)}%."
        subtitle_en = f"Low AI likelihood: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Camera evidence: {int(camera_evidence_score*100)}%."
        tags = ["Photo-like", eq_level.capitalize()]
        if gps_present:
            tags.append("GPS")
    
    # PRIORITY 7: Possible Composite (soft label, no strong evidence)
    elif composite_score >= 0.70 and not composite_evidence_strong and eq_level != "low":
        verdict_key = VerdictKey.COMPOSITE_POSSIBLE
        title_tr = "Kompozit/Düzenleme ihtimali var"
        title_en = "Possible composite/edit"
        subtitle_tr = "İşaretler var ama kanıt zayıf. Daha net sonuç için orijinal dosya önerilir."
        subtitle_en = "There are hints but evidence is weak. Upload the original file for a clearer result."
        tags = ["Composite?", eq_level.capitalize()]
    
    # PRIORITY 8: Medium AI
    elif p >= 0.60 and not truly_inconclusive:
        verdict_key = VerdictKey.AI_MEDIUM
        title_tr = "AI olasılığı yüksek"
        title_en = "High AI likelihood"
        subtitle_tr = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Kanıt kalitesi: {eq_score}/100."
        subtitle_en = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Evidence quality: {eq_score}/100."
        tags = [eq_level.capitalize()]
    
    # PRIORITY 9: Medium Real
    # CRITICAL: Do NOT output "Photo likely" if camera_evidence is very low
    elif p <= 0.40 and not truly_inconclusive and camera_evidence_score >= 0.20:
        verdict_key = VerdictKey.REAL_MEDIUM
        title_tr = "Gerçek fotoğraf olması olası"
        title_en = "Probably a real photo"
        subtitle_tr = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
        subtitle_en = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
        tags = [eq_level.capitalize()]
    
    # PRIORITY 9.5: Low pAI but no camera evidence
    # CRITICAL FIX: Be more conservative - only suggest AI if diffusion is high enough
    elif p <= 0.40 and not truly_inconclusive and camera_evidence_score < 0.20:
        # This catches cases where pAI is low but there's no camera evidence
        # Be conservative: only suggest AI if diffusion is significantly high
        # For very low pAI (< 30%), require higher diffusion threshold
        diffusion_threshold = 0.45 if p < 0.30 else 0.35
        
        if is_generative_candidate and diffusion_score >= diffusion_threshold:
            verdict_key = VerdictKey.AI_GENERATIVE_UNKNOWN
            title_tr = "AI Üretimi Olası — Düşük Güven"
            title_en = "Possibly AI-Generated — Low Confidence"
            subtitle_tr = f"Kamera kanıtı yok ({int(camera_evidence_score*100)}%). pAI düşük ({p_pct}%) ancak üretim sinyalleri mevcut."
            subtitle_en = f"No camera evidence ({int(camera_evidence_score*100)}%). pAI low ({p_pct}%) but generative signals present."
            tags = ["AI Possible", "Low Confidence", "No Camera Evidence"]
        else:
            # Low pAI + low camera evidence + low diffusion = truly inconclusive
            verdict_key = VerdictKey.INCONCLUSIVE
            title_tr = "Belirsiz — Yetersiz Kanıt"
            title_en = "Inconclusive — Insufficient Evidence"
            subtitle_tr = f"pAI düşük ({p_pct}%) ancak kamera kanıtı da yok. Orijinal dosya önerilir."
            subtitle_en = f"pAI low ({p_pct}%) but no camera evidence either. Upload original file recommended."
            tags = ["Inconclusive", "No Camera Evidence"]
    
    # PRIORITY 10: True Inconclusive (ONLY place for BELİRSİZ)
    else:
        verdict_key = VerdictKey.INCONCLUSIVE
        title_tr = "Belirsiz (Yeterli kanıt yok)"
        title_en = "Inconclusive (Not enough evidence)"
        subtitle_tr = "Model sonuçları çelişkili veya güven aralığı geniş. Orijinal dosya önerilir."
        subtitle_en = "Models disagree or uncertainty is high. Upload the original file for a clearer result."
        tags = [eq_level.capitalize(), "Uncertain"]
    
    # Determine banner
    banner_key = None
    banner_tr = None
    banner_en = None
    
    if eq_level == "low":
        banner_key = BannerKey.LOW_EVIDENCE
        banner_tr = BANNERS[BannerKey.LOW_EVIDENCE]["tr"]
        banner_en = BANNERS[BannerKey.LOW_EVIDENCE]["en"]
    elif verdict_key == VerdictKey.INCONCLUSIVE:
        banner_key = BannerKey.INCONCLUSIVE
        banner_tr = BANNERS[BannerKey.INCONCLUSIVE]["tr"]
        banner_en = BANNERS[BannerKey.INCONCLUSIVE]["en"]
    
    # Compute GPS consistency for conditional messaging
    gps_consistency = _check_gps_consistency(report)
    
    return {
        "verdict_key": verdict_key,
        "title_tr": title_tr,
        "title_en": title_en,
        "subtitle_tr": subtitle_tr,
        "subtitle_en": subtitle_en,
        "banner_key": banner_key,
        "banner_tr": banner_tr,
        "banner_en": banner_en,
        "footer_key": FOOTER_KEY,
        "footer_tr": FOOTER_TR,
        "footer_en": FOOTER_EN,
        "tags": tags,
        "summary_axes": summary_axes,
        "gps_consistency": gps_consistency  # NEW: GPS consistency info for conditional messaging
    }
