"""
Verdict Text Generator - UX Text Layer for Forensic AI Analysis
================================================================
Generates consistent, non-redundant verdict text based on analysis results.

Key principles:
- Two axes: content_type_axis + evidential_quality
- "BELİRSİZ/Inconclusive" only when truly_inconclusive
- Composite labels require evidence-backed triggers
- Single global footer + optional conditional banner (no ⚠️ spam)
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
    AI_PHOTOREAL_HIGH = "AI_PHOTOREAL_HIGH"  # NEW: Photoreal AI (popular closed-model family)
    AI_PHOTOREAL_MEDIUM = "AI_PHOTOREAL_MEDIUM"  # NEW: Photoreal AI medium confidence
    AI_HIGH = "AI_HIGH"
    AI_MEDIUM = "AI_MEDIUM"
    REAL_LIKELY = "REAL_LIKELY"
    REAL_MEDIUM = "REAL_MEDIUM"
    GENERATOR_ARTIFACTS = "GENERATOR_ARTIFACTS"  # NEW: AI generation artifacts (not manipulation)
    INCONCLUSIVE = "INCONCLUSIVE"


# Banner keys
class BannerKey:
    LOW_EVIDENCE = "BANNER_LOW_EVIDENCE"
    INCONCLUSIVE = "BANNER_INCONCLUSIVE"


# Footer key
FOOTER_KEY = "FOOTER_PROBABILISTIC"

# Footer text (always present)
FOOTER_TR = "Not: Bu sonuç olasılıksal analizdir. Kesin doğrulama için C2PA/Content Credentials veya güvenilir provenance gerekir."
FOOTER_EN = "Note: This is a probabilistic analysis. Definitive verification requires C2PA/Content Credentials or trusted provenance."

# Banner texts
BANNERS = {
    BannerKey.LOW_EVIDENCE: {
        "tr": "Kanıt kalitesi düşük (yeniden sıkıştırma/metadata eksik). Güven aralığı genişleyebilir.",
        "en": "Low evidential quality (recompressed/missing metadata). Uncertainty may be higher."
    },
    BannerKey.INCONCLUSIVE: {
        "tr": "Sonuç belirsiz. Daha net sonuç için orijinal dosyayı yükleyin.",
        "en": "Result is inconclusive. Upload the original file for a clearer result."
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
    """
    edit_assessment = report.get("edit_assessment", {})
    
    if edit_assessment:
        edit_type = edit_assessment.get("edit_type", "none_detected")
        confidence = edit_assessment.get("confidence", "low")
        global_score = edit_assessment.get("global_adjustment_score", 0)
        
        if edit_type == "global_postprocess":
            return True, confidence
        
        # Also check if global score is high even if not classified as global
        if global_score >= 0.65:
            return True, "low"
    
    # Fallback: check domain characteristics
    domain = report.get("domain", {})
    characteristics = domain.get("characteristics", [])
    
    if "social_media" in characteristics or "recompressed" in characteristics:
        # Check if manipulation scores are high but not corroborated
        manipulation = report.get("manipulation", {})
        splice = _safe_get(manipulation, "splice", "score", default=0)
        blur = _safe_get(manipulation, "blur_noise_mismatch", "score", default=0)
        
        # High scores but likely from recompression
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
    """Check if result is truly inconclusive (only case for BELİRSİZ)"""
    p = ai_likelihood.get("probability", 0.5)
    ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
    ci_width = ci[1] - ci[0] if len(ci) >= 2 else 0.4
    
    uncertainty = report.get("uncertainty", {})
    disagreement = uncertainty.get("disagreement", 0)
    
    # Must be in uncertain probability band
    if not (0.40 <= p <= 0.60):
        return False
    
    # Must have high disagreement OR wide CI
    if not (disagreement > 0.35 or ci_width > 0.35):
        return False
    
    # Content type must be unknown/low confidence
    if content_axis.get("value") not in ["unknown", "uncertain"] and content_axis.get("confidence") != "low":
        return False
    
    return True


def _format_percentage(value: float) -> str:
    """Format as percentage string"""
    return f"{int(value * 100)}"


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
    
    # Build summary axes
    summary_axes = {
        "content_type_axis": content_axis,
        "ai_likelihood": ai_likelihood,
        "evidential_quality": eq
    }
    
    # Determine verdict using priority order
    verdict_key = None
    title_tr = ""
    title_en = ""
    subtitle_tr = ""
    subtitle_en = ""
    tags = []
    
    # PRIORITY 1: Non-photo digital artwork (CGI/3D/Illustration)
    # This OVERRIDES composite labels for non-photo content
    if non_photo_high_conf:
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
    elif global_edit and not composite_evidence_strong and not generator_artifacts:
        verdict_key = VerdictKey.PHOTO_GLOBAL_EDIT
        title_tr = "Fotoğraf (global düzenleme/filtre izi)"
        title_en = "Photo (global edit/filter detected)"
        subtitle_tr = f"Renk düzeltme, filtre veya platform işleme izleri var. Yerel manipülasyon kanıtı yok. (pAI={p_pct}%)"
        subtitle_en = f"Color grading, filter, or platform processing detected. No local manipulation evidence. (pAI={p_pct}%)"
        tags = ["Global Edit", eq_level.capitalize()]
        if gps_present:
            tags.append("GPS")
    
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
    
    # PRIORITY 5: Likely AI (high confidence)
    elif ai_high_conf and not (gps_present and domain_original):
        verdict_key = VerdictKey.AI_HIGH
        title_tr = "AI üretimi muhtemel (Yüksek güven)"
        title_en = "Likely AI-generated (High confidence)"
        subtitle_tr = f"İstatistiksel skor yüksek: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Kanıt kalitesi: {eq_score}/100."
        subtitle_en = f"High statistical score: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%). Evidence quality: {eq_score}/100."
        tags = [eq_level.capitalize(), "AI High"]
    
    # PRIORITY 6: Likely Real Photo (REGRESSION FIX)
    elif ai_low_conf or (p <= 0.40 and gps_present and domain_original and not composite_evidence_strong):
        verdict_key = VerdictKey.REAL_LIKELY
        title_tr = "Gerçek fotoğraf olması muhtemel"
        title_en = "Likely a real photo"
        subtitle_tr = f"AI olasılığı düşük: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
        subtitle_en = f"Low AI likelihood: {p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
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
    elif p <= 0.40 and not truly_inconclusive:
        verdict_key = VerdictKey.REAL_MEDIUM
        title_tr = "Gerçek fotoğraf olması olası"
        title_en = "Probably a real photo"
        subtitle_tr = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
        subtitle_en = f"pAI={p_pct}% (CI {ci_low_pct}–{ci_high_pct}%)."
        tags = [eq_level.capitalize()]
    
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
        "summary_axes": summary_axes
    }
