"""
Evidence Fusion Module - Weighted Bayesian/Logit Fusion
=========================================================
Computes posterior probabilities using calibrated evidence weighting.

Key principles:
- Explicit evidence graph with weighted fusion
- Provenance boosts photo posterior, never directly sets "proof"
- T2I pathway is evidence, NOT a verdict switch
- Consistency checker prevents contradictory headlines

Posteriors computed:
- posterior_ai ∈ [0,1]
- posterior_photo ∈ [0,1]
- posterior_composite ∈ [0,1]
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvidenceWeights:
    """Configurable evidence weights for fusion"""
    # Generative evidence weight
    w_generative: float = 2.2
    # Camera pipeline evidence weight
    w_camera: float = 2.0
    # Provenance (EXIF/GPS) weight
    w_provenance: float = 1.6
    # Domain penalty weight
    w_domain: float = 0.8
    # Model disagreement weight
    w_disagreement: float = 0.9
    # Local manipulation weight
    w_local_tamper: float = 2.0
    # Boundary corroboration weight
    w_boundary: float = 1.2
    # AI probability weight (from ensemble)
    w_ai_prob: float = 1.5


@dataclass
class EvidenceGraph:
    """Evidence graph with all computed scores"""
    # Core scores
    camera_score: float = 0.0
    generative_score: float = 0.0
    provenance_score: float = 0.0
    tamper_score_local: float = 0.0
    edit_score_global: float = 0.0
    
    # Quality metrics
    evidence_quality: float = 50.0
    disagreement: float = 0.0
    ci_width: float = 0.4
    
    # Domain factors
    domain_penalty: float = 0.0
    is_recompressed: bool = False
    is_exif_stripped: bool = False
    is_platform_reencoded: bool = False
    
    # Pathway info
    pathway_pred: str = "unknown"
    pathway_confidence: str = "low"
    diffusion_score: float = 0.0
    
    # Posteriors (computed)
    posterior_ai: float = 0.5
    posterior_photo: float = 0.5
    posterior_composite: float = 0.0
    
    # Raw AI probability from ensemble
    ai_probability: float = 50.0
    
    # Rationale
    rationale: List[str] = field(default_factory=list)


class EvidenceFusion:
    """
    Weighted evidence fusion for forensic verdict generation.
    
    Uses logistic fusion:
    posterior_ai = σ(w_g*gen - w_c*cam - w_p*prov + w_d*domain + w_u*disagree)
    posterior_photo = σ(w_c*cam + w_p*prov - w_g*gen - w_d*domain - w_u*disagree)
    """
    
    def __init__(self, weights: EvidenceWeights = None):
        self.weights = weights or EvidenceWeights()
    
    def compute_evidence_graph(self,
                                report: Dict[str, Any],
                                provenance_result: Dict[str, Any]) -> EvidenceGraph:
        """
        Build evidence graph from analysis report.
        
        Args:
            report: Full analysis report dict
            provenance_result: Result from ProvenanceScorer
        
        Returns:
            EvidenceGraph with all scores and posteriors
        """
        graph = EvidenceGraph()
        
        # === Extract core scores ===
        
        # Provenance score
        graph.provenance_score = provenance_result.get("score", 0.0)
        graph.is_platform_reencoded = provenance_result.get("is_platform_reencoded", False)
        
        # Pathway and diffusion
        pathway = report.get("pathway", {})
        graph.pathway_pred = pathway.get("pred", "unknown")
        graph.pathway_confidence = pathway.get("confidence", "low")
        graph.diffusion_score = pathway.get("evidence", {}).get("diffusion_score", 0.0)
        
        # Fallback to diffusion_fingerprint if pathway doesn't have it
        if graph.diffusion_score == 0:
            diffusion_fp = report.get("diffusion_fingerprint", {})
            graph.diffusion_score = diffusion_fp.get("diffusion_score", 0.0)
        
        # Camera evidence from pathway
        graph.camera_score = pathway.get("evidence", {}).get("camera_evidence_score", 0.0)
        
        # CRITICAL FIX: Boost camera_score with provenance
        # If provenance is strong, camera_score should be boosted
        if graph.provenance_score >= 0.45:
            # Strong provenance boosts camera score
            camera_boost = graph.provenance_score * 0.4
            graph.camera_score = min(1.0, graph.camera_score + camera_boost)
            graph.rationale.append(f"Camera score boosted by provenance: +{camera_boost:.2f}")
        
        # CRITICAL FIX: Don't penalize missing camera make/model too hard
        # If platform software detected, camera_score should not collapse
        if graph.is_platform_reencoded and graph.camera_score < 0.20:
            graph.camera_score = max(graph.camera_score, 0.20)
            graph.rationale.append("Camera score floor applied (platform re-encoding detected)")
        
        # Generative score
        graph.generative_score = self._compute_generative_score(report, graph)
        
        # Edit assessment
        edit_assessment = report.get("edit_assessment", {})
        graph.tamper_score_local = edit_assessment.get("local_manipulation_score", 0.0)
        graph.edit_score_global = edit_assessment.get("global_adjustment_score", 0.0)
        
        # Domain factors
        domain = report.get("domain", {})
        characteristics = domain.get("characteristics", [])
        graph.is_recompressed = "recompressed" in characteristics or domain.get("type") == "recompressed"
        graph.is_exif_stripped = "exif_stripped" in characteristics
        graph.domain_penalty = domain.get("confidence_penalty", 0.0)
        
        # Quality metrics
        uncertainty = report.get("uncertainty", {})
        ci = uncertainty.get("confidence_interval", [0.3, 0.7])
        graph.ci_width = ci[1] - ci[0] if len(ci) >= 2 else 0.4
        graph.disagreement = uncertainty.get("disagreement", 0.0)
        
        # Evidence quality
        graph.evidence_quality = self._compute_evidence_quality(graph, report)
        
        # AI probability from ensemble
        graph.ai_probability = report.get("ai_probability", 50.0)
        
        # === Compute posteriors ===
        graph.posterior_ai, graph.posterior_photo, graph.posterior_composite = \
            self._compute_posteriors(graph)
        
        return graph
    
    def _compute_generative_score(self, report: Dict, graph: EvidenceGraph) -> float:
        """
        Compute generative evidence score.
        
        Combines:
        - Diffusion fingerprint
        - AI ensemble probability
        - Non-photo score
        - Pathway confidence
        """
        score = 0.0
        
        # Diffusion fingerprint contribution
        if graph.diffusion_score >= 0.70:
            score += 0.40
        elif graph.diffusion_score >= 0.50:
            score += 0.25
        elif graph.diffusion_score >= 0.30:
            score += 0.15
        
        # AI probability contribution (normalized)
        ai_prob = report.get("ai_probability", 50) / 100.0
        if ai_prob >= 0.70:
            score += 0.30
        elif ai_prob >= 0.55:
            score += 0.20
        elif ai_prob >= 0.40:
            score += 0.10
        
        # Non-photo score contribution
        content_type = report.get("content_type", {})
        non_photo_score = content_type.get("non_photo_score", 0.5)
        if non_photo_score >= 0.70:
            score += 0.20
        elif non_photo_score >= 0.55:
            score += 0.10
        
        # Pathway confidence contribution
        if graph.pathway_pred == "t2i":
            if graph.pathway_confidence == "high":
                score += 0.15
            elif graph.pathway_confidence == "medium":
                score += 0.10
        
        # CRITICAL: Reduce generative score if provenance is strong
        # This prevents T2I override when EXIF/GPS strongly supports photo
        if graph.provenance_score >= 0.45:
            reduction = graph.provenance_score * 0.3
            score = max(0.0, score - reduction)
            graph.rationale.append(f"Generative score reduced by provenance: -{reduction:.2f}")
        
        return min(1.0, score)
    
    def _compute_evidence_quality(self, graph: EvidenceGraph, report: Dict) -> float:
        """Compute evidence quality score (0-100)"""
        score = 100.0
        
        # Domain penalties
        if graph.is_recompressed:
            score -= 20
        if graph.is_exif_stripped:
            score -= 20
        
        # Screenshot penalty
        domain = report.get("domain", {})
        if domain.get("type") == "screenshot":
            score -= 15
        
        # JPEG double compression
        jpeg = report.get("jpeg_forensics", {})
        if jpeg.get("double_compression_probability", 0) >= 0.40:
            score -= 10
        
        # Wide CI penalty
        if graph.ci_width >= 0.40:
            score -= 10
        
        # High disagreement penalty
        if graph.disagreement >= 0.45:
            score -= 10
        
        # Provenance bonus (good metadata improves quality)
        if graph.provenance_score >= 0.45:
            score += 10
        
        return max(0, min(100, score))
    
    def _compute_posteriors(self, graph: EvidenceGraph) -> Tuple[float, float, float]:
        """
        Compute posterior probabilities using logistic fusion.
        
        Returns:
            Tuple of (posterior_ai, posterior_photo, posterior_composite)
        """
        w = self.weights
        
        # Logit for AI
        logit_ai = (
            w.w_generative * graph.generative_score
            - w.w_camera * graph.camera_score
            - w.w_provenance * graph.provenance_score
            + w.w_domain * graph.domain_penalty
            + w.w_disagreement * graph.disagreement * 0.5
        )
        
        # Logit for Photo
        logit_photo = (
            w.w_camera * graph.camera_score
            + w.w_provenance * graph.provenance_score
            - w.w_generative * graph.generative_score
            - w.w_domain * graph.domain_penalty
            - w.w_disagreement * graph.disagreement * 0.5
        )
        
        # Logit for Composite
        boundary_corroborated = graph.tamper_score_local >= 0.50
        logit_composite = (
            w.w_local_tamper * graph.tamper_score_local
            + (w.w_boundary if boundary_corroborated else 0)
            - w.w_provenance * graph.provenance_score * 0.5
        )
        
        # Apply sigmoid
        posterior_ai = self._sigmoid(logit_ai)
        posterior_photo = self._sigmoid(logit_photo)
        posterior_composite = self._sigmoid(logit_composite - 1.0)  # Higher threshold for composite
        
        # Normalize so they sum to ~1 (soft normalization)
        total = posterior_ai + posterior_photo + posterior_composite + 0.01
        posterior_ai = posterior_ai / total
        posterior_photo = posterior_photo / total
        posterior_composite = posterior_composite / total
        
        # Log rationale
        graph.rationale.append(f"Logit AI: {logit_ai:.2f} -> {posterior_ai:.2f}")
        graph.rationale.append(f"Logit Photo: {logit_photo:.2f} -> {posterior_photo:.2f}")
        graph.rationale.append(f"Logit Composite: {logit_composite:.2f} -> {posterior_composite:.2f}")
        
        return round(posterior_ai, 3), round(posterior_photo, 3), round(posterior_composite, 3)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with overflow protection"""
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)
    
    def check_consistency(self, 
                          graph: EvidenceGraph,
                          verdict_label: str,
                          report: Dict) -> Dict[str, Any]:
        """
        Check logical consistency between verdict and evidence.
        
        Rules:
        - If headline == AI but provenance_score >= 0.45 and tamper_local < 0.25
          => FAIL, downgrade to PHOTO or INCONCLUSIVE
        - If pathway == T2I high but provenance_score high and tamper_local low
          => Do NOT allow "Likely AI" headline
        - If local tamper high but UI says "global edit only" => fix
        
        Returns:
            Dict with:
            - passed: bool
            - fixes_applied: List[str]
            - corrected_label: str (if fix needed)
        """
        result = {
            "passed": True,
            "fixes_applied": [],
            "corrected_label": verdict_label,
            "rationale": []
        }
        
        # === RULE 1: AI verdict with strong provenance ===
        if verdict_label in ["AI", "AI_HIGH", "AI_MEDIUM", "AI_T2I_HIGH", "AI_T2I_MEDIUM", 
                             "AI_PHOTOREAL_HIGH", "AI_PHOTOREAL_MEDIUM", "AI_GENERATIVE_UNKNOWN"]:
            
            # Check if provenance strongly supports photo
            if graph.provenance_score >= 0.45 and graph.tamper_score_local < 0.25:
                result["passed"] = False
                result["fixes_applied"].append("AI_VERDICT_WITH_STRONG_PROVENANCE")
                result["rationale"].append(
                    f"AI verdict contradicts strong provenance ({graph.provenance_score:.2f}) "
                    f"and low local tamper ({graph.tamper_score_local:.2f})"
                )
                
                # Determine corrected label
                gap = graph.posterior_photo - graph.posterior_ai
                if gap >= 0.15 and graph.evidence_quality >= 55:
                    result["corrected_label"] = "PHOTO_GLOBAL_EDIT"
                else:
                    result["corrected_label"] = "INCONCLUSIVE"
        
        # === RULE 2: T2I pathway with strong provenance ===
        if graph.pathway_pred == "t2i" and graph.pathway_confidence in ["high", "medium"]:
            if graph.provenance_score >= 0.45 and graph.tamper_score_local < 0.25:
                # T2I pathway should NOT override strong provenance
                if verdict_label in ["AI_T2I_HIGH", "AI_T2I_MEDIUM", "AI_PHOTOREAL_HIGH", "AI_PHOTOREAL_MEDIUM"]:
                    result["passed"] = False
                    result["fixes_applied"].append("T2I_OVERRIDE_WITH_STRONG_PROVENANCE")
                    result["rationale"].append(
                        f"T2I pathway ({graph.pathway_confidence}) should not override "
                        f"strong provenance ({graph.provenance_score:.2f})"
                    )
                    
                    gap = graph.posterior_photo - graph.posterior_ai
                    if gap >= 0.15:
                        result["corrected_label"] = "PHOTO_GLOBAL_EDIT"
                    else:
                        result["corrected_label"] = "INCONCLUSIVE"
        
        # === RULE 3: Photo verdict with no camera evidence ===
        if verdict_label in ["REAL_LIKELY", "REAL_MEDIUM", "PHOTO_GLOBAL_EDIT"]:
            if graph.camera_score < 0.15 and graph.provenance_score < 0.30:
                # Photo verdict without camera/provenance evidence
                if graph.generative_score >= 0.50:
                    result["passed"] = False
                    result["fixes_applied"].append("PHOTO_VERDICT_WITHOUT_EVIDENCE")
                    result["rationale"].append(
                        f"Photo verdict without camera evidence ({graph.camera_score:.2f}) "
                        f"or provenance ({graph.provenance_score:.2f})"
                    )
                    result["corrected_label"] = "INCONCLUSIVE"
        
        # === RULE 4: Composite verdict without boundary corroboration ===
        if verdict_label in ["COMPOSITE_LIKELY", "COMPOSITE_POSSIBLE"]:
            edit_assessment = report.get("edit_assessment", {})
            boundary_corroborated = edit_assessment.get("boundary_corroborated", False)
            
            if not boundary_corroborated and graph.tamper_score_local < 0.60:
                result["passed"] = False
                result["fixes_applied"].append("COMPOSITE_WITHOUT_BOUNDARY")
                result["rationale"].append(
                    "Composite verdict without boundary corroboration"
                )
                result["corrected_label"] = "INCONCLUSIVE"
        
        return result


# Global instance
_evidence_fusion: Optional[EvidenceFusion] = None

def get_evidence_fusion() -> EvidenceFusion:
    global _evidence_fusion
    if _evidence_fusion is None:
        _evidence_fusion = EvidenceFusion()
    return _evidence_fusion
