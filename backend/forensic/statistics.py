"""
Statistical Analysis Module
============================
Ensemble, calibration, uncertainty quantification, ECE.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelScore:
    name: str
    raw: float
    calibrated: float
    weight: float
    latency_ms: float = 0.0

class StatisticalAnalyzer:
    """Statistical analysis for AI detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.inconclusive_band = self.config.get("inconclusive_band", (0.4, 0.6))
        self.disagreement_threshold = self.config.get("disagreement_threshold", 0.35)
        
        # Calibration parameters (can be loaded from file)
        self.calibration = {
            "method": "none",
            "per_model": {},
            "ensemble": {}
        }
        
        # Known ECE from benchmarks
        self.known_ece = {
            "overall": 0.12,
            "recompressed": 0.18,
            "screenshot": 0.22,
            "dataset_version": "v1.0"
        }
    
    def analyze(self, model_scores: List, domain: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis
        
        Args:
            model_scores: List of ModelScore objects or dicts with score data
            domain: Domain detection results
        """
        result = {
            "models": [],
            "ensemble": {},
            "ece": None,  # Use None instead of empty dict for missing data
            "disagreement": 0.0,
            "confidence_interval": [0.3, 0.7],
            "expected_error_rate": None,  # Use None for missing data
            "inconclusive_reason": None,
            "calibration_applied": False,
            "model_agreement": 1.0
        }
        
        if not model_scores:
            return result
        
        # Convert dicts to ModelScore if needed
        scores = []
        for s in model_scores:
            if isinstance(s, ModelScore):
                scores.append(s)
            elif isinstance(s, dict):
                scores.append(ModelScore(
                    name=s.get("name", "unknown"),
                    raw=s.get("raw_score", s.get("raw", 0.5)),
                    calibrated=s.get("calibrated_score", s.get("calibrated", 0.5)),
                    weight=s.get("weight", 1.0),
                    latency_ms=s.get("latency_ms", 0)
                ))
        
        if not scores:
            return result
        
        # Process model scores
        for score in scores:
            calibrated = self._calibrate_score(score.name, score.raw)
            result["models"].append({
                "name": score.name,
                "raw": round(score.raw, 4),
                "calibrated": round(calibrated, 4),
                "weight": score.weight,
                "latency_ms": round(score.latency_ms, 1)
            })
        
        # Calculate ensemble
        result["ensemble"] = self._calculate_ensemble(scores)
        
        # Calculate disagreement
        raw_scores = [s.raw for s in scores]
        result["disagreement"] = round(max(raw_scores) - min(raw_scores), 4) if raw_scores else 0
        result["model_agreement"] = 1.0 - result["disagreement"]
        
        # Calculate confidence interval
        result["confidence_interval"] = self._calculate_ci(raw_scores, domain)
        
        # ECE from benchmarks - return None if not available
        domain_type = domain.get("domain_type", "overall") if domain else "overall"
        ece_value = self.known_ece.get(domain_type, self.known_ece.get("overall"))
        
        # Sanitize NaN values
        if ece_value is not None and not np.isnan(ece_value):
            result["ece"] = {
                "value": float(ece_value),
                "dataset": self.known_ece["dataset_version"],
                "domain": domain_type
            }
        else:
            result["ece"] = None
        
        # Expected error rate - sanitize NaN
        error_rate = self._calculate_error_rate(
            result["ensemble"]["calibrated"],
            result["disagreement"],
            domain
        )
        # Check for NaN values
        if error_rate:
            sanitized_error = {}
            for k, v in error_rate.items():
                if v is not None and not np.isnan(v):
                    sanitized_error[k] = float(v)
            result["expected_error_rate"] = sanitized_error if sanitized_error else None
        else:
            result["expected_error_rate"] = None
        
        # Check for inconclusive
        result["inconclusive_reason"] = self._check_inconclusive(
            result["ensemble"]["calibrated"],
            result["disagreement"],
            domain
        )
        
        return result
    
    def _calibrate_score(self, model_name: str, raw_score: float) -> float:
        """Apply calibration to raw score"""
        # Placeholder - would load from calibration.json
        # For now, apply simple Platt-like scaling
        
        if model_name in self.calibration.get("per_model", {}):
            params = self.calibration["per_model"][model_name]
            # Platt scaling: 1 / (1 + exp(A*score + B))
            a = params.get("a", 1.0)
            b = params.get("b", 0.0)
            return 1.0 / (1.0 + np.exp(-(a * raw_score + b)))
        
        return raw_score
    
    def _calculate_ensemble(self, scores: List[ModelScore]) -> Dict:
        """Calculate weighted ensemble"""
        if not scores:
            return {"raw": 0.5, "calibrated": 0.5, "weights": {}, "method": "none"}
        
        total_weight = sum(s.weight for s in scores)
        
        raw_ensemble = sum(s.raw * s.weight for s in scores) / total_weight
        calibrated_ensemble = sum(
            self._calibrate_score(s.name, s.raw) * s.weight for s in scores
        ) / total_weight
        
        weights = {s.name: round(s.weight / total_weight, 3) for s in scores}
        
        return {
            "raw": round(raw_ensemble, 4),
            "calibrated": round(calibrated_ensemble, 4),
            "weights": weights,
            "method": "weighted_average"
        }
    
    def _calculate_ci(self, scores: List[float], domain: Dict = None) -> List[float]:
        """Calculate confidence interval with domain adjustment"""
        if len(scores) < 2:
            return [0.3, 0.7]
        
        # Bootstrap CI
        n_bootstrap = 200
        means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            means.append(np.mean(sample))
        
        ci_low = np.percentile(means, 5)
        ci_high = np.percentile(means, 95)
        
        # Domain penalty
        if domain:
            penalty = 0.0
            characteristics = domain.get("characteristics", [])
            
            if "screenshot" in characteristics:
                penalty += 0.08
            if "recompressed" in characteristics:
                penalty += 0.06
            if "exif_stripped" in characteristics:
                penalty += 0.04
            
            ci_low = max(0.0, ci_low - penalty)
            ci_high = min(1.0, ci_high + penalty)
        
        return [round(ci_low, 3), round(ci_high, 3)]
    
    def _calculate_error_rate(self, calibrated_p: float, disagreement: float,
                             domain: Dict = None) -> Dict:
        """Calculate expected error rate"""
        # Base error rate from ECE
        base_error = self.known_ece["overall"]
        
        # Adjust for uncertainty
        if 0.4 <= calibrated_p <= 0.6:
            base_error += 0.10  # Higher error in uncertain region
        
        if disagreement > 0.3:
            base_error += 0.05  # Model disagreement increases error
        
        # Domain adjustment
        domain_adjusted = base_error
        if domain:
            domain_type = domain.get("domain_type", "original")
            if domain_type in self.known_ece:
                domain_adjusted = self.known_ece[domain_type]
            
            # Additional penalties
            if "recompressed" in domain.get("characteristics", []):
                domain_adjusted += 0.05
            if "screenshot" in domain.get("characteristics", []):
                domain_adjusted += 0.08
        
        return {
            "overall": round(min(0.5, base_error), 3),
            "domain_adjusted": round(min(0.5, domain_adjusted), 3)
        }
    
    def _check_inconclusive(self, calibrated_p: float, disagreement: float,
                           domain: Dict = None) -> Optional[str]:
        """Check if result should be marked inconclusive"""
        reasons = []
        
        # Check if in uncertain band with high disagreement
        low, high = self.inconclusive_band
        if low <= calibrated_p <= high and disagreement > self.disagreement_threshold:
            reasons.append("p_in_band_and_disagreement_high")
        
        # Domain penalties
        if domain:
            chars = domain.get("characteristics", [])
            severe_domain = sum(1 for c in ["recompressed", "exif_stripped", "screenshot"] if c in chars)
            
            if severe_domain >= 2:
                reasons.append("domain_penalty_severe")
            
            if domain.get("confidence_penalty", 0) > 0.2:
                reasons.append("high_domain_penalty")
        
        return "|".join(reasons) if reasons else None


def analyze_statistics(model_scores: List[Dict], domain: Dict = None) -> Dict[str, Any]:
    """Convenience function"""
    scores = [
        ModelScore(
            name=s["name"],
            raw=s.get("raw_score", s.get("raw", 0.5)),
            calibrated=s.get("calibrated_score", s.get("calibrated", 0.5)),
            weight=s.get("weight", 1.0),
            latency_ms=s.get("latency_ms", 0)
        )
        for s in model_scores
    ]
    
    analyzer = StatisticalAnalyzer()
    return analyzer.analyze(scores, domain)
