"""
Forensic AI Detection Module - Production Grade
================================================
Multi-model ensemble with calibration and uncertainty quantification.

Evidence Levels:
- DEFINITIVE: Cryptographic proof (C2PA) or AI software signature
- STRONG_EVIDENCE: Multiple corroborating signals
- STATISTICAL: ML model predictions with measured uncertainty
- INCONCLUSIVE: Insufficient evidence

This module NEVER claims certainty without definitive evidence.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EvidenceLevel:
    DEFINITIVE_AI = "DEFINITIVE_AI"
    DEFINITIVE_REAL = "DEFINITIVE_REAL"
    STRONG_AI_EVIDENCE = "STRONG_AI_EVIDENCE"
    STRONG_REAL_EVIDENCE = "STRONG_REAL_EVIDENCE"
    STATISTICAL_AI = "STATISTICAL_AI"
    STATISTICAL_REAL = "STATISTICAL_REAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class AILikelihood:
    """AI likelihood levels for two-axis output"""
    HIGH = "high"      # >= 0.75
    MEDIUM = "medium"  # 0.45 - 0.75
    LOW = "low"        # <= 0.45


class EvidentialQuality:
    """Evidential quality levels for two-axis output"""
    HIGH = "high"      # Strong metadata, low disagreement, no domain penalties
    MEDIUM = "medium"  # Some metadata or moderate disagreement
    LOW = "low"        # Missing metadata, high disagreement, domain penalties

@dataclass
class ModelResult:
    name: str
    raw_score: float
    calibrated_score: float
    weight: float
    latency_ms: float

class ForensicAIDetector:
    """
    Production-grade forensic AI detector with:
    - Multi-model ensemble
    - Calibrated probabilities
    - Uncertainty quantification
    - Domain-aware adjustments
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models
        self.models = {}
        self.clip_model = None
        self.clip_preprocess = None
        
        # Calibration (placeholder - can be loaded from file)
        self.calibration = {
            "method": "none",
            "per_model": {},
            "ensemble": {}
        }
        
    async def load_model(self):
        """Load all detection models"""
        await self._load_primary_model()
        await self._load_secondary_model()
        await self._load_clip()
        
    async def _load_primary_model(self):
        """Load primary AI detector"""
        try:
            from transformers import pipeline
            logger.info("Loading primary model: umm-maybe/AI-image-detector")
            self.models["primary"] = {
                "pipeline": pipeline(
                    "image-classification",
                    model="umm-maybe/AI-image-detector",
                    device=0 if self.device == "cuda" else -1
                ),
                "weight": 1.0,
                "label_map": {"artificial": "ai", "human": "real"}
            }
            logger.info("✅ Primary model loaded")
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            
    async def _load_secondary_model(self):
        """Load secondary AI detector for ensemble"""
        try:
            from transformers import pipeline
            logger.info("Loading secondary model: Organika/sdxl-detector")
            self.models["secondary"] = {
                "pipeline": pipeline(
                    "image-classification",
                    model="Organika/sdxl-detector",
                    device=0 if self.device == "cuda" else -1
                ),
                "weight": 0.8,
                "label_map": {"AI": "ai", "Real": "real"}
            }
            logger.info("✅ Secondary model loaded")
        except Exception as e:
            logger.warning(f"Secondary model failed (optional): {e}")
            
    async def _load_clip(self):
        """Load CLIP for semantic analysis"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("✅ CLIP model loaded")
        except Exception as e:
            logger.warning(f"CLIP failed (optional): {e}")


    async def analyze(self, image: Image.Image, metadata: Dict[str, Any],
                     domain: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis entry point
        
        Returns comprehensive forensic report with:
        - Evidence level classification
        - Ensemble predictions with uncertainty
        - Domain-adjusted confidence intervals
        - Two-axis output (AI likelihood + Evidential quality)
        """
        start_time = datetime.now()
        
        report = {
            "evidence_level": EvidenceLevel.INCONCLUSIVE,
            "ai_probability": 50,
            "confidence": 30,
            "models": [],
            "ensemble": {},
            "uncertainty": {},
            "domain": domain or {},
            "definitive_findings": [],
            "strong_evidence": [],
            "statistical_indicators": [],
            "authenticity_indicators": [],
            "recommendation": "",
            "calibration_applied": False,
            "summary_axes": None  # Two-axis output
        }
        
        try:
            # PHASE 1: Check for definitive evidence
            definitive = self._check_definitive_evidence(metadata)
            if definitive["found"]:
                report.update(definitive["report"])
                report["analysis_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
                return report
            
            # PHASE 2: Run ensemble models
            ensemble_result = await self._run_ensemble(image)
            report["models"] = ensemble_result["models"]
            report["ensemble"] = ensemble_result["ensemble"]
            report["uncertainty"] = ensemble_result["uncertainty"]
            
            # PHASE 3: Technical analysis
            technical = self._technical_analysis(image)
            report["statistical_indicators"].extend(technical["indicators"])
            
            # PHASE 4: Authenticity indicators
            authenticity = self._check_authenticity(metadata)
            report["authenticity_indicators"] = authenticity["indicators"]
            
            # PHASE 5: Domain adjustments
            if domain:
                report = self._apply_domain_adjustments(report, domain)
            
            # PHASE 6: Compute two-axis summary (before verdict)
            report["summary_axes"] = self._compute_summary_axes(
                report, metadata, authenticity, domain or {}
            )
            
            # PHASE 7: Final verdict (uses new two-axis logic)
            report = self._determine_verdict(report, metadata, authenticity)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            report["error"] = str(e)
        
        report["analysis_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        return report
    
    def _check_definitive_evidence(self, metadata: Dict) -> Dict:
        """Check for definitive evidence (AI signature or C2PA)"""
        result = {"found": False, "report": {}}
        
        # Check AI software detection
        ai_detection = metadata.get("ai_detection", {})
        if ai_detection.get("detected"):
            result["found"] = True
            result["report"] = {
                "evidence_level": EvidenceLevel.DEFINITIVE_AI,
                "ai_probability": 98,
                "confidence": 98,
                "definitive_findings": [
                    f"AI yazılım imzası tespit edildi: {ai_detection.get('software')}",
                    f"Üretici: {ai_detection.get('vendor')}",
                    f"Tür: {ai_detection.get('type')}"
                ],
                "recommendation": "Bu görsel KESİN olarak AI tarafından üretilmiştir. Metadata'da AI yazılım imzası bulundu."
            }
            logger.info(f"🚨 DEFINITIVE AI: {ai_detection.get('software')}")
        
        return result
    
    async def _run_ensemble(self, image: Image.Image) -> Dict:
        """Run all models and calculate ensemble"""
        results = {
            "models": [],
            "ensemble": {"raw": 0.5, "calibrated": 0.5},
            "uncertainty": {}
        }
        
        model_scores = []
        
        # Run each model
        for name, model_info in self.models.items():
            try:
                start = datetime.now()
                score = await self._run_single_model(model_info, image)
                latency = (datetime.now() - start).total_seconds() * 1000
                
                results["models"].append({
                    "name": name,
                    "raw_score": round(score, 4),
                    "calibrated_score": round(score, 4),  # TODO: apply calibration
                    "weight": model_info["weight"],
                    "latency_ms": round(latency, 1)
                })
                model_scores.append((score, model_info["weight"]))
                logger.info(f"  {name}: {score:.3f}")
                
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        # Run CLIP
        clip_score = await self._run_clip(image)
        if clip_score is not None:
            results["models"].append({
                "name": "clip_semantic",
                "raw_score": round(clip_score, 4),
                "calibrated_score": round(clip_score, 4),
                "weight": 0.6,
                "latency_ms": 0
            })
            model_scores.append((clip_score, 0.6))
            logger.info(f"  clip_semantic: {clip_score:.3f}")
        
        # Calculate ensemble
        if model_scores:
            results["ensemble"] = self._calculate_ensemble(model_scores)
            results["uncertainty"] = self._calculate_uncertainty(model_scores)
        
        return results
    
    async def _run_single_model(self, model_info: Dict, image: Image.Image) -> float:
        """Run a single model and normalize to AI probability"""
        pipeline = model_info["pipeline"]
        label_map = model_info["label_map"]
        
        predictions = pipeline(image)
        
        ai_score = 0.5
        for pred in predictions:
            label = pred['label']
            score = pred['score']
            
            for key, value in label_map.items():
                if key.lower() in label.lower():
                    if value == "ai":
                        ai_score = score
                    else:
                        ai_score = 1.0 - score
                    break
        
        return ai_score
    
    async def _run_clip(self, image: Image.Image) -> Optional[float]:
        """Run CLIP semantic analysis"""
        if not self.clip_model:
            return None
        
        try:
            import clip
            
            ai_labels = [
                "a 3D rendered image", "a CGI character", "an AI generated image",
                "a digital artwork", "a synthetic photo", "a computer generated face",
                "a hyperrealistic render", "a video game screenshot"
            ]
            real_labels = [
                "a real photograph", "a camera photo", "a natural photograph",
                "a candid photo", "a genuine photograph", "a smartphone photo"
            ]
            
            all_labels = ai_labels + real_labels
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = clip.tokenize(all_labels).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()
            
            ai_prob = float(sum(probs[:len(ai_labels)]))
            return ai_prob
            
        except Exception as e:
            logger.warning(f"CLIP failed: {e}")
            return None
    
    def _calculate_ensemble(self, scores: List[tuple]) -> Dict:
        """Calculate weighted ensemble"""
        if not scores:
            return {"raw": 0.5, "calibrated": 0.5, "method": "none"}
        
        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)
        
        raw = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            "raw": round(raw, 4),
            "calibrated": round(raw, 4),  # TODO: apply calibration
            "method": "weighted_average",
            "model_count": len(scores)
        }
    
    def _calculate_uncertainty(self, scores: List[tuple]) -> Dict:
        """Calculate uncertainty metrics"""
        if len(scores) < 2:
            return {
                "disagreement": 0,
                "entropy": 0.5,
                "confidence_interval": [0.3, 0.7],
                "std": 0
            }
        
        raw_scores = [s for s, _ in scores]
        ensemble = np.mean(raw_scores)
        
        # Disagreement
        disagreement = max(raw_scores) - min(raw_scores)
        
        # Entropy
        p = max(0.001, min(0.999, ensemble))
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        # Bootstrap CI
        ci = self._bootstrap_ci(raw_scores)
        
        return {
            "disagreement": round(disagreement, 4),
            "entropy": round(entropy, 4),
            "confidence_interval": ci,
            "std": round(np.std(raw_scores), 4)
        }
    
    def _bootstrap_ci(self, scores: List[float], n: int = 100) -> List[float]:
        """Bootstrap confidence interval"""
        if len(scores) < 2:
            return [0.3, 0.7]
        
        means = []
        for _ in range(n):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            means.append(np.mean(sample))
        
        return [round(np.percentile(means, 5), 3), round(np.percentile(means, 95), 3)]


    def _technical_analysis(self, image: Image.Image) -> Dict:
        """Pixel-level technical analysis"""
        indicators = []
        
        try:
            img_array = np.array(image.convert('RGB')).astype(np.float32)
            gray = np.mean(img_array, axis=2)
            
            # Noise analysis
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(gray, sigma=1.5)
            noise = gray - smoothed
            noise_std = np.std(noise)
            
            if noise_std >= 3 and noise_std <= 15:
                indicators.append({
                    "type": "natural_noise",
                    "description": f"Doğal sensör gürültüsü (σ={noise_std:.1f})",
                    "suggests": "real",
                    "weight": 0.2
                })
            elif noise_std < 2:
                indicators.append({
                    "type": "low_noise",
                    "description": f"Çok düşük gürültü (σ={noise_std:.1f})",
                    "suggests": "ai",
                    "weight": 0.15
                })
            
            # Texture analysis
            from scipy.ndimage import uniform_filter
            local_var = uniform_filter(gray**2, size=5) - uniform_filter(gray, size=5)**2
            avg_var = np.mean(local_var)
            
            if avg_var < 50:
                indicators.append({
                    "type": "smooth_texture",
                    "description": "Aşırı pürüzsüz doku",
                    "suggests": "ai",
                    "weight": 0.2
                })
            elif avg_var > 200:
                indicators.append({
                    "type": "natural_texture",
                    "description": "Doğal doku varyansı",
                    "suggests": "real",
                    "weight": 0.15
                })
            
            # Edge analysis
            from scipy.ndimage import sobel
            edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
            edge_std = np.std(edges)
            edge_mean = np.mean(edges)
            
            if edge_mean > 0:
                edge_ratio = edge_std / edge_mean
                if edge_ratio > 1.5:
                    indicators.append({
                        "type": "natural_edges",
                        "description": "Doğal kenar varyasyonu",
                        "suggests": "real",
                        "weight": 0.1
                    })
                elif edge_ratio < 0.8:
                    indicators.append({
                        "type": "uniform_edges",
                        "description": "Yapay kenar düzgünlüğü",
                        "suggests": "ai",
                        "weight": 0.1
                    })
                    
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
        
        return {"indicators": indicators}
    
    def _check_authenticity(self, metadata: Dict) -> Dict:
        """Check authenticity indicators from metadata"""
        indicators = []
        score = 0
        
        camera = metadata.get("camera", {})
        gps = metadata.get("gps", {})
        technical = metadata.get("technical", {})
        
        # Camera info
        if camera.get("has_camera_info"):
            indicators.append({
                "type": "camera_info",
                "description": f"Kamera: {camera.get('make', '')} {camera.get('model', '')}",
                "weight": 25
            })
            score += 25
            
            if camera.get("is_known_maker"):
                indicators.append({
                    "type": "known_maker",
                    "description": "Bilinen kamera üreticisi",
                    "weight": 15
                })
                score += 15
        
        # GPS
        if gps.get("present") and gps.get("confidence") in ["high", "medium"]:
            indicators.append({
                "type": "gps_data",
                "description": f"GPS: {gps.get('latitude')}, {gps.get('longitude')}",
                "weight": 20
            })
            score += 20
        
        # Technical settings
        tech_fields = ["exposure_time", "f_number", "iso", "focal_length"]
        tech_count = sum(1 for f in tech_fields if technical.get(f))
        
        if tech_count >= 3:
            indicators.append({
                "type": "exposure_settings",
                "description": f"Pozlama ayarları mevcut ({tech_count}/4)",
                "weight": 20
            })
            score += 20
        
        # Timestamps
        if metadata.get("timestamps", {}).get("original"):
            indicators.append({
                "type": "timestamp",
                "description": f"Çekim tarihi: {metadata['timestamps']['original']}",
                "weight": 10
            })
            score += 10
        
        return {
            "indicators": indicators,
            "score": min(score, 100),
            "is_likely_authentic": score >= 40
        }
    
    def _apply_domain_adjustments(self, report: Dict, domain: Dict) -> Dict:
        """Apply domain-based adjustments to confidence"""
        penalty = domain.get("confidence_penalty", 0)
        ci_expansion = domain.get("ci_expansion", 0)
        
        if penalty > 0:
            report["confidence"] = max(20, report.get("confidence", 50) - int(penalty * 100))
        
        if ci_expansion > 0 and "uncertainty" in report:
            ci = report["uncertainty"].get("confidence_interval", [0.3, 0.7])
            report["uncertainty"]["confidence_interval"] = [
                max(0, ci[0] - ci_expansion),
                min(1, ci[1] + ci_expansion)
            ]
        
        if domain.get("warnings"):
            report["domain_warnings"] = domain["warnings"]
        
        return report
    
    def _compute_summary_axes(self, report: Dict, metadata: Dict, 
                               authenticity: Dict, domain: Dict) -> Dict:
        """
        Compute two-axis output: AI likelihood vs Evidential quality
        This separates "what we think" from "how confident we are in the evidence"
        """
        ensemble_score = report.get("ensemble", {}).get("raw", 0.5)
        calibrated_score = report.get("ensemble", {}).get("calibrated", ensemble_score)
        uncertainty = report.get("uncertainty", {})
        ci = uncertainty.get("confidence_interval", [0.3, 0.7])
        disagreement = uncertainty.get("disagreement", 0)
        auth_score = authenticity.get("score", 0)
        
        # === AI LIKELIHOOD AXIS ===
        # Based on calibrated probability and CI
        if calibrated_score >= 0.75:
            ai_level = AILikelihood.HIGH
        elif calibrated_score <= 0.25:
            ai_level = AILikelihood.LOW
        else:
            ai_level = AILikelihood.MEDIUM
        
        ai_likelihood = {
            "level": ai_level,
            "probability": round(calibrated_score, 4),
            "confidence_interval": [round(ci[0], 3), round(ci[1], 3)],
            "ci_width": round(ci[1] - ci[0], 3)
        }
        
        # === EVIDENTIAL QUALITY AXIS ===
        quality_reasons = []
        quality_score = 100  # Start at 100, subtract penalties
        
        # Domain penalties (lower quality but don't change verdict)
        domain_penalty = domain.get("confidence_penalty", 0) if domain else 0
        if domain_penalty > 0.1:
            quality_score -= 30
            quality_reasons.append("domain_penalty_high")
        elif domain_penalty > 0.05:
            quality_score -= 15
            quality_reasons.append("domain_penalty_moderate")
        
        # Model disagreement
        if disagreement > 0.4:
            quality_score -= 25
            quality_reasons.append("high_model_disagreement")
        elif disagreement > 0.25:
            quality_score -= 12
            quality_reasons.append("moderate_model_disagreement")
        
        # Metadata quality
        if auth_score >= 50:
            quality_score += 15
            quality_reasons.append("strong_metadata")
        elif auth_score < 20:
            quality_score -= 15
            quality_reasons.append("weak_metadata")
        
        # CI width (wider = less certain)
        ci_width = ci[1] - ci[0]
        if ci_width > 0.4:
            quality_score -= 20
            quality_reasons.append("wide_confidence_interval")
        elif ci_width > 0.25:
            quality_score -= 10
            quality_reasons.append("moderate_confidence_interval")
        
        # Determine quality level
        quality_score = max(0, min(100, quality_score))
        if quality_score >= 70:
            quality_level = EvidentialQuality.HIGH
        elif quality_score >= 40:
            quality_level = EvidentialQuality.MEDIUM
        else:
            quality_level = EvidentialQuality.LOW
        
        evidential_quality = {
            "level": quality_level,
            "score": quality_score,
            "reasons": quality_reasons
        }
        
        return {
            "ai_likelihood": ai_likelihood,
            "evidential_quality": evidential_quality
        }

    def _determine_verdict(self, report: Dict, metadata: Dict, authenticity: Dict) -> Dict:
        """
        Determine final verdict based on all evidence.
        
        NEW LOGIC (Two-Axis):
        - Domain penalties affect evidential_quality but NOT verdict directly
        - If calibrated_p >= 0.75 AND CI_low >= 0.55 → "Likely AI" even with low quality
        - If calibrated_p <= 0.25 AND CI_high <= 0.45 → "Likely real" even with low quality
        - INCONCLUSIVE only when probability is truly uncertain (0.35-0.65 range)
        """
        
        ensemble_score = report.get("ensemble", {}).get("raw", 0.5)
        calibrated_score = report.get("ensemble", {}).get("calibrated", ensemble_score)
        auth_score = authenticity.get("score", 0)
        uncertainty = report.get("uncertainty", {})
        disagreement = uncertainty.get("disagreement", 0)
        ci = uncertainty.get("confidence_interval", [0.3, 0.7])
        ci_low, ci_high = ci[0], ci[1]
        
        # Calculate AI indicators from technical analysis
        ai_indicators = sum(1 for ind in report.get("statistical_indicators", []) 
                          if ind.get("suggests") == "ai")
        real_indicators = sum(1 for ind in report.get("statistical_indicators", [])
                            if ind.get("suggests") == "real")
        
        # Get domain info for logging
        domain_penalty = report.get("domain", {}).get("confidence_penalty", 0)
        
        logger.info(f"Verdict calc: ensemble={ensemble_score:.2f}, calibrated={calibrated_score:.2f}, "
                   f"auth={auth_score}, ai_ind={ai_indicators}, real_ind={real_indicators}, "
                   f"disagree={disagreement:.2f}, ci=[{ci_low:.2f},{ci_high:.2f}], domain_penalty={domain_penalty:.2f}")
        
        # CASE 0: Definitive evidence already set (AI signature found)
        if report.get("evidence_level") in [EvidenceLevel.DEFINITIVE_AI, EvidenceLevel.DEFINITIVE_REAL]:
            return report
        
        # CASE 1: Strong authenticity evidence (camera, GPS, EXIF)
        if auth_score >= 50:
            if ensemble_score > 0.6:
                # High AI score but strong real evidence - trust evidence
                report["evidence_level"] = EvidenceLevel.STRONG_REAL_EVIDENCE
                report["ai_probability"] = max(20, int(ensemble_score * 50))
                report["confidence"] = min(85, auth_score + 20)
                report["recommendation"] = (
                    "Güçlü gerçeklik kanıtları mevcut (kamera, EXIF, GPS). "
                    "ML modelleri yüksek AI skoru verse de, metadata kanıtları "
                    "bu görselin gerçek bir fotoğraf olduğunu destekliyor."
                )
                report["strong_evidence"] = authenticity["indicators"]
            else:
                report["evidence_level"] = EvidenceLevel.STRONG_REAL_EVIDENCE
                report["ai_probability"] = max(10, int(ensemble_score * 100) - auth_score // 2)
                report["confidence"] = min(90, auth_score + 30)
                report["recommendation"] = (
                    "Bu görsel büyük olasılıkla gerçek bir fotoğraftır. "
                    "Kamera metadata'sı ve teknik göstergeler bunu destekliyor."
                )
                report["strong_evidence"] = authenticity["indicators"]
            return report
        
        # === NEW TWO-AXIS LOGIC ===
        # High AI likelihood with reasonable CI → STRONG_AI even with domain penalties
        if calibrated_score >= 0.75 and ci_low >= 0.55:
            report["evidence_level"] = EvidenceLevel.STRONG_AI_EVIDENCE
            report["ai_probability"] = int(calibrated_score * 100)
            report["confidence"] = min(80, int((1 - disagreement) * 70) + 20)
            report["recommendation"] = (
                "Güçlü AI üretim göstergeleri mevcut. Model tahminleri yüksek AI olasılığı "
                f"gösteriyor (güven aralığı: %{int(ci_low*100)}-%{int(ci_high*100)}). "
                "Ancak kesin kanıt (AI yazılım imzası) bulunamadı."
            )
            return report
        
        # Low AI likelihood with reasonable CI → STATISTICAL_REAL
        if calibrated_score <= 0.25 and ci_high <= 0.45:
            report["evidence_level"] = EvidenceLevel.STATISTICAL_REAL
            report["ai_probability"] = int(calibrated_score * 100)
            report["confidence"] = max(50, int((1 - disagreement) * 60) + 20)
            report["recommendation"] = (
                "İstatistiksel analiz bu görselin gerçek bir fotoğraf olduğunu destekliyor. "
                f"AI olasılığı düşük (güven aralığı: %{int(ci_low*100)}-%{int(ci_high*100)})."
            )
            return report
        
        # CASE 2: High AI score + no authenticity + AI indicators
        if ensemble_score >= 0.65 and auth_score < 20 and ai_indicators >= real_indicators:
            report["evidence_level"] = EvidenceLevel.STRONG_AI_EVIDENCE
            report["ai_probability"] = int(ensemble_score * 100)
            report["confidence"] = min(80, int((1 - disagreement) * 70) + 20)
            report["recommendation"] = (
                "Güçlü AI üretim göstergeleri mevcut. Metadata eksikliği ve "
                "teknik analiz AI üretimini destekliyor. Ancak kesin kanıt "
                "(AI yazılım imzası) bulunamadı."
            )
            return report
        
        # CASE 3: Statistical AI (moderate-high probability)
        if ensemble_score >= 0.55:
            report["evidence_level"] = EvidenceLevel.STATISTICAL_AI
            report["ai_probability"] = int(ensemble_score * 100)
            report["confidence"] = max(30, int((1 - disagreement) * 50) + 20)
            report["recommendation"] = (
                f"⚠️ İSTATİSTİKSEL TAHMİN: ML modelleri %{int(ensemble_score*100)} AI olasılığı "
                f"tahmin ediyor (güven aralığı: %{int(ci_low*100)}-%{int(ci_high*100)}). "
                "Bu kesin bir sonuç DEĞİLDİR. Kesin sonuç için AI yazılım imzası gereklidir."
            )
            return report
        
        # CASE 4: Statistical Real (low-moderate probability)
        if ensemble_score <= 0.45:
            report["evidence_level"] = EvidenceLevel.STATISTICAL_REAL
            report["ai_probability"] = int(ensemble_score * 100)
            report["confidence"] = max(40, int((1 - disagreement) * 50) + 20)
            report["recommendation"] = (
                "İstatistiksel analiz bu görselin gerçek bir fotoğraf olduğunu destekliyor. "
                f"AI olasılığı: %{int(ensemble_score*100)} (güven aralığı: %{int(ci_low*100)}-%{int(ci_high*100)})."
            )
            return report
        
        # CASE 5: Truly Inconclusive (probability in 0.45-0.55 range)
        report["evidence_level"] = EvidenceLevel.INCONCLUSIVE
        report["ai_probability"] = int(ensemble_score * 100)
        report["confidence"] = 30
        report["recommendation"] = (
            f"⚠️ BELİRSİZ: AI olasılığı belirsiz bölgede (%{int(ensemble_score*100)}). "
            f"Güven aralığı: %{int(ci_low*100)}-%{int(ci_high*100)}. "
            "Ne AI üretimi ne de gerçek fotoğraf olduğu kesin olarak belirlenemiyor."
        )
        
        return report


# Backward compatibility
CalibratedAIDetector = ForensicAIDetector