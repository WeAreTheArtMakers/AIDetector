"""
Model Registry - Multi-Model Ensemble System
=============================================
Manages multiple AI detection models for ensemble predictions.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image
import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelSpec:
    """Model specification"""
    name: str
    hf_id: str
    task: str  # image-classification, zero-shot-classification
    weight: float = 1.0
    label_mapping: Dict[str, str] = None  # maps model labels to ai/real
    enabled: bool = True

# Default model configurations
DEFAULT_MODELS = [
    ModelSpec(
        name="ai-image-detector",
        hf_id="umm-maybe/AI-image-detector",
        task="image-classification",
        weight=1.0,
        label_mapping={"artificial": "ai", "human": "real"}
    ),
    ModelSpec(
        name="sdxl-detector", 
        hf_id="Organika/sdxl-detector",
        task="image-classification",
        weight=0.8,
        label_mapping={"AI": "ai", "Real": "real"}
    ),
]

class ModelRegistry:
    """Registry for AI detection models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.specs: Dict[str, ModelSpec] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.clip_preprocess = None
        
    async def load_models(self, specs: List[ModelSpec] = None):
        """Load all registered models"""
        specs = specs or DEFAULT_MODELS
        
        for spec in specs:
            if not spec.enabled:
                continue
            await self._load_model(spec)
        
        # Load CLIP for semantic analysis
        await self._load_clip()
        
    async def _load_model(self, spec: ModelSpec):
        """Load a single model"""
        try:
            from transformers import pipeline
            
            logger.info(f"Loading model: {spec.name} ({spec.hf_id})")
            
            model = pipeline(
                spec.task,
                model=spec.hf_id,
                device=0 if self.device == "cuda" else -1
            )
            
            self.models[spec.name] = model
            self.specs[spec.name] = spec
            logger.info(f"✅ Loaded: {spec.name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {spec.name}: {e}")
            
    async def _load_clip(self):
        """Load CLIP model for semantic analysis"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("✅ CLIP model loaded")
        except Exception as e:
            logger.warning(f"⚠️ CLIP load failed: {e}")


    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run all models and return ensemble prediction
        """
        results = {
            "models": [],
            "ensemble": {},
            "uncertainty": {}
        }
        
        model_scores = []
        
        # Run each model
        for name, model in self.models.items():
            spec = self.specs[name]
            try:
                score = await self._run_model(model, spec, image)
                results["models"].append({
                    "name": name,
                    "raw_score": score,
                    "weight": spec.weight
                })
                model_scores.append((score, spec.weight))
                logger.info(f"  {name}: {score:.3f}")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        
        # Run CLIP semantic analysis
        clip_score = await self._run_clip_analysis(image)
        if clip_score is not None:
            results["models"].append({
                "name": "clip-semantic",
                "raw_score": clip_score,
                "weight": 0.7
            })
            model_scores.append((clip_score, 0.7))
            logger.info(f"  clip-semantic: {clip_score:.3f}")
        
        # Calculate ensemble
        if model_scores:
            ensemble = self._calculate_ensemble(model_scores)
            results["ensemble"] = ensemble
            results["uncertainty"] = self._calculate_uncertainty(model_scores, ensemble["score"])
        
        return results
    
    async def _run_model(self, model, spec: ModelSpec, image: Image.Image) -> float:
        """Run a single model and normalize output to AI probability"""
        predictions = model(image)
        
        ai_score = 0.5  # default
        
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            
            # Check label mapping
            if spec.label_mapping:
                for key, value in spec.label_mapping.items():
                    if key.lower() in label:
                        if value == "ai":
                            ai_score = score
                        else:
                            ai_score = 1.0 - score
                        break
            else:
                # Generic mapping
                if any(x in label for x in ['artificial', 'ai', 'fake', 'generated', 'synthetic']):
                    ai_score = score
                elif any(x in label for x in ['human', 'real', 'natural', 'photo']):
                    ai_score = 1.0 - score
        
        return ai_score
    
    async def _run_clip_analysis(self, image: Image.Image) -> Optional[float]:
        """Run CLIP semantic analysis"""
        if not self.clip_model:
            return None
            
        try:
            import clip
            
            # Semantic labels
            ai_labels = [
                "a 3D rendered image", "a CGI character", "an AI generated image",
                "a digital artwork", "a synthetic image", "a computer generated face",
                "a video game character", "a hyperrealistic render"
            ]
            real_labels = [
                "a real photograph", "a camera photo", "a natural photograph",
                "a candid photo", "a genuine photograph"
            ]
            
            all_labels = ai_labels + real_labels
            
            # Process
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
            logger.warning(f"CLIP analysis failed: {e}")
            return None
    
    def _calculate_ensemble(self, scores: List[tuple]) -> Dict:
        """Calculate weighted ensemble score"""
        if not scores:
            return {"score": 0.5, "method": "none"}
        
        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)
        
        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            "score": round(ensemble_score, 4),
            "method": "weighted_average",
            "total_weight": round(total_weight, 2),
            "model_count": len(scores)
        }
    
    def _calculate_uncertainty(self, scores: List[tuple], ensemble: float) -> Dict:
        """Calculate uncertainty metrics"""
        if len(scores) < 2:
            return {"disagreement": 0, "entropy": 0, "confidence_interval": [0.3, 0.7]}
        
        raw_scores = [s for s, _ in scores]
        
        # Disagreement: max - min
        disagreement = max(raw_scores) - min(raw_scores)
        
        # Entropy of ensemble prediction
        p = max(0.001, min(0.999, ensemble))
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        # Bootstrap confidence interval
        ci = self._bootstrap_ci(raw_scores)
        
        return {
            "disagreement": round(disagreement, 4),
            "entropy": round(entropy, 4),
            "confidence_interval": ci,
            "std": round(np.std(raw_scores), 4)
        }
    
    def _bootstrap_ci(self, scores: List[float], n_bootstrap: int = 100) -> List[float]:
        """Calculate bootstrap confidence interval"""
        if len(scores) < 2:
            return [0.3, 0.7]
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_low = np.percentile(bootstrap_means, 5)
        ci_high = np.percentile(bootstrap_means, 95)
        
        return [round(ci_low, 3), round(ci_high, 3)]