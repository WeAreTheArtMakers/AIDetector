"""
AI Image Detector Backend - Production Grade Forensic Analysis v6
==================================================================
Multi-model ensemble with calibration, uncertainty, GPS, JPEG forensics.
Backward compatible with feature flags.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any

from forensic.config import get_config, ForensicConfig
from forensic.ai_detector import ForensicAIDetector, EvidenceLevel
from forensic.metadata import MetadataExtractor
from forensic.domain import DomainDetector
from forensic.jpeg_forensics import JPEGForensics
from forensic.statistics import StatisticalAnalyzer, ModelScore
from forensic.ai_type import AITypeClassifier
from forensic.content_type import ContentTypeClassifier, get_content_type_classifier
from forensic.verdict_generator import generate_verdict_text, VerdictKey
from forensic.manipulation import get_manipulation_detector
from forensic.edit_assessment import get_edit_assessor, EditType
from forensic.diffusion_fingerprint import get_diffusion_fingerprint
from forensic.pathway_classifier import get_pathway_classifier, PathwayType
from forensic.visualization import get_forensic_visualizer, VisualizationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Image Detector - Forensic Analysis",
    description="Production-grade AI detection with ensemble models, calibration, and uncertainty",
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config: ForensicConfig = None
detector: ForensicAIDetector = None
metadata_extractor: MetadataExtractor = None
domain_detector: DomainDetector = None
jpeg_forensics: JPEGForensics = None
stats_analyzer: StatisticalAnalyzer = None
ai_type_classifier: AITypeClassifier = None
content_type_classifier: ContentTypeClassifier = None

@app.on_event("startup")
async def startup():
    global config, detector, metadata_extractor, domain_detector
    global jpeg_forensics, stats_analyzer, ai_type_classifier, content_type_classifier
    
    logger.info("🚀 Starting Forensic AI Detector v6.0...")
    
    config = get_config()
    
    detector = ForensicAIDetector({})
    await detector.load_model()
    
    metadata_extractor = MetadataExtractor()
    domain_detector = DomainDetector()
    jpeg_forensics = JPEGForensics()
    stats_analyzer = StatisticalAnalyzer()
    ai_type_classifier = AITypeClassifier()
    
    # Initialize content type classifier with CLIP model from detector
    content_type_classifier = get_content_type_classifier()
    if detector.clip_model and config.ENABLE_CLIP_PROMPT_ENSEMBLE:
        content_type_classifier.set_clip_model(
            detector.clip_model, 
            detector.clip_preprocess, 
            detector.device
        )
        logger.info("✅ Content type classifier initialized with CLIP")
    
    logger.info("✅ All systems ready")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "6.5.0",
        "models_loaded": len(detector.models) if detector else 0,
        "features": {
            "jpeg_forensics": config.ENABLE_JPEG_FORENSICS if config else False,
            "xmp_iptc": config.ENABLE_XMP_IPTC if config else False,
            "calibration": config.ENABLE_CALIBRATION if config else False,
            "non_photo_gate": config.ENABLE_NON_PHOTO_GATE if config else False,
            "clip_prompt_ensemble": config.ENABLE_CLIP_PROMPT_ENSEMBLE if config else False,
            "two_axis_output": config.ENABLE_TWO_AXIS_OUTPUT if config else False,
            "verdict_text": config.ENABLE_VERDICT_TEXT if config else False,
            "manipulation_detection": config.ENABLE_MANIPULATION_MODULES if config else False,
            "pathway_classifier": config.ENABLE_PATHWAY_CLASSIFIER if config else False,
            "diffusion_fingerprint": config.ENABLE_DIFFUSION_FINGERPRINT if config else False
        }
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), file_chain: str = "unknown"):
    """
    Main forensic analysis endpoint
    
    Args:
        file: Image file to analyze
        file_chain: Source chain (original, whatsapp, instagram, screenshot, unknown)
    
    Returns comprehensive forensic report (backward compatible)
    """
    start_time = datetime.now()
    
    try:
        # Validate
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid file type")
        
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "File too large (max 10MB)")
        
        # Open image
        try:
            image = Image.open(io.BytesIO(content))
            original_format = image.format
            if image.mode not in ['RGB', 'RGBA', 'L']:
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(400, f"Invalid image: {e}")
        
        # === PHASE 1: Metadata Extraction ===
        logger.info("📋 Extracting metadata...")
        metadata = metadata_extractor.extract_all(content)
        
        # === PHASE 2: Domain Detection ===
        logger.info("🔍 Detecting domain...")
        domain = domain_detector.analyze(image, metadata)
        
        # Apply file chain penalty
        if file_chain in ["whatsapp", "instagram"]:
            domain["characteristics"].append("social_media")
            domain["confidence_penalty"] += config.DOMAIN_PENALTIES.get("social_media", 0.12)
            domain["warnings"].append(f"File from {file_chain} - likely recompressed")
        
        # === PHASE 3: JPEG Forensics ===
        jpeg_result = {}
        if config.ENABLE_JPEG_FORENSICS:
            logger.info("🔬 JPEG forensics...")
            jpeg_result = jpeg_forensics.analyze(content, image)
        
        # === PHASE 3.5: Content Type Classification (Photo vs Non-Photo) ===
        content_type_result = {}
        if config.ENABLE_NON_PHOTO_GATE and config.ENABLE_CLIP_PROMPT_ENSEMBLE:
            logger.info("🎭 Content type classification...")
            content_type_result = content_type_classifier.classify(image)
        
        # === PHASE 4: AI Detection ===
        logger.info("🤖 Running AI analysis...")
        ai_report = await detector.analyze(image, metadata, domain)
        
        # === PHASE 5: Statistical Analysis ===
        stats_result = {}
        if config.ENABLE_CALIBRATION or config.ENABLE_UNCERTAINTY:
            logger.info("📊 Statistical analysis...")
            model_scores = ai_report.get("models", [])
            stats_result = stats_analyzer.analyze(model_scores, domain)
        
        # === PHASE 6: AI Type Classification ===
        ai_type_result = {}
        if config.ENABLE_AI_TYPE_CLASSIFIER:
            logger.info("🎨 AI type classification...")
            ai_type_result = ai_type_classifier.classify(
                image, metadata, ai_report.get("models", [])
            )
        
        # === PHASE 6.5: Diffusion Fingerprint + Pathway Classification ===
        diffusion_result = {}
        pathway_result = {}
        
        if config.ENABLE_PATHWAY_CLASSIFIER:
            # Run diffusion fingerprint analysis
            if config.ENABLE_DIFFUSION_FINGERPRINT:
                logger.info("🔬 Diffusion fingerprint analysis...")
                diffusion_analyzer = get_diffusion_fingerprint()
                diffusion_result = diffusion_analyzer.analyze(image)
        
        # === PHASE 7: Manipulation Detection + Edit Assessment ===
        manipulation_result = {}
        localization_result = {}
        edit_assessment_result = {}
        
        if config.ENABLE_MANIPULATION_MODULES:
            logger.info("🔍 Manipulation detection...")
            manipulation_detector = get_manipulation_detector()
            deep_scan = file_chain == "original"  # Deep scan only for original files
            manipulation_result = manipulation_detector.analyze(image, deep_scan=deep_scan)
            
            # === NEW: Edit Assessment (Global vs Local vs Generator Artifacts) ===
            logger.info("📊 Edit assessment...")
            edit_assessor = get_edit_assessor()
            edit_assessment_result = edit_assessor.assess(
                image=image,
                manipulation_result=manipulation_result,
                domain=domain,
                content_type=content_type_result,
                metadata=metadata,
                pathway_result=pathway_result,  # NEW: pass pathway for AI detection
                diffusion_result=diffusion_result  # NEW: pass diffusion for generator artifacts
            )
            
            # Build localization result (only show regions if local manipulation)
            all_regions = []
            methods_used = []
            
            # Use filtered regions from edit assessment if available
            if edit_assessment_result.get("edit_type") == EditType.LOCAL_MANIPULATION:
                # Show regions only for confirmed local manipulation
                filtered_regions = edit_assessment_result.get("filtered_regions", [])
                all_regions = filtered_regions
                methods_used = edit_assessment_result.get("corroborated_methods", [])
            elif edit_assessment_result.get("edit_type") in [EditType.GLOBAL_POSTPROCESS, EditType.GENERATOR_ARTIFACTS]:
                # Don't show regions for global edits or generator artifacts (they're not splices)
                all_regions = []
                methods_used = []
            else:
                # Fallback: collect regions but mark as unconfirmed
                for method in ["splice", "copy_move", "edge_matte", "blur_noise_mismatch"]:
                    method_data = manipulation_result.get(method, {})
                    if method_data.get("regions"):
                        all_regions.extend(method_data["regions"])
                        methods_used.append(method)
            
            localization_result = {
                "enabled": True,
                "top_regions": sorted(all_regions, key=lambda r: -r.get("score", 0))[:10],
                "methods": methods_used,
                "notes": [],
                "edit_type": edit_assessment_result.get("edit_type", "none_detected"),
                "show_regions": edit_assessment_result.get("edit_type") == EditType.LOCAL_MANIPULATION,
                "boundary_corroborated": edit_assessment_result.get("boundary_corroborated", False)
            }
            
            # Add appropriate notes based on edit type
            edit_type = edit_assessment_result.get("edit_type", "none_detected")
            if edit_type == EditType.LOCAL_MANIPULATION:
                localization_result["notes"].append("Local manipulation evidence detected (boundary-corroborated)")
            elif edit_type == EditType.GLOBAL_POSTPROCESS:
                localization_result["notes"].append("Global post-processing detected (filter/color grading)")
            elif edit_type == EditType.GENERATOR_ARTIFACTS:
                localization_result["notes"].append("AI generation artifacts detected (not manipulation)")
            elif manipulation_result.get("overall_score", 0) > 0.5:
                localization_result["notes"].append("Artifacts detected but not confirmed as manipulation")
        
        # === PHASE 7.3: Visualization Generation ===
        visualization_result = {}
        if config.ENABLE_MANIPULATION_MODULES and edit_assessment_result:
            logger.info("🎨 Generating visualization...")
            visualizer = get_forensic_visualizer()
            visualization_result = visualizer.generate_visualization(
                image=image,
                edit_assessment=edit_assessment_result,
                manipulation_result=manipulation_result
            )
            logger.info(f"🎨 Visualization mode: {visualization_result.get('mode', 'none')}")
        
        # === PHASE 7.5: Pathway Classification (after manipulation detection) ===
        if config.ENABLE_PATHWAY_CLASSIFIER:
            logger.info("🛤️ Pathway classification (T2I vs I2I vs Real)...")
            pathway_classifier = get_pathway_classifier()
            pathway_result = pathway_classifier.classify(
                image=image,
                diffusion_result=diffusion_result,
                content_type=content_type_result,
                manipulation_result=manipulation_result,
                edit_assessment=edit_assessment_result,
                metadata=metadata,
                domain=domain,
                model_scores=ai_report.get("models", [])
            )
            logger.info(f"🛤️ Pathway: {pathway_result.get('pred')} ({pathway_result.get('confidence')})")
        
        # === BUILD RESPONSE (Backward Compatible) ===
        processing_time = (datetime.now() - start_time).total_seconds()
        
        evidence_level = ai_report.get("evidence_level", EvidenceLevel.INCONCLUSIVE)
        summary_axes = ai_report.get("summary_axes")
        
        # Check for inconclusive from statistics (but respect two-axis logic)
        # Only force INCONCLUSIVE if probability is truly in uncertain range
        if stats_result.get("inconclusive_reason"):
            ai_likelihood = summary_axes.get("ai_likelihood", {}) if summary_axes else {}
            ai_prob = ai_likelihood.get("probability", 0.5)
            ci = ai_likelihood.get("confidence_interval", [0.3, 0.7])
            
            # Only override to INCONCLUSIVE if probability is in uncertain range (0.35-0.65)
            # AND CI doesn't strongly support a direction
            if 0.35 <= ai_prob <= 0.65 and ci[0] < 0.55 and ci[1] > 0.45:
                if evidence_level in [EvidenceLevel.STATISTICAL_AI, EvidenceLevel.STATISTICAL_REAL]:
                    evidence_level = EvidenceLevel.INCONCLUSIVE
                    ai_report["recommendation"] = (
                        f"⚠️ BELİRSİZ: {stats_result['inconclusive_reason']}. "
                        "Model uyuşmazlığı yüksek ve/veya olasılık belirsiz bölgede."
                    )
        
        verdict_map = {
            EvidenceLevel.DEFINITIVE_AI: {
                "text": "KESİN: AI Tarafından Üretilmiş",
                "text_en": "DEFINITIVE: AI Generated",
                "color": "red",
                "icon": "alert-octagon",
                "is_definitive": True
            },
            EvidenceLevel.DEFINITIVE_REAL: {
                "text": "KESİN: Doğrulanmış Gerçek Fotoğraf",
                "text_en": "DEFINITIVE: Verified Real Photo",
                "color": "green",
                "icon": "shield-check",
                "is_definitive": True
            },
            EvidenceLevel.STRONG_AI_EVIDENCE: {
                "text": "GÜÇLÜ KANIT: AI Üretimi Muhtemel",
                "text_en": "STRONG EVIDENCE: Likely AI Generated",
                "color": "orange",
                "icon": "alert-triangle",
                "is_definitive": False
            },
            EvidenceLevel.STRONG_REAL_EVIDENCE: {
                "text": "GÜÇLÜ KANIT: Gerçek Fotoğraf",
                "text_en": "STRONG EVIDENCE: Likely Real Photo",
                "color": "green",
                "icon": "camera",
                "is_definitive": False
            },
            EvidenceLevel.STATISTICAL_AI: {
                "text": "İSTATİSTİKSEL: AI Olabilir (Belirsiz)",
                "text_en": "STATISTICAL: Possibly AI (Uncertain)",
                "color": "yellow",
                "icon": "bar-chart-2",
                "is_definitive": False
            },
            EvidenceLevel.STATISTICAL_REAL: {
                "text": "İSTATİSTİKSEL: Gerçek Olabilir",
                "text_en": "STATISTICAL: Possibly Real",
                "color": "blue",
                "icon": "image",
                "is_definitive": False
            },
            EvidenceLevel.INCONCLUSIVE: {
                "text": "BELİRSİZ: Yeterli Kanıt Yok",
                "text_en": "INCONCLUSIVE: Insufficient Evidence",
                "color": "gray",
                "icon": "help-circle",
                "is_definitive": False
            }
        }
        
        verdict = verdict_map.get(evidence_level, verdict_map[EvidenceLevel.INCONCLUSIVE])
        
        # === NON-PHOTO GATE: Override verdict for CGI/Illustration ===
        ai_probability_informational = False
        non_photo_override = False
        
        if (config.ENABLE_NON_PHOTO_GATE and 
            content_type_result.get("type") == "non_photo_like" and 
            content_type_result.get("confidence") == "high"):
            
            non_photo_override = True
            ai_probability_informational = True
            
            verdict = {
                "text": "NON-PHOTO: CGI/İllüstrasyon (Yüksek Güven)",
                "text_en": "NON-PHOTO: CGI/Illustration (High Confidence)",
                "color": "purple",
                "icon": "palette",
                "is_definitive": False,
                "category": "non_photo"
            }
            evidence_level = "CONTENT_TYPE_NON_PHOTO"
            
            ai_report["recommendation"] = (
                "🎨 Bu görsel CGI/İllüstrasyon olarak tespit edildi. "
                "AI-vs-gerçek fotoğraf olasılığı bu içerik türü için birincil karar değildir. "
                "AI olasılığı sadece bilgi amaçlıdır."
            )
            
            logger.info(f"🎭 Non-photo gate triggered: CGI/Illustration detected")
        
        # === GENERATE VERDICT TEXT (New UX Layer) ===
        verdict_text_data = None
        if config.ENABLE_TWO_AXIS_OUTPUT and config.ENABLE_VERDICT_TEXT:
            # Build report dict for verdict generator
            verdict_report = {
                "ai_probability": ai_report.get("ai_probability", 50),
                "confidence": ai_report.get("confidence", 30),
                "uncertainty": ai_report.get("uncertainty", {}),
                "models": ai_report.get("models", []),
                "content_type": content_type_result,
                "domain": domain,
                "gps": metadata.get("gps", {}),
                "jpeg_forensics": jpeg_result,
                "summary_axes": summary_axes,
                "manipulation": manipulation_result,
                "edit_assessment": edit_assessment_result,
                "pathway": pathway_result,  # NEW: Add pathway for T2I/photoreal detection
                "diffusion_fingerprint": diffusion_result  # NEW: Add diffusion for fingerprint score
            }
            verdict_text_data = generate_verdict_text(verdict_report)
            
            # Update summary_axes with the computed values from verdict generator
            if verdict_text_data.get("summary_axes"):
                summary_axes = verdict_text_data["summary_axes"]
            
            logger.info(f"📝 Verdict: {verdict_text_data.get('verdict_key')} - {verdict_text_data.get('title_en')}")
        
        # === RESPONSE STRUCTURE (Backward Compatible + New Fields) ===
        response = {
            # === EXISTING FIELDS (DO NOT REMOVE) ===
            "success": True,
            "report_id": hashlib.md5(content[:1000]).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(processing_time, 2),
            
            "evidence_level": evidence_level,
            "verdict": verdict,
            "ai_probability": ai_report.get("ai_probability", 50),
            "ai_probability_informational": ai_probability_informational,
            "confidence": ai_report.get("confidence", 30),
            
            "models": ai_report.get("models", []),
            "ensemble": ai_report.get("ensemble", {}),
            "uncertainty": ai_report.get("uncertainty", {}),
            
            "definitive_findings": ai_report.get("definitive_findings", []),
            "strong_evidence": ai_report.get("strong_evidence", []),
            "statistical_indicators": ai_report.get("statistical_indicators", []),
            "authenticity_indicators": ai_report.get("authenticity_indicators", []),
            
            "ai_detection": metadata.get("ai_detection", {}),
            "camera": metadata.get("camera", {}),
            "gps": metadata.get("gps", {}),
            
            "metadata": {
                "exif_present": bool(metadata.get("exif")),
                "exif_count": len(metadata.get("exif", {})),
                "completeness_score": metadata.get("completeness_score", 0),
                "timestamps": metadata.get("timestamps", {}),
                "technical": metadata.get("technical", {}),
                "software": metadata.get("software", {})
            },
            
            "domain": {
                "type": domain.get("domain_type", "unknown"),
                "characteristics": domain.get("characteristics", []),
                "warnings": domain.get("warnings", [])
            },
            
            "image_info": {
                "width": image.size[0],
                "height": image.size[1],
                "format": original_format,
                "mode": image.mode,
                "file_size_kb": round(len(content) / 1024, 1)
            },
            
            "recommendation": ai_report.get("recommendation", ""),
            
            "warning": (
                "🎨 Bu görsel CGI/İllüstrasyon gibi görünüyor. AI olasılığı sadece bilgi amaçlıdır."
                if non_photo_override else
                (None if verdict.get("is_definitive") else 
                 None)  # Warnings now handled by verdict_text banner
            ),
            
            # === NEW FIELDS (Progressive Enhancement) ===
            "content_type": content_type_result if config.ENABLE_NON_PHOTO_GATE else None,
            
            # Two-axis output: AI likelihood vs Evidential quality
            "summary_axes": summary_axes if config.ENABLE_TWO_AXIS_OUTPUT else None,
            
            # Verdict text layer (title, subtitle, banner, footer, tags)
            "verdict_text": verdict_text_data if (config.ENABLE_TWO_AXIS_OUTPUT and config.ENABLE_VERDICT_TEXT) else None,
            
            "metadata_extended": {
                "exif": metadata.get("exif", {}),
                "xmp": metadata.get("xmp", {}) if config.ENABLE_XMP_IPTC else None,
                "iptc": metadata.get("iptc", {}) if config.ENABLE_XMP_IPTC else None,
                "software_history": metadata.get("software_history", []),
                "format_info": metadata.get("format_info", {})
            },
            
            "jpeg_forensics": jpeg_result if config.ENABLE_JPEG_FORENSICS else None,
            
            "statistics": stats_result if (config.ENABLE_CALIBRATION or config.ENABLE_UNCERTAINTY) else None,
            
            "ai_generation_type": ai_type_result if config.ENABLE_AI_TYPE_CLASSIFIER else None,
            
            # Generation pathway classification (T2I vs I2I vs Real)
            "pathway": pathway_result if config.ENABLE_PATHWAY_CLASSIFIER else None,
            "diffusion_fingerprint": diffusion_result if config.ENABLE_DIFFUSION_FINGERPRINT else None,
            
            # Manipulation detection and localization
            "manipulation": manipulation_result if config.ENABLE_MANIPULATION_MODULES else None,
            "localization": localization_result if config.ENABLE_MANIPULATION_MODULES else None,
            "edit_assessment": edit_assessment_result if config.ENABLE_MANIPULATION_MODULES else None,
            "visualization": visualization_result if config.ENABLE_MANIPULATION_MODULES else None,
            
            "file_chain": file_chain,
            
            "feature_flags": {
                "jpeg_forensics": config.ENABLE_JPEG_FORENSICS,
                "xmp_iptc": config.ENABLE_XMP_IPTC,
                "calibration": config.ENABLE_CALIBRATION,
                "ai_type": config.ENABLE_AI_TYPE_CLASSIFIER,
                "non_photo_gate": config.ENABLE_NON_PHOTO_GATE,
                "clip_prompt_ensemble": config.ENABLE_CLIP_PROMPT_ENSEMBLE,
                "two_axis_output": config.ENABLE_TWO_AXIS_OUTPUT,
                "verdict_text": config.ENABLE_VERDICT_TEXT,
                "manipulation_detection": config.ENABLE_MANIPULATION_MODULES,
                "pathway_classifier": config.ENABLE_PATHWAY_CLASSIFIER,
                "diffusion_fingerprint": config.ENABLE_DIFFUSION_FINGERPRINT
            }
        }
        
        # Remove None values for cleaner response
        response = {k: v for k, v in response.items() if v is not None}
        
        logger.info(f"✅ Analysis complete: {evidence_level} - AI: {ai_report.get('ai_probability')}%")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)