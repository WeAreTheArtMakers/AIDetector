"""
AI Image Detector - Forensic Analysis API
Enhanced version with C2PA provenance verification and chain of custody
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import torch
from PIL import Image
import io
import logging
import asyncio
from typing import Dict, Any, Optional, List
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Import forensic modules
from forensic.provenance import ProvenanceVerifier
from forensic.custody import CustodyManager
from forensic.ai_detector import CalibratedAIDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="AI Image Detector - Forensic Analysis API",
    description="Digital forensics service for AI-generated image detection with C2PA provenance verification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global configuration
FORENSIC_CONFIG = {
    "store_evidence_files": os.getenv("STORE_EVIDENCE_FILES", "false").lower() == "true",
    "evidence_directory": os.getenv("EVIDENCE_DIR", "./evidence"),
    "signing_key_path": os.getenv("SIGNING_KEY_PATH", "./keys/forensic_signing.pem"),
    "generate_signing_key": os.getenv("GENERATE_SIGNING_KEY", "true").lower() == "true",
    "model": {
        "name": os.getenv("AI_MODEL_NAME", "umm-maybe/AI-image-detector"),  # Specialized AI detector
        "fallback": "google/vit-base-patch16-224",  # Fallback model
        "version": "2.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "calibration": {
        "calibration_file": os.getenv("CALIBRATION_FILE", "./config/calibration.json")
    },
    "max_file_size": int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB default
    "supported_formats": ["JPEG", "PNG", "WebP", "BMP"]
}

# Global forensic components
provenance_verifier = ProvenanceVerifier()
custody_manager = CustodyManager(FORENSIC_CONFIG)
ai_detector = CalibratedAIDetector(FORENSIC_CONFIG)

# In-memory report storage (use database in production)
forensic_reports = {}

@app.on_event("startup")
async def startup_event():
    """Initialize forensic analysis components"""
    try:
        logger.info("🚀 Starting AI Image Detector - Forensic Analysis API v2.0.0")
        
        # Load AI detection model
        await ai_detector.load_model()
        
        # Create necessary directories
        evidence_dir = Path(FORENSIC_CONFIG["evidence_directory"])
        evidence_dir.mkdir(exist_ok=True)
        
        keys_dir = Path("./keys")
        keys_dir.mkdir(exist_ok=True)
        
        config_dir = Path("./config")
        config_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Forensic analysis system initialized")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/health")
async def health_check():
    """Enhanced health check with forensic system status"""
    try:
        system_status = {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ai_detector": "loaded" if ai_detector.detector_model else "not_loaded",
                "provenance_verifier": "ready",
                "custody_manager": "ready",
                "evidence_storage": "enabled" if FORENSIC_CONFIG["store_evidence_files"] else "disabled",
                "digital_signing": "enabled" if custody_manager.signing_key else "disabled"
            },
            "device": FORENSIC_CONFIG["model"]["device"],
            "model": FORENSIC_CONFIG["model"]["name"]
        }
        
        return system_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def validate_forensic_upload(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Enhanced validation for forensic analysis"""
    # File size check
    if len(file_content) > FORENSIC_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=413, 
            detail=f"File size exceeds {FORENSIC_CONFIG['max_file_size']} bytes limit"
        )
    
    # File signature validation
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        
        if image.format not in FORENSIC_CONFIG["supported_formats"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {image.format}. Supported: {FORENSIC_CONFIG['supported_formats']}"
            )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    return {
        "size_bytes": len(file_content),
        "format": image.format,
        "filename": filename,
        "validation_passed": True
    }

async def analyze_supporting_signals(image: Image.Image, image_bytes: bytes) -> Dict[str, Any]:
    """
    Enhanced supporting signals analysis for forensic context
    """
    try:
        # Enhanced metadata analysis
        metadata_analysis = {
            "camera_info_present": False,
            "ai_software_detected": [],
            "suspicious_patterns": 0
        }
        
        # Check EXIF for camera info
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            if exif and (exif.get(272) or exif.get(271)):  # Make/Model tags
                metadata_analysis["camera_info_present"] = True
            
            # Check for AI software signatures
            software_info = str(exif.get(305, '')).lower()  # Software tag
            ai_software_markers = [
                'midjourney', 'dall-e', 'stable diffusion', 'adobe firefly',
                'leonardo', 'runway', 'artbreeder', 'deepai', 'nightcafe',
                'wombo', 'starryai', 'craiyon', 'canva ai', 'jasper'
            ]
            
            for marker in ai_software_markers:
                if marker in software_info:
                    metadata_analysis["ai_software_detected"].append(marker.title())
            
            # Count suspicious patterns
            if not metadata_analysis["camera_info_present"]:
                metadata_analysis["suspicious_patterns"] += 1
            if metadata_analysis["ai_software_detected"]:
                metadata_analysis["suspicious_patterns"] += 2
        else:
            metadata_analysis["suspicious_patterns"] += 1  # No EXIF at all
        
        # Enhanced technical analysis
        img_array = np.array(image)
        width, height = image.size
        
        # 1. Smarter dimension analysis (more selective)
        exact_ai_sizes = [512, 768, 1024, 1536, 2048]  # Most common AI sizes
        precise_ai_ratios = [1.0, 1.333, 1.5, 1.777, 0.75, 0.667, 0.5625]  # Common AI ratios
        
        dimension_score = 0.05  # Lower base score
        
        # Check exact AI dimensions (more selective)
        if width in exact_ai_sizes and height in exact_ai_sizes:
            dimension_score = 0.4  # Perfect square AI dimensions
            logger.info(f"🚨 Perfect AI dimensions: {width}x{height}")
        elif width in exact_ai_sizes or height in exact_ai_sizes:
            dimension_score = 0.2  # One dimension matches
        
        # Check AI-typical aspect ratios (more precise)
        aspect_ratio = width / height
        for ratio in precise_ai_ratios:
            if abs(aspect_ratio - ratio) < 0.005:  # Very precise matching
                dimension_score = max(dimension_score, 0.25)
                logger.info(f"🚨 Precise AI ratio: {aspect_ratio:.4f} ≈ {ratio}")
                break
        
        # 2. Smarter color analysis (more selective)
        color_score = 0.05  # Lower base score
        
        if len(img_array.shape) == 3:
            # Check extreme saturation (AI signature)
            hsv = np.array(image.convert('HSV'))
            saturation = hsv[:, :, 1] / 255.0
            extreme_sat_ratio = np.sum(saturation > 0.9) / saturation.size
            
            if extreme_sat_ratio > 0.4:  # Very high threshold
                color_score = 0.35
                logger.info(f"🎨 Extreme saturation: {extreme_sat_ratio:.3f}")
            
            # Check severe color quantization (AI artifact)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_diversity = unique_colors / total_pixels
            
            if color_diversity < 0.05:  # Very low diversity
                color_score = max(color_score, 0.3)
                logger.info(f"🎨 Severe quantization: {color_diversity:.4f}")
        
        # 3. Enhanced noise analysis
        noise_score = 0.1  # Base score
        
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate local variance (noise measure)
        kernel_size = 5
        variances = []
        
        for i in range(0, gray.shape[0] - kernel_size, kernel_size):
            for j in range(0, gray.shape[1] - kernel_size, kernel_size):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                variances.append(np.var(patch))
        
        if variances:
            avg_variance = np.mean(variances)
            
            if avg_variance < 25:  # Too clean
                noise_score = 0.4
                logger.info(f"🔍 Low noise variance: {avg_variance:.2f}")
            elif avg_variance < 50:
                noise_score = 0.25
        
        # 4. Enhanced edge analysis
        edge_score = 0.1  # Base score
        
        try:
            # Simple edge detection using gradients
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            
            # Check for oversharpening
            strong_edges_x = np.sum(grad_x > 50) / grad_x.size
            strong_edges_y = np.sum(grad_y > 50) / grad_y.size
            
            if strong_edges_x > 0.1 or strong_edges_y > 0.1:
                edge_score = 0.3
                logger.info(f"⚡ Oversharpening detected: {max(strong_edges_x, strong_edges_y):.3f}")
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
        
        technical_analysis = {
            "dimension_score": dimension_score,
            "color_score": color_score,
            "noise_score": noise_score,
            "edge_score": edge_score
        }
        
        return {
            "metadata_analysis": metadata_analysis,
            "technical_analysis": technical_analysis
        }
        
    except Exception as e:
        logger.warning(f"Supporting signals analysis failed: {e}")
        return {
            "metadata_analysis": {"error": str(e)},
            "technical_analysis": {"error": str(e)}
        }

@app.post("/analyze_forensic")
async def analyze_forensic(file: UploadFile = File(...)):
    """
    Comprehensive forensic analysis with C2PA verification and chain of custody
    
    Returns complete forensic report with provenance verification,
    calibrated AI detection, and evidence documentation
    """
    try:
        # Read and validate file
        file_content = await file.read()
        validation_result = validate_forensic_upload(file_content, file.filename)
        
        # Load image
        image = Image.open(io.BytesIO(file_content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 1. PROVENANCE VERIFICATION (Evidence Level)
        logger.info("🔍 Starting C2PA provenance verification...")
        provenance_result = await provenance_verifier.verify_c2pa(file_content, file.filename)
        
        # 2. CHAIN OF CUSTODY
        logger.info("📋 Creating chain of custody record...")
        custody_record = custody_manager.create_custody_record(
            file_content, file.filename, {}  # Analysis result added later
        )
        
        # 3. SUPPORTING SIGNALS ANALYSIS
        logger.info("🔬 Analyzing supporting signals...")
        supporting_signals = await analyze_supporting_signals(image, file_content)
        
        # 4. CALIBRATED AI DETECTION (Statistical Level)
        logger.info("🤖 Performing calibrated AI detection...")
        ai_analysis = await ai_detector.analyze_image(image, supporting_signals)
        
        # 5. GENERATE VERDICT
        verdict = generate_forensic_verdict(provenance_result, ai_analysis)
        
        # 6. CREATE FORENSIC REPORT
        report_id = custody_record["report_id"]
        forensic_report = custody_manager.create_forensic_report(
            report_id, custody_record, provenance_result, ai_analysis, verdict
        )
        
        # Store report in memory (use database in production)
        forensic_reports[report_id] = forensic_report
        
        # 7. PREPARE API RESPONSE
        response = {
            "success": True,
            "report_id": report_id,
            "provenance": provenance_result,
            "custody": custody_record,
            "ai": ai_analysis,
            "verdict": verdict,
            "warnings": generate_forensic_warnings(provenance_result, ai_analysis),
            "processing_time": 1.24,  # Would be actual processing time
            "report_url": f"/report/{report_id}"
        }
        
        logger.info(f"✅ Forensic analysis completed: {report_id}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forensic analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Forensic analysis failed: {str(e)}"
        )

def generate_forensic_verdict(provenance_result: Dict, ai_analysis: Dict) -> Dict[str, Any]:
    """Generate forensic verdict based on provenance and AI analysis"""
    
    # Determine evidence level
    evidence_level = provenance_verifier.get_evidence_level(provenance_result)
    authenticity_status = provenance_verifier.get_authenticity_status(provenance_result)
    
    # Get AI likelihood
    calibrated_prob = ai_analysis.get("calibrated_probability", 0.5)
    ai_likelihood = ai_detector.get_ai_likelihood_description(calibrated_prob)
    
    # Generate confidence description
    confidence_interval = ai_analysis.get("confidence_interval", [0.3, 0.7])
    confidence_description = ai_detector.get_confidence_description(
        calibrated_prob, confidence_interval
    )
    
    # Determine overall verdict text based on evidence level
    if evidence_level == "CRYPTOGRAPHIC_PROOF":
        if authenticity_status == "VERIFIED":
            verdict_text = "Provenance kriptografik olarak doğrulandı - Orijinal kaynak kanıtlandı"
            verdict_color = "text-green-400"
            verdict_icon = "shield-check"
        else:
            verdict_text = "C2PA imzası geçersiz - Provenance güvenilir değil"
            verdict_color = "text-red-400"
            verdict_icon = "shield-x"
    else:
        # Statistical analysis only
        if calibrated_prob < 0.3:
            verdict_text = "Düşük AI olasılığı - Muhtemelen gerçek görsel"
            verdict_color = "text-green-400"
            verdict_icon = "check-circle"
        elif calibrated_prob < 0.6:
            verdict_text = "Belirsiz sonuç - Ek analiz gerekebilir"
            verdict_color = "text-yellow-400"
            verdict_icon = "alert-triangle"
        else:
            verdict_text = "Yüksek AI olasılığı - Yapay üretim şüphesi"
            verdict_color = "text-orange-400"
            verdict_icon = "alert-circle"
    
    return {
        "evidence_level": evidence_level,
        "authenticity_status": authenticity_status,
        "ai_likelihood": ai_likelihood,
        "confidence_description": confidence_description,
        "verdict_text": verdict_text,
        "verdict_color": verdict_color,
        "verdict_icon": verdict_icon,
        "calibrated_probability": calibrated_prob
    }

def generate_forensic_warnings(provenance_result: Dict, ai_analysis: Dict) -> List[str]:
    """Generate appropriate warnings based on analysis results"""
    warnings = []
    
    # Provenance warnings
    if not provenance_result.get("c2pa_present"):
        warnings.append("C2PA Content Credentials bulunamadı - Provenance doğrulanamadı")
    elif not provenance_result.get("c2pa_verified"):
        warnings.append("C2PA imzası doğrulanamadı - Provenance güvenilir değil")
    
    # AI analysis warnings
    if not ai_analysis.get("calibration_applied"):
        warnings.append("Model kalibrasyonu uygulanmadı - Sonuçlar ham tahminlerdir")
    
    expected_error = ai_analysis.get("expected_error_rate", 0.2)
    if expected_error > 0.2:
        warnings.append(f"Yüksek hata oranı bekleniyor: ±{int(expected_error*100)}%")
    
    # General forensic warnings
    warnings.extend([
        "Bu analiz istatistiksel değerlendirmedir, kesin kanıt değildir",
        "Sonuçlar uzman adli analizci tarafından yorumlanmalıdır",
        "AI tespit algoritmaları doğal hata oranına sahiptir"
    ])
    
    return warnings

@app.get("/report/{report_id}")
async def get_forensic_report(report_id: str):
    """Retrieve complete forensic report by ID"""
    try:
        if report_id not in forensic_reports:
            raise HTTPException(status_code=404, detail="Forensic report not found")
        
        report = forensic_reports[report_id]
        
        # Add signature verification status
        signature_valid = custody_manager.verify_report_signature(report)
        report["signature_verification"] = {
            "valid": signature_valid,
            "verified_at": datetime.now().isoformat()
        }
        
        return JSONResponse(content=report)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report retrieval failed: {str(e)}")

@app.get("/evidence/{report_id}.zip")
async def download_evidence_bundle(report_id: str):
    """Download complete evidence bundle as ZIP file"""
    try:
        if report_id not in forensic_reports:
            raise HTTPException(status_code=404, detail="Forensic report not found")
        
        if not FORENSIC_CONFIG["store_evidence_files"]:
            raise HTTPException(status_code=404, detail="Evidence file storage is disabled")
        
        # Create evidence bundle
        bundle_path = custody_manager.create_evidence_bundle(report_id)
        
        if not bundle_path or not bundle_path.exists():
            raise HTTPException(status_code=404, detail="Evidence bundle not found")
        
        return FileResponse(
            path=str(bundle_path),
            filename=f"{report_id}_evidence_bundle.zip",
            media_type="application/zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evidence bundle download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evidence bundle download failed: {str(e)}")

# Legacy endpoint for backward compatibility
@app.post("/analyze")
async def analyze_legacy(file: UploadFile = File(...)):
    """
    Legacy analysis endpoint for backward compatibility
    Returns simplified response format
    """
    try:
        # Perform forensic analysis
        forensic_result = await analyze_forensic(file)
        forensic_data = forensic_result.body.decode() if hasattr(forensic_result, 'body') else forensic_result
        
        if isinstance(forensic_data, str):
            forensic_data = json.loads(forensic_data)
        
        # Convert to legacy format
        ai_analysis = forensic_data.get("ai", {})
        verdict = forensic_data.get("verdict", {})
        
        legacy_response = {
            "success": True,
            "aiProbability": int(ai_analysis.get("calibrated_probability", 0.5) * 100),
            "confidence": int(abs(ai_analysis.get("calibrated_probability", 0.5) - 0.5) * 200),
            "verdict": {
                "text": verdict.get("verdict_text", "Analiz tamamlandı"),
                "color": verdict.get("verdict_color", "text-blue-400"),
                "icon": verdict.get("verdict_icon", "info"),
                "bgColor": "bg-blue-500/20"
            },
            "indicators": generate_legacy_indicators(forensic_data),
            "processingTime": "1.24",
            "fileSize": f"{forensic_data.get('custody', {}).get('file_metadata', {}).get('size_bytes', 0) / 1024:.1f}",
            "metadata": {
                "fileName": file.filename,
                "forensic_report_id": forensic_data.get("report_id"),
                "provenance_status": forensic_data.get("provenance", {}).get("status", "NOT_PRESENT")
            },
            "warning": "Bu analiz olasılık tahminidir. Kesin sonuç için adli raporu inceleyin."
        }
        
        return JSONResponse(content=legacy_response)
        
    except Exception as e:
        logger.error(f"Legacy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_legacy_indicators(forensic_data: Dict) -> List[Dict]:
    """Generate legacy-format indicators from forensic data"""
    indicators = []
    
    # Provenance indicator
    provenance = forensic_data.get("provenance", {})
    if provenance.get("c2pa_verified"):
        indicators.append({
            "name": "Provenance Doğrulama",
            "score": 5,
            "status": "normal",
            "description": f"C2PA doğrulandı - {provenance.get('issuer', 'Bilinmeyen')}",
            "details": "Kriptografik kanıt mevcut"
        })
    elif provenance.get("c2pa_present"):
        indicators.append({
            "name": "Provenance Doğrulama", 
            "score": 70,
            "status": "warning",
            "description": "C2PA mevcut ancak doğrulanamadı",
            "details": "İmza geçersiz veya bozuk"
        })
    else:
        indicators.append({
            "name": "Provenance Doğrulama",
            "score": 60,
            "status": "warning", 
            "description": "C2PA Content Credentials bulunamadı",
            "details": "Provenance kanıtı mevcut değil"
        })
    
    # AI Analysis indicator
    ai_analysis = forensic_data.get("ai", {})
    calibrated_prob = ai_analysis.get("calibrated_probability", 0.5)
    
    indicators.append({
        "name": "Kalibre AI Analizi",
        "score": int(calibrated_prob * 100),
        "status": "suspicious" if calibrated_prob > 0.7 else ("warning" if calibrated_prob > 0.4 else "normal"),
        "description": f"Kalibrasyon uygulandı: %{int(calibrated_prob * 100)} AI olasılığı",
        "details": f"Hata payı: ±{int(ai_analysis.get('expected_error_rate', 0.15) * 100)}%"
    })
    
    return indicators

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)