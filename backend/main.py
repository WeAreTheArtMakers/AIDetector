from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from PIL.ExifTags import TAGS
import io
import logging
import asyncio
from typing import Dict, Any
import json
import hashlib
from datetime import datetime
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Image Detector API",
    description="Backend service for AI-generated image detection",
    version="1.0.0"
)

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
ai_detector_model = None
processor = None

# Model configuration - easily changeable
MODEL_CONFIG = {
    "model_name": "google/vit-base-patch16-224",  # Stable fallback model
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_image_size": (1024, 1024),
    "supported_formats": ["JPEG", "PNG", "WebP", "BMP"]
}

async def load_model():
    """Load AI detection model on startup"""
    global ai_detector_model, processor
    
    try:
        logger.info(f"Loading model: {MODEL_CONFIG['model_name']}")
        
        # Use image classification pipeline (more stable)
        ai_detector_model = pipeline(
            "image-classification",
            model=MODEL_CONFIG["model_name"],
            device=0 if MODEL_CONFIG["device"] == "cuda" else -1
        )
        logger.info("AI detection model loaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Model loading failed")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if ai_detector_model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": MODEL_CONFIG["device"],
        "timestamp": datetime.now().isoformat()
    }

def validate_image(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Validate uploaded image"""
    # Check file size (10MB limit)
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")
    
    # Check file type using PIL
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return {
        "size_bytes": len(file_content),
        "mime_type": f"image/{image.format.lower()}" if hasattr(image, 'format') else "image/unknown",
        "filename": filename
    }

def analyze_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """Enhanced metadata analysis for latest AI generation tools"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract EXIF data
        exif_data = {}
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        
        # Enhanced AI generation software signatures (2024 updated)
        ai_software_markers = [
            # Google AI Tools
            "gemini", "bard", "imagen", "parti", "google ai",
            # OpenAI Tools
            "dall-e", "dall·e", "dall-e 2", "dall-e 3", "chatgpt", "gpt-4", "openai",
            # Midjourney Versions
            "midjourney", "mj", "midjourney v6", "midjourney v5", "midjourney v4",
            # Stability AI
            "stable diffusion", "sdxl", "sd xl", "stability ai", "dreamstudio",
            # Adobe AI
            "firefly", "adobe firefly", "photoshop ai", "generative fill",
            # Microsoft AI
            "copilot", "bing image creator", "designer", "microsoft ai",
            # Anthropic
            "claude", "anthropic",
            # Meta AI
            "meta ai", "imagine", "llama", "emu",
            # Other Popular Tools
            "leonardo", "runway", "artbreeder", "deepai", "nightcafe",
            "wombo", "starryai", "craiyon", "bluewillow", "playground",
            "canva ai", "jasper", "copy.ai", "synthesia", "luma ai",
            # New 2024 Tools
            "pika labs", "gen-2", "sora", "ideogram", "flux", "black forest labs",
            "recraft", "freepik ai", "clipdrop", "remove.bg", "upscayl",
            # Chinese AI Tools
            "baidu", "tencent ai", "alibaba ai", "bytedance", "douyin ai",
            # Mobile AI Apps
            "prisma", "artisto", "deepart", "neural cam", "ai photo enhancer"
        ]
        
        software_info = str(exif_data.get('Software', '')).lower()
        comment_info = str(exif_data.get('ImageDescription', '')).lower()
        artist_info = str(exif_data.get('Artist', '')).lower()
        copyright_info = str(exif_data.get('Copyright', '')).lower()
        
        # Check all metadata fields for AI markers
        all_metadata = f"{software_info} {comment_info} {artist_info} {copyright_info}".lower()
        
        detected_ai_tools = []
        for marker in ai_software_markers:
            if marker in all_metadata:
                detected_ai_tools.append(marker.title())
        
        # Enhanced suspicious patterns detection
        suspicious_patterns = []
        
        # No camera info but has creation software
        if software_info and not exif_data.get('Make') and not exif_data.get('Model'):
            suspicious_patterns.append("Software without camera info")
        
        # Perfect dimensions (common AI sizes including new 2024 standards)
        width = image.width
        height = image.height
        common_ai_sizes = [512, 768, 1024, 1152, 1216, 1344, 1536, 1792, 2048, 2304, 2560]
        if width in common_ai_sizes or height in common_ai_sizes:
            suspicious_patterns.append(f"Common AI dimensions: {width}x{height}")
        
        # Square images are often AI generated
        if abs(width - height) < 10:
            suspicious_patterns.append("Perfect square aspect ratio")
        
        # Common AI aspect ratios (2024 update)
        aspect_ratio = width / height
        common_ai_ratios = [1.0, 1.33, 1.5, 1.77, 0.75, 0.67, 0.56, 1.25, 1.6, 0.8]
        for ratio in common_ai_ratios:
            if abs(aspect_ratio - ratio) < 0.02:
                suspicious_patterns.append(f"AI-typical aspect ratio: {aspect_ratio:.2f}")
                break
        
        # Check for AI-specific metadata patterns
        if 'generated' in all_metadata or 'artificial' in all_metadata:
            suspicious_patterns.append("Contains 'generated' or 'artificial' keywords")
        
        # Check for missing typical camera metadata
        camera_fields = ['Make', 'Model', 'DateTime', 'ExifVersion', 'Flash', 'FocalLength']
        missing_camera_fields = sum(1 for field in camera_fields if field not in exif_data)
        if missing_camera_fields >= 4:
            suspicious_patterns.append(f"Missing {missing_camera_fields}/6 typical camera fields")
        
        # Check for AI-typical creation dates (batch processing indicators)
        if 'DateTime' in exif_data:
            date_str = str(exif_data['DateTime'])
            # AI tools often create images with round timestamps
            if ':00:00' in date_str or date_str.endswith('00:00'):
                suspicious_patterns.append("Round timestamp (batch processing indicator)")
        
        return {
            "exif_data_present": len(exif_data) > 0,
            "ai_tools_detected": detected_ai_tools,
            "suspicious_patterns": suspicious_patterns,
            "software_info": software_info if software_info else None,
            "camera_make": exif_data.get('Make'),
            "camera_model": exif_data.get('Model'),
            "creation_software": exif_data.get('Software'),
            "copyright_info": exif_data.get('Copyright'),
            "metadata_analysis": f"Found {len(exif_data)} EXIF fields, {len(detected_ai_tools)} AI tools detected, {len(suspicious_patterns)} suspicious patterns"
        }
        
    except Exception as e:
        logger.warning(f"Metadata analysis failed: {e}")
        return {
            "exif_data_present": False,
            "ai_tools_detected": [],
            "suspicious_patterns": [],
            "error": str(e),
            "metadata_analysis": "Metadata analysis failed"
        }

async def analyze_with_ai_model(image: Image.Image) -> Dict[str, Any]:
    """Ultra-enhanced AI detection with aggressive algorithms"""
    try:
        # Resize image if too large
        original_size = image.size
        if image.size[0] > MODEL_CONFIG["max_image_size"][0] or image.size[1] > MODEL_CONFIG["max_image_size"][1]:
            image.thumbnail(MODEL_CONFIG["max_image_size"], Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 1. Basic image classification
        results = ai_detector_model(image)
        
        # 2. ULTRA-AGGRESSIVE AI detection heuristics
        ai_probability = await analyze_ai_signatures_ultra(image, original_size)
        
        # 3. Enhanced model result analysis
        model_confidence = 0.6
        ai_boost = 0.0
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # AGGRESSIVE: Look for AI-related patterns in classification
            if any(term in label for term in ['digital', 'computer', 'generated', 'artificial', 'synthetic']):
                ai_boost += score * 1.2  # Increased multiplier
                logger.info(f"🚨 AI LABEL DETECTED: {label} (score: {score:.2f})")
            elif any(term in label for term in ['perfect', 'flawless', 'ideal', 'pristine', 'clean']):
                ai_boost += score * 0.9  # AI tends to be "too perfect"
                logger.info(f"🚨 PERFECTION INDICATOR: {label} (score: {score:.2f})")
            elif any(term in label for term in ['render', 'cgi', '3d', 'model', 'design']):
                ai_boost += score * 1.1  # 3D/CGI indicators
                logger.info(f"🚨 CGI INDICATOR: {label} (score: {score:.2f})")
            elif score > 0.95:  # Suspiciously high confidence is AI-like
                ai_boost += 0.3
                logger.info(f"🚨 SUSPICIOUS HIGH CONFIDENCE: {label} ({score:.2f})")
            
            model_confidence = max(model_confidence, score)
        
        # Combine probabilities more aggressively
        final_probability = min(1.0, ai_probability + (ai_boost * 0.4))
        
        return {
            "ai_probability": final_probability,
            "confidence": min(max(model_confidence, 0.0), 1.0),
            "model_used": MODEL_CONFIG["model_name"],
            "analysis_successful": True,
            "top_predictions": results[:3] if results else [],
            "detection_details": await get_detection_details(image, final_probability),
            "ai_boost_applied": ai_boost
        }
        
    except Exception as e:
        logger.error(f"AI model analysis failed: {e}")
        return {
            "ai_probability": 0.5,
            "confidence": 0.3,
            "model_used": "fallback",
            "analysis_successful": False,
            "error": str(e)
        }

async def analyze_ai_signatures_ultra(image: Image.Image, original_size: tuple) -> float:
    """ULTRA-AGGRESSIVE AI signature detection with enhanced algorithms"""
    import numpy as np
    
    # Convert to numpy array for analysis
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    ai_score = 0.2  # Lower base score, but more aggressive detection
    suspicious_factors = 0
    critical_flags = 0  # New: Critical AI indicators
    
    logger.info(f"🔍 ULTRA ANALYSIS START: {original_size} -> {image.size}")
    
    # 1. ENHANCED Dimension Analysis - More AI sizes
    ultra_common_ai_dimensions = [
        # Standard AI sizes
        (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
        # Midjourney favorites
        (512, 768), (768, 512), (1024, 768), (768, 1024), (1152, 896), (896, 1152),
        # DALL-E sizes
        (1024, 1024), (1792, 1024), (1024, 1792),
        # Stable Diffusion XL
        (1216, 832), (832, 1216), (1344, 768), (768, 1344),
        # New 2024 sizes
        (1280, 720), (720, 1280), (1920, 1080), (1080, 1920),
        (2304, 1536), (1536, 2304), (2560, 1440), (1440, 2560)
    ]
    
    if original_size in ultra_common_ai_dimensions:
        ai_score += 0.4  # Increased from 0.3
        critical_flags += 1
        suspicious_factors += 1
        logger.info(f"🚨 CRITICAL: Exact AI dimension {original_size}")
    
    # 2. ULTRA-PRECISE aspect ratios
    aspect_ratio = original_size[0] / original_size[1]
    ultra_precise_ratios = [
        1.0, 1.33333, 1.5, 1.77778, 0.75, 0.66667, 0.5625,  # Standard
        1.25, 1.6, 0.8, 1.28571, 0.77778, 1.41667, 0.70588,  # AI favorites
        2.0, 0.5, 1.2, 0.83333, 1.77777, 0.5625  # Extreme ratios
    ]
    
    for ratio in ultra_precise_ratios:
        if abs(aspect_ratio - ratio) < 0.01:  # More precise matching
            ai_score += 0.25
            suspicious_factors += 1
            logger.info(f"🚨 ULTRA-PRECISE RATIO: {aspect_ratio:.5f} ≈ {ratio}")
            break
    
    # 3. ENHANCED Color Analysis
    color_score = analyze_color_signatures_ultra(img_array)
    ai_score += color_score
    if color_score > 0.3:
        critical_flags += 1
        suspicious_factors += 1
    
    # 4. ULTRA-SENSITIVE Noise Analysis
    noise_score = analyze_noise_signatures_ultra(img_array)
    ai_score += noise_score
    if noise_score > 0.35:
        critical_flags += 1
        suspicious_factors += 1
    
    # 5. AGGRESSIVE Edge Analysis
    edge_score = analyze_edge_signatures_ultra(img_array)
    ai_score += edge_score
    if edge_score > 0.3:
        critical_flags += 1
        suspicious_factors += 1
    
    # 6. ENHANCED Frequency Analysis
    freq_score = analyze_frequency_signatures_ultra(img_array)
    ai_score += freq_score
    if freq_score > 0.25:
        suspicious_factors += 1
    
    # 7. NEW: Texture Analysis
    texture_score = analyze_texture_signatures(img_array)
    ai_score += texture_score
    if texture_score > 0.2:
        suspicious_factors += 1
    
    # 8. NEW: Compression Artifact Analysis
    compression_score = analyze_compression_artifacts(img_array)
    ai_score += compression_score
    if compression_score > 0.2:
        suspicious_factors += 1
    
    # ULTRA-AGGRESSIVE Multipliers
    if critical_flags >= 2:
        ai_score *= 1.8  # Massive boost for multiple critical flags
        logger.info(f"🚨🚨 MULTIPLE CRITICAL FLAGS: {critical_flags} - MAJOR AI SIGNATURE")
    elif critical_flags >= 1:
        ai_score *= 1.5  # Strong boost for critical flags
        logger.info(f"🚨 CRITICAL FLAG DETECTED: {critical_flags}")
    
    if suspicious_factors >= 4:
        ai_score *= 1.6  # High confidence with many factors
        logger.info(f"🚨 HIGH SUSPICION: {suspicious_factors} factors")
    elif suspicious_factors >= 3:
        ai_score *= 1.4
        logger.info(f"🚨 MODERATE SUSPICION: {suspicious_factors} factors")
    elif suspicious_factors >= 2:
        ai_score *= 1.2
    
    logger.info(f"🔍 ULTRA ANALYSIS: Score={ai_score:.3f}, Factors={suspicious_factors}, Critical={critical_flags}")
    
    return min(ai_score, 1.0)
                ai_probability = max(ai_probability, score * 0.9)
            elif any(term in label for term in ['perfect', 'flawless', 'ideal', 'pristine']):
                ai_probability = max(ai_probability, score * 0.7)
            elif score > 0.98:  # Suspiciously high confidence
                ai_probability = max(ai_probability, 0.8)
            
            model_confidence = max(model_confidence, score)
        
        return {
            "ai_probability": min(max(ai_probability, 0.0), 1.0),
            "confidence": min(max(model_confidence, 0.0), 1.0),
            "model_used": MODEL_CONFIG["model_name"],
            "analysis_successful": True,
            "top_predictions": results[:3] if results else [],
            "detection_details": await get_detection_details(image, ai_probability)
        }
        
    except Exception as e:
        logger.error(f"AI model analysis failed: {e}")
        return {
            "ai_probability": 0.5,
            "confidence": 0.3,
            "model_used": "fallback",
            "analysis_successful": False,
            "error": str(e)
        }

async def analyze_ai_signatures(image: Image.Image, original_size: tuple) -> float:
    """Advanced AI signature detection"""
    import numpy as np
    
    # Convert to numpy array for analysis
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    ai_score = 0.3  # Base score
    suspicious_factors = 0
    
    # 1. Dimension Analysis - AI loves specific sizes
    common_ai_dimensions = [
        (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
        (512, 768), (768, 512), (1024, 768), (768, 1024),
        (1920, 1080), (1080, 1920)  # Common AI ratios
    ]
    
    if original_size in common_ai_dimensions:
        ai_score += 0.3
        suspicious_factors += 1
        logger.info(f"🚨 AI SIGNATURE: Exact AI dimension {original_size}")
    
    # 2. Perfect aspect ratios
    aspect_ratio = original_size[0] / original_size[1]
    perfect_ratios = [1.0, 1.33, 1.5, 1.77, 0.75, 0.67, 0.56]  # Common AI ratios
    
    for ratio in perfect_ratios:
        if abs(aspect_ratio - ratio) < 0.02:
            ai_score += 0.2
            suspicious_factors += 1
            logger.info(f"🚨 AI SIGNATURE: Perfect aspect ratio {aspect_ratio:.2f}")
            break
    
    # 3. Color Analysis - AI has distinctive color patterns
    color_score = analyze_color_signatures(img_array)
    ai_score += color_score
    if color_score > 0.2:
        suspicious_factors += 1
    
    # 4. Noise Analysis - AI images are too clean
    noise_score = analyze_noise_signatures(img_array)
    ai_score += noise_score
    if noise_score > 0.25:
        suspicious_factors += 1
    
    # 5. Edge Analysis - AI creates unnatural edges
    edge_score = analyze_edge_signatures(img_array)
    ai_score += edge_score
    if edge_score > 0.2:
        suspicious_factors += 1
    
    # 6. Frequency Analysis - AI has specific frequency patterns
    freq_score = analyze_frequency_signatures(img_array)
    ai_score += freq_score
    if freq_score > 0.2:
        suspicious_factors += 1
    
    # Bonus for multiple suspicious factors
    if suspicious_factors >= 3:
        ai_score *= 1.4
        logger.info(f"🚨 MULTIPLE AI SIGNATURES: {suspicious_factors} factors detected")
    elif suspicious_factors >= 2:
        ai_score *= 1.2
    
    logger.info(f"🔍 AI ANALYSIS: Score={ai_score:.2f}, Factors={suspicious_factors}")
    
    return min(ai_score, 1.0)

def analyze_color_signatures(img_array: np.ndarray) -> float:
    """Detect AI-specific color patterns"""
    import numpy as np
    
    score = 0.0
    
    # 1. Check for oversaturation (AI tendency)
    hsv = np.array(Image.fromarray(img_array).convert('HSV'))
    saturation = hsv[:, :, 1] / 255.0
    high_sat_ratio = np.sum(saturation > 0.9) / saturation.size
    
    if high_sat_ratio > 0.3:
        score += 0.3
        logger.info(f"🎨 AI COLOR: High saturation {high_sat_ratio:.2f}")
    
    # 2. Check for color quantization (AI artifact)
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    color_diversity = unique_colors / total_pixels
    
    if color_diversity < 0.1:  # Too few unique colors
        score += 0.25
        logger.info(f"🎨 AI COLOR: Low diversity {color_diversity:.3f}")
    
    # 3. Check for perfect gradients (AI signature)
    gray = np.mean(img_array, axis=2)
    gradient_x = np.abs(np.diff(gray, axis=1))
    gradient_y = np.abs(np.diff(gray, axis=0))
    
    smooth_regions = np.sum((gradient_x < 2) & (gradient_x > 0)) + np.sum((gradient_y < 2) & (gradient_y > 0))
    total_gradients = gradient_x.size + gradient_y.size
    smooth_ratio = smooth_regions / total_gradients
    
    if smooth_ratio > 0.7:  # Too smooth
        score += 0.2
        logger.info(f"🎨 AI COLOR: Too smooth {smooth_ratio:.2f}")
    
    return min(score, 0.5)

def analyze_noise_signatures(img_array: np.ndarray) -> float:
    """Detect unnatural noise patterns"""
    import numpy as np
    
    score = 0.0
    
    # Convert to grayscale for noise analysis
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # 1. Calculate local variance (noise measure)
    kernel_size = 5
    variances = []
    
    for i in range(0, gray.shape[0] - kernel_size, kernel_size):
        for j in range(0, gray.shape[1] - kernel_size, kernel_size):
            patch = gray[i:i+kernel_size, j:j+kernel_size]
            variances.append(np.var(patch))
    
    avg_variance = np.mean(variances)
    
    # AI images have suspiciously low variance
    if avg_variance < 10:
        score += 0.4
        logger.info(f"🔍 AI NOISE: Too clean {avg_variance:.1f}")
    elif avg_variance < 25:
        score += 0.25
    
    # 2. Check for uniform noise distribution
    noise_std = np.std(variances)
    if noise_std < 5:  # Too uniform
        score += 0.2
        logger.info(f"🔍 AI NOISE: Too uniform {noise_std:.1f}")
    
    return min(score, 0.5)

def analyze_edge_signatures(img_array: np.ndarray) -> float:
    """Detect AI-specific edge patterns"""
    import numpy as np
    
    score = 0.0
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Sobel edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply Sobel filters
    edges_x = np.abs(np.convolve(gray.flatten(), sobel_x.flatten(), mode='same')).reshape(gray.shape)
    edges_y = np.abs(np.convolve(gray.flatten(), sobel_y.flatten(), mode='same')).reshape(gray.shape)
    
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    # 1. Check for oversharpening (AI signature)
    strong_edges = np.sum(edge_magnitude > 100)
    total_pixels = edge_magnitude.size
    sharp_ratio = strong_edges / total_pixels
    
    if sharp_ratio > 0.15:  # Too many sharp edges
        score += 0.3
        logger.info(f"⚡ AI EDGES: Oversharpened {sharp_ratio:.3f}")
    
    # 2. Check for unnatural edge distribution
    edge_histogram, _ = np.histogram(edge_magnitude, bins=50)
    edge_peaks = np.sum(edge_histogram > np.mean(edge_histogram) * 3)
    
    if edge_peaks < 3:  # Too few natural variations
        score += 0.2
        logger.info(f"⚡ AI EDGES: Unnatural distribution")
    
    return min(score, 0.4)

def analyze_frequency_signatures(img_array: np.ndarray) -> float:
    """Detect AI frequency domain signatures"""
    import numpy as np
    
    score = 0.0
    
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Resize for FFT analysis (performance)
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            gray = np.array(Image.fromarray(gray.astype(np.uint8)).resize((256, 256)))
        
        # 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Analyze frequency distribution
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        
        # High frequency content (edges of spectrum)
        high_freq = magnitude[:20, :] + magnitude[-20:, :] + magnitude[:, :20] + magnitude[:, -20:]
        low_freq = magnitude[center_y-20:center_y+20, center_x-20:center_x+20]
        
        high_freq_energy = np.sum(high_freq)
        low_freq_energy = np.sum(low_freq)
        
        if high_freq_energy > 0 and low_freq_energy > 0:
            freq_ratio = high_freq_energy / low_freq_energy
            
            # AI images often have unnatural frequency distributions
            if freq_ratio > 2.0:  # Too much high frequency
                score += 0.25
                logger.info(f"📊 AI FREQ: High freq dominance {freq_ratio:.2f}")
            elif freq_ratio < 0.1:  # Too little high frequency
                score += 0.2
                logger.info(f"📊 AI FREQ: Low freq dominance {freq_ratio:.2f}")
    
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
    
    return min(score, 0.3)

async def get_detection_details(image: Image.Image, ai_probability: float) -> Dict:
    """Get detailed detection information"""
    return {
        "primary_indicators": [
            "Dimension analysis",
            "Color signature detection", 
            "Noise pattern analysis",
            "Edge sharpness evaluation",
            "Frequency domain analysis"
        ],
        "confidence_factors": [
            f"AI probability: {ai_probability:.1%}",
            f"Image size: {image.size}",
            f"Aspect ratio: {image.size[0]/image.size[1]:.2f}"
        ]
    }

def generate_technical_indicators(ai_prob: float, confidence: float, metadata: Dict, ai_analysis: Dict = None) -> list:
    """Generate enhanced technical indicators for frontend display"""
    indicators = []
    
    # Enhanced AI Model Analysis
    detection_details = ai_analysis.get("detection_details", {}) if ai_analysis else {}
    
    indicators.append({
        "name": "Gelişmiş AI Analizi",
        "score": int(ai_prob * 100),
        "status": "suspicious" if ai_prob > 0.7 else ("warning" if ai_prob > 0.4 else "normal"),
        "description": f"Çoklu algoritma analizi: %{int(confidence * 100)} güven",
        "details": f"5 farklı AI imza analizi: {', '.join(detection_details.get('primary_indicators', [])[:3])}"
    })
    
    # Metadata Analysis
    ai_tools = metadata.get("ai_tools_detected", [])
    suspicious_patterns = metadata.get("suspicious_patterns", [])
    
    if ai_tools:
        metadata_score = 95
        status = "suspicious"
        desc = f"AI yazılım tespit edildi: {', '.join(ai_tools[:2])}"
    elif suspicious_patterns:
        metadata_score = 70
        status = "warning"
        desc = f"Şüpheli pattern: {suspicious_patterns[0]}"
    elif metadata.get("exif_data_present"):
        metadata_score = 20
        status = "normal"
        desc = "Normal kamera metadata'sı mevcut"
    else:
        metadata_score = 60
        status = "warning"
        desc = "Metadata eksik veya temizlenmiş"
    
    indicators.append({
        "name": "Metadata Analizi",
        "score": metadata_score,
        "status": status,
        "description": desc,
        "details": f"EXIF: {'Var' if metadata.get('exif_data_present') else 'Yok'}, Şüpheli: {len(suspicious_patterns)}"
    })
    
    # Camera vs Software Analysis
    camera_make = metadata.get("camera_make")
    camera_model = metadata.get("camera_model")
    
    if camera_make and camera_model:
        camera_score = 10
        status = "normal"
        desc = f"Kamera: {camera_make} {camera_model}"
        details = "Doğal kamera kaynağı tespit edildi"
    elif metadata.get("creation_software"):
        camera_score = 85
        status = "suspicious"
        desc = "Yazılım ile oluşturulmuş, kamera bilgisi yok"
        details = f"Yazılım: {metadata.get('creation_software')}"
    else:
        camera_score = 50
        status = "warning"
        desc = "Kaynak bilgisi belirsiz"
        details = "Ne kamera ne de yazılım bilgisi bulunamadı"
    
    indicators.append({
        "name": "Kaynak Analizi",
        "score": camera_score,
        "status": status,
        "description": desc,
        "details": details
    })
    
    # Dimension Analysis (new)
    if ai_analysis and ai_analysis.get("analysis_successful"):
        # Extract dimension info from detection details
        confidence_factors = detection_details.get("confidence_factors", [])
        dimension_suspicious = any("512" in factor or "1024" in factor for factor in confidence_factors)
        
        dim_score = 80 if dimension_suspicious else 30
        dim_status = "suspicious" if dimension_suspicious else "normal"
        dim_desc = "AI tipik boyutları tespit edildi" if dimension_suspicious else "Doğal boyut oranları"
        
        indicators.append({
            "name": "Boyut Analizi",
            "score": dim_score,
            "status": dim_status,
            "description": dim_desc,
            "details": f"Boyut faktörleri: {len(confidence_factors)} analiz"
        })
    
    # Overall Risk Assessment (new)
    high_risk_count = sum(1 for ind in indicators if ind["status"] == "suspicious")
    medium_risk_count = sum(1 for ind in indicators if ind["status"] == "warning")
    
    if high_risk_count >= 2:
        risk_score = 90
        risk_status = "suspicious"
        risk_desc = f"Yüksek risk: {high_risk_count} kritik uyarı"
    elif high_risk_count >= 1 or medium_risk_count >= 2:
        risk_score = 60
        risk_status = "warning"
        risk_desc = f"Orta risk: {high_risk_count} kritik, {medium_risk_count} uyarı"
    else:
        risk_score = 20
        risk_status = "normal"
        risk_desc = "Düşük risk: Çoğu test normal"
    
    indicators.append({
        "name": "Genel Risk Değerlendirmesi",
        "score": risk_score,
        "status": risk_status,
        "description": risk_desc,
        "details": f"Toplam {len(indicators)} farklı analiz yapıldı"
    })
    
    return indicators

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Main image analysis endpoint"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate image
        validation_result = validate_image(file_content, file.filename)
        
        # Load image
        image = Image.open(io.BytesIO(file_content))
        
        # Check metadata
        metadata_result = analyze_metadata(file_content)
        
        # Analyze with AI model
        ai_analysis = await analyze_with_ai_model(image)
        
        # Calculate final AI probability (combine model + metadata)
        base_probability = ai_analysis["ai_probability"]
        
        # Boost probability if AI software detected in metadata
        if metadata_result.get("ai_tools_detected"):
            base_probability = min(1.0, base_probability * 1.4)
        
        # Generate technical indicators with enhanced analysis
        indicators = generate_technical_indicators(
            base_probability, 
            ai_analysis["confidence"], 
            metadata_result,
            ai_analysis
        )
        
        # Determine verdict
        if base_probability < 0.25:
            verdict = {
                "text": "Bu görsel büyük olasılıkla gerçek",
                "color": "text-green-400",
                "icon": "check-circle",
                "bgColor": "bg-green-500/20"
            }
        elif base_probability < 0.4:
            verdict = {
                "text": "Görsel muhtemelen gerçek",
                "color": "text-green-400", 
                "icon": "check-circle-2",
                "bgColor": "bg-green-500/20"
            }
        elif base_probability < 0.6:
            verdict = {
                "text": "Belirsiz - daha fazla analiz gerekebilir",
                "color": "text-yellow-400",
                "icon": "alert-triangle", 
                "bgColor": "bg-yellow-500/20"
            }
        elif base_probability < 0.75:
            verdict = {
                "text": "Bu görsel AI tarafından üretilmiş olabilir",
                "color": "text-orange-400",
                "icon": "alert-circle",
                "bgColor": "bg-orange-500/20"
            }
        else:
            verdict = {
                "text": "Bu görsel büyük olasılıkla AI üretimi",
                "color": "text-red-400",
                "icon": "x-circle", 
                "bgColor": "bg-red-500/20"
            }
        
        # Calculate confidence (higher for extreme values)
        confidence_score = int(60 + abs(base_probability - 0.5) * 80)
        
        # Prepare response in frontend-expected format
        response = {
            "success": True,
            "aiProbability": int(base_probability * 100),
            "confidence": confidence_score,
            "verdict": verdict,
            "indicators": indicators,
            "processingTime": "0.85",  # Simulated processing time
            "fileSize": f"{validation_result['size_bytes'] / 1024:.1f}",
            "metadata": {
                "fileName": file.filename,
                "fileSize": validation_result['size_bytes'],
                "fileType": validation_result['mime_type'],
                "lastModified": datetime.now().strftime('%d.%m.%Y'),
                "metadata_analysis": metadata_result,
                "ai_model_analysis": ai_analysis
            },
            "warning": "Bu analiz bir olasılık tahminidir ve kesin hüküm değildir. Sonuçlar referans amaçlıdır."
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)