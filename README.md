# 🔬 AI Image Detector - Forensic Analysis Platform

[![License: WATAM](https://img.shields.io/badge/License-WATAM-blue.svg)](https://WeAreTheArtMakers.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-orange.svg)](https://huggingface.co/transformers/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/wearetheartmakers/ai-image-detector)

A production-grade **forensic image analysis platform** for detecting AI-generated images. Features multi-model ensemble detection, CLIP semantic analysis, manipulation detection with heatmap visualization, OCR-based text forensics, generator fingerprinting, and comprehensive metadata forensics.

![AI Image Detector](ai-detect.PNG)

## ✨ Key Features

### 🧠 Multi-Model Ensemble AI Detection
- **Ensemble Analysis**: Multiple AI detection models with uncertainty quantification
- **CLIP Semantic Analysis**: Content type classification (photo vs CGI/illustration)
- **Calibrated Probabilities**: Confidence intervals and model agreement metrics
- **Two-Axis Output**: AI likelihood + evidential quality scoring

### 🔬 Advanced Forensic Analysis
- **Diffusion Fingerprint Detection**: Identifies diffusion model artifacts in frequency domain
- **Generation Pathway Classification**: T2I vs I2I vs Real Photo with evidence-based gating
- **Enhanced I2I Detection**: Perfect square detection, noise uniformity variation, model disagreement analysis
- **JPEG Forensics**: Double compression detection, quantization fingerprinting
- **Manipulation Detection**: Copy-move, splice, edge matte, blur/noise mismatch
- **Evidence Fusion**: Weighted combination of multiple forensic signals

### 📝 Text Forensics (OCR)
- **OCR Detection**: EasyOCR-based text extraction with optimized parameters
- **Multi-language OCR**: 7 languages supported (EN, TR, ES, DE, FR, IT, PT)
- **Image Preprocessing**: Auto contrast/sharpness enhancement, upscaling for small images
- **Gibberish Detection**: Identifies nonsensical AI-generated text on signs/labels
- **AI Text Artifacts**: Detects malformed characters, unusual casing, invalid words
- **Gibberish Boost**: High gibberish ratio increases AI probability (+5% to +15%)

### 🔍 Generator Fingerprinting
- **AI Tool Identification**: Statistical analysis to identify likely generator
- **Supported Generators**: Midjourney V5/V6, DALL-E 3, Stable Diffusion XL, Flux, Grok/Aurora, Firefly, Ideogram, Gemini Imagen, ChatGPT/DALL-E
- **Dimension Matching**: Known AI output dimensions (1024x1024, 784x1168, etc.) boost detection
- **Spectrum/Noise/Color Analysis**: Frequency rolloff, noise uniformity, color signatures
- **Confidence Scoring**: Match confidence with feature breakdown

### 🎨 Forensic Visualization
- **Heatmap Overlays**: Visual representation of detected anomalies
- **Global Edit Mode**: Shows processing intensity for filters/color grading
- **Local Manipulation Mode**: Highlights boundary-corroborated suspicious regions
- **T2I Artifacts Mode**: Visualizes AI generation artifacts (diffusion signatures)
- **Deterministic Generation**: SHA256 hashes for integrity verification

### 📊 Edit Assessment System
- **3-Way Edit Taxonomy**: `none_detected`, `global_postprocess`, `local_manipulation`, `generator_artifacts`
- **Boundary Corroboration**: Local manipulation requires edge_matte/copy_move/inpainting evidence ≥0.60
- **False Positive Prevention**: Noise inconsistency alone does NOT trigger composite labels
- **AI Generation Artifacts**: Distinguishes AI texture artifacts from true manipulation

### 💭 Prompt Analysis
- **Prompt Recovery**: Extracts exact prompts from AI tool metadata (Midjourney, DALL-E, etc.)
- **Prompt Reconstruction**: Infers plausible prompts from visual content analysis
- **BLIP Captioning**: AI-powered image description for prompt inference
- **Clear Distinction**: "RECOVERED" (exact) vs "RECONSTRUCTED" (hypothesis)

### 🗺️ GPS & Provenance
- **GPS Verification**: Location data with confidence scoring
- **Provenance Scoring**: EXIF/GPS evidence weighting for authenticity
- **Platform Detection**: Instagram, WhatsApp re-encoding detection
- **Old Timestamp Bonus**: Pre-2020 images get authenticity boost

### 🛠️ Metadata & Camera Evidence
- **EXIF/XMP/IPTC Extraction**: Comprehensive metadata analysis
- **Camera Evidence**: CFA artifacts, PRNU proxy, camera settings
- **AI Software Detection**: 20+ AI tool signatures (Midjourney, DALL-E, Stable Diffusion, etc.)

### 🌐 Internationalization
- **Bilingual Support**: Turkish (TR) and English (EN)
- **Dynamic Language Switching**: Real-time UI translation
- **Localized Verdicts**: Evidence-based verdict text in both languages

## 🏗️ Architecture

```
ai-image-detector/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── forensic/
│   │   ├── ai_detector.py           # Multi-model ensemble
│   │   ├── content_type.py          # CLIP-based content classification
│   │   ├── diffusion_fingerprint.py # Diffusion model detection
│   │   ├── pathway_classifier.py    # T2I/I2I/Real classification
│   │   ├── manipulation.py          # Manipulation detection
│   │   ├── edit_assessment.py       # Edit type taxonomy
│   │   ├── visualization.py         # Heatmap generation
│   │   ├── verdict_generator.py     # Evidence-based verdicts
│   │   ├── text_forensics.py        # OCR + AI text artifact detection
│   │   ├── generator_fingerprint.py # AI tool identification
│   │   ├── prompt_analysis.py       # Unified prompt recovery/reconstruction
│   │   ├── prompt_recovery.py       # Metadata-based prompt extraction
│   │   ├── content_extraction.py    # BLIP captioning
│   │   ├── provenance_score.py      # EXIF/GPS evidence scoring
│   │   ├── evidence_fusion.py       # Weighted evidence combination
│   │   ├── metadata.py              # EXIF/XMP/IPTC extraction
│   │   ├── jpeg_forensics.py        # JPEG analysis
│   │   ├── statistics.py            # Calibration & uncertainty
│   │   └── config.py                # Feature flags
│   └── requirements.txt
├── index.html                       # Frontend UI
├── script.js                        # Analysis display logic
├── style.css                        # Modern dark theme
├── language-manager.js              # i18n support
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB recommended for full model ensemble)
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone repository
git clone https://github.com/wearetheartmakers/ai-image-detector.git
cd ai-image-detector

# Setup virtual environment
./setup_venv.sh          # macOS/Linux
# or
setup_venv.bat           # Windows

# Start backend
./run_backend_venv.sh    # Terminal 1

# Start frontend (separate terminal)
./run_frontend.sh        # Terminal 2
```

### Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📡 API Reference

### POST /analyze
Analyze an image for AI generation and manipulation.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze?file_chain=original" \
  -F "file=@image.jpg"
```

**Key Response Fields:**
```json
{
  "success": true,
  "report_id": "abc123",
  "evidence_level": "STRONG_AI_EVIDENCE",
  "ai_probability": 78,
  "confidence": 65,
  
  "verdict_text": {
    "verdict_key": "AI_T2I_HIGH",
    "title_en": "AI Generated (T2I)",
    "subtitle_en": "High confidence text-to-image generation"
  },
  
  "pathway": {
    "pred": "t2i",
    "confidence": "high",
    "evidence": {
      "diffusion_score": 0.82,
      "camera_evidence_score": 0.08
    }
  },
  
  "text_forensics": {
    "has_text": true,
    "text_count": 5,
    "ai_text_score": 0.72,
    "verdict": "likely_ai",
    "ai_text_evidence": ["'KOEMO': Gibberish/nonsensical text"]
  },
  
  "generator_fingerprint": {
    "top_match": {
      "name": "Grok Aurora",
      "family": "grok",
      "confidence": 0.65
    }
  },
  
  "prompt_analysis": {
    "source": "RECONSTRUCTED_MEDIUM",
    "has_reconstructed": true,
    "reconstructed": {
      "short_en": "European street scene, parked cars, sunny day",
      "confidence": "medium"
    }
  }
}
```

## 🔧 Configuration

### Feature Flags (backend/forensic/config.py)
```python
# Core Features
ENABLE_AI_DETECTION = True
ENABLE_JPEG_FORENSICS = True
ENABLE_MANIPULATION_MODULES = True

# Advanced Features
ENABLE_PATHWAY_CLASSIFIER = True
ENABLE_DIFFUSION_FINGERPRINT = True
ENABLE_TEXT_FORENSICS = True          # OCR + AI text artifacts
ENABLE_GENERATOR_FINGERPRINT = True   # AI tool identification
ENABLE_PROMPT_RECOVERY = True         # Metadata prompt extraction
ENABLE_BLIP_CAPTIONING = True         # Visual prompt inference

# Output Features
ENABLE_TWO_AXIS_OUTPUT = True
ENABLE_VERDICT_TEXT = True
ENABLE_GENERATIVE_HEATMAP = True
```

### Key Thresholds
| Parameter | Value | Description |
|-----------|-------|-------------|
| `diffusion_high_threshold` | 0.70 | High confidence diffusion detection |
| `camera_evidence_threshold` | 0.35 | Min camera evidence for "Photo" verdict |
| `boundary_corroboration_threshold` | 0.60 | Min score for local manipulation |
| `pAI_threshold` | 0.40 | Min AI probability for "Likely AI" verdict |
| `gibberish_ratio_threshold` | 0.50 | High gibberish = likely AI text |
| `generator_match_threshold` | 0.50 | Medium confidence generator match |

## 📊 Detection Accuracy

| Content Type | Detection Rate | Notes |
|--------------|----------------|-------|
| T2I (Midjourney, DALL-E, SD) | 85-95% | High diffusion fingerprint |
| Photoreal AI (Grok, Gemini) | 75-90% | CLIP photoreal hints |
| Real Photos | 90-95% | Camera evidence + low diffusion |
| AI Text Artifacts | 70-85% | Gibberish on signs/labels |
| Composites/Splices | 80-90% | Boundary corroboration required |

## ⚠️ Important Disclaimers

> **This tool provides probabilistic analysis, not definitive proof.**
> 
> - Results are NOT 100% accurate
> - "Definitive" verdicts require cryptographic proof (C2PA) or AI software signatures
> - False positives/negatives are possible
> - Professional verification recommended for critical decisions
> - AI detection technology evolves rapidly; regular updates required

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Or manually
docker build -t ai-detector ./backend
docker run -p 8000:8000 ai-detector
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the [WATAM License](LICENSE) - WeAreTheArtMakers.com

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) - Transformers & CLIP models
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [OpenAI CLIP](https://openai.com/research/clip) - Vision-language model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
- [Lucide Icons](https://lucide.dev/) - Beautiful icon set

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by [WATAM](https://github.com/wearetheartmakers)

</div>
