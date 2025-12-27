# AI Image Detector - Setup Guide

## 🚀 Quick Start (Recommended)

### 1. Setup Virtual Environment
```bash
# macOS/Linux
./setup_venv.sh

# Windows
setup_venv.bat
```

### 2. Run the Application
```bash
# Terminal 1 - Backend (Virtual Environment)
./run_backend_venv.sh     # macOS/Linux
run_backend_venv.bat      # Windows

# Terminal 2 - Frontend
./run_frontend.sh         # macOS/Linux  
run_frontend.bat          # Windows
```

## 🌐 Access URLs
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ⏱️ First Run
- **Model Download**: First run downloads ~500MB models (3-10 minutes)
- **Subsequent Starts**: ~15-30 seconds

## 📦 Dependencies

### Required
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- 2GB+ disk space

### Optional (for full features)
- **EasyOCR**: For text forensics (OCR)
  ```bash
  pip install easyocr
  ```
- **CUDA GPU**: For faster inference

## 🔧 Manual Installation

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
python3 -m http.server 3000
```

## 🧪 Testing
1. Go to Frontend: http://localhost:3000
2. Upload an image (JPG, PNG, WebP, HEIC)
3. Click "Forensic Analysis" button
4. View AI analysis results

## ⚠️ Troubleshooting

### "Address already in use" Error:
```bash
# Try different port
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Update BACKEND_URL in script.js
const BACKEND_URL = 'http://localhost:8001';
```

### Model Download Error:
- Check internet connection
- Check disk space (min 2GB free)
- Check firewall/antivirus settings

### CORS Error:
- Ensure backend is running
- Check browser console for details
- Check backend logs

### OCR Not Working:
```bash
# Install EasyOCR
pip install easyocr

# Or install Tesseract
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
pip install pytesseract
```

### Memory Error:
- Close other applications
- Use smaller images (< 2048px)
- Reduce batch size in config

## 📊 Performance
- **First model load**: 3-10 minutes
- **Analysis time**: 3-8 seconds
- **Supported formats**: JPG, PNG, WebP, HEIC, BMP
- **Max file size**: 10MB
- **Recommended image size**: 2048x2048 and below

## 🔒 Security Notes
- This is a forensic analysis tool
- Results are probabilistic, not definitive
- For production use, add authentication
- Do not expose backend directly to internet

## 📝 Feature Flags

Edit `backend/forensic/config.py` to enable/disable features:

```python
ENABLE_TEXT_FORENSICS = True          # OCR + AI text artifacts
ENABLE_GENERATOR_FINGERPRINT = True   # AI tool identification
ENABLE_PROMPT_RECOVERY = True         # Metadata prompt extraction
ENABLE_BLIP_CAPTIONING = True         # Visual prompt inference
ENABLE_GENERATIVE_HEATMAP = True      # AI artifact visualization
```
