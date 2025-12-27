@echo off
chcp 65001 >nul
cls

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║     🔬 AI Image Detector - Forensic Analysis Platform        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check if virtual environment exists
if not exist "ai_detector_env" (
    echo ⚠️  Virtual environment not found. Running setup...
    call setup_venv.bat
)

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call ai_detector_env\Scripts\activate.bat

cd backend

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo                     📋 Dependency Check
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

:: Check core dependencies
python -c "import fastapi" 2>nul && (echo   ✓ FastAPI) || (echo   ✗ FastAPI - installing... && pip install fastapi -q)
python -c "import uvicorn" 2>nul && (echo   ✓ Uvicorn) || (echo   ✗ Uvicorn - installing... && pip install uvicorn[standard] -q)
python -c "import PIL" 2>nul && (echo   ✓ Pillow) || (echo   ✗ Pillow - installing... && pip install pillow -q)
python -c "import torch" 2>nul && (echo   ✓ PyTorch) || (echo   ✗ PyTorch - installing... && pip install torch -q)
python -c "import transformers" 2>nul && (echo   ✓ Transformers) || (echo   ✗ Transformers - installing... && pip install transformers -q)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo                     📝 OCR Support (Optional)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python -c "import easyocr" 2>nul && (echo   ✓ EasyOCR - Text Forensics enabled) || (echo   ○ EasyOCR not installed - Text Forensics disabled && echo     → Install with: pip install easyocr)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo                     🧠 AI Models
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   ○ Primary: umm-maybe/AI-image-detector
echo   ○ Secondary: Organika/sdxl-detector
echo   ○ CLIP: openai/clip-vit-base-patch32
echo   ○ BLIP: Salesforce/blip-image-captioning-base
echo.
echo   ℹ️  Models will be downloaded on first run (~500MB)
echo.

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🚀 Starting Backend Server...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.

:: Start frontend in new window
start "Frontend Server" cmd /c "cd .. && python -m http.server 3000"

:: Wait and open browser
timeout /t 5 /nobreak >nul
start http://localhost:3000

:: Start backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
