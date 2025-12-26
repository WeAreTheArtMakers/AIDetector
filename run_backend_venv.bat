@echo off
echo 🚀 AI Image Detector Backend (Sanal Ortam)
echo ==========================================

REM Sanal ortamı aktifleştir
echo 🔧 Sanal ortam aktifleştiriliyor...
call ai_detector_env\Scripts\activate.bat

REM Backend dizinine git
cd backend

REM Uvicorn ile başlat
echo 🌟 Backend başlatılıyor...
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause