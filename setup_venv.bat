@echo off
echo 🚀 AI Image Detector - Sanal Ortam Kurulumu
echo ============================================

REM Ana dizinde sanal ortam oluştur
echo 📦 Python sanal ortamı oluşturuluyor...
python -m venv ai_detector_env

REM Sanal ortamı aktifleştir
echo 🔧 Sanal ortam aktifleştiriliyor...
call ai_detector_env\Scripts\activate.bat

REM Pip'i güncelle
echo ⬆️ Pip güncelleniyor...
python -m pip install --upgrade pip

REM Backend dependencies'i yükle
echo 📚 Backend bağımlılıkları yükleniyor...
cd backend
pip install -r requirements.txt

echo ✅ Kurulum tamamlandı!
echo.
echo 🎯 Çalıştırma komutları:
echo Backend: run_backend_venv.bat
echo Frontend: run_frontend.bat
echo.
echo 📍 Erişim adresleri:
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
pause