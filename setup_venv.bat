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

REM AI Detection modelini önceden indir
echo 🤖 AI Detection modeli indiriliyor (umm-maybe/AI-image-detector)...
echo    Bu işlem ilk seferde ~350MB indirecek, lütfen bekleyin...
python -c "from transformers import pipeline; print('📥 Model indiriliyor...'); detector = pipeline('image-classification', model='umm-maybe/AI-image-detector'); print('✅ Model başarıyla indirildi!')"

cd ..

echo.
echo ✅ Kurulum tamamlandı!
echo.
echo 🎯 Çalıştırma komutları:
echo    Forensic Backend: run_forensic_backend.bat
echo    Frontend: run_frontend.bat
echo.
echo 📍 Erişim adresleri:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8001
echo    API Docs: http://localhost:8001/docs
pause