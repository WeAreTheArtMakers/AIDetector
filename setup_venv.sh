#!/bin/bash
echo "🚀 AI Image Detector - Sanal Ortam Kurulumu"
echo "============================================"

# Ana dizinde sanal ortam oluştur
echo "📦 Python sanal ortamı oluşturuluyor..."
python3 -m venv ai_detector_env

# Sanal ortamı aktifleştir
echo "🔧 Sanal ortam aktifleştiriliyor..."
source ai_detector_env/bin/activate

# Pip'i güncelle
echo "⬆️ Pip güncelleniyor..."
pip install --upgrade pip

# Backend dependencies'i yükle
echo "📚 Backend bağımlılıkları yükleniyor..."
cd backend
pip install -r requirements.txt

# AI Detection modelini önceden indir
echo "🤖 AI Detection modeli indiriliyor (umm-maybe/AI-image-detector)..."
echo "   Bu işlem ilk seferde ~350MB indirecek, lütfen bekleyin..."
python3 -c "
from transformers import pipeline
print('📥 Model indiriliyor...')
try:
    detector = pipeline('image-classification', model='umm-maybe/AI-image-detector')
    print('✅ Model başarıyla indirildi!')
except Exception as e:
    print(f'⚠️ Model indirilemedi: {e}')
    print('   Uygulama ilk çalıştırmada indirecek.')
"

cd ..

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "🎯 Çalıştırma komutları:"
echo "   Forensic Backend: ./run_forensic_backend.sh"
echo "   Frontend: ./run_frontend.sh"
echo ""
echo "📍 Erişim adresleri:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8001"
echo "   API Docs: http://localhost:8001/docs"