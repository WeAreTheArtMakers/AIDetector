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

echo "✅ Kurulum tamamlandı!"
echo ""
echo "🎯 Çalıştırma komutları:"
echo "Backend: ./run_backend_venv.sh"
echo "Frontend: ./run_frontend.sh"
echo ""
echo "📍 Erişim adresleri:"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"