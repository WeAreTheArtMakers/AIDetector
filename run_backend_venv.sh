#!/bin/bash
echo "🚀 AI Image Detector Backend (Sanal Ortam)"
echo "=========================================="

# Sanal ortamı aktifleştir
echo "🔧 Sanal ortam aktifleştiriliyor..."
source ai_detector_env/bin/activate

# Backend dizinine git
cd backend

# Uvicorn ile başlat
echo "🌟 Backend başlatılıyor..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload