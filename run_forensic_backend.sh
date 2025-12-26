#!/bin/bash

echo "🚀 AI Image Detector - Forensic Backend"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "ai_detector_env" ]; then
    echo "❌ Virtual environment not found. Run setup_venv.sh first."
    exit 1
fi

echo "🔧 Activating virtual environment..."
source ai_detector_env/bin/activate

echo "🌟 Starting Forensic Backend..."
echo "📍 API Documentation: http://localhost:8001/docs"
echo "📋 Health Check: http://localhost:8001/health"
echo "🔍 Forensic Analysis: http://localhost:8001/analyze_forensic"

cd backend
uvicorn main_forensic:app --host 0.0.0.0 --port 8001 --reload