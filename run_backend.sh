#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║${NC}     ${CYAN}🔬 AI Image Detector - Forensic Analysis Platform${NC}        ${PURPLE}║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "ai_detector_env" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Running setup...${NC}"
    ./setup_venv.sh
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source ai_detector_env/bin/activate

# Check and install dependencies
echo -e "${BLUE}📦 Checking dependencies...${NC}"
cd backend

# Check required packages
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                    📋 Dependency Check${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Core dependencies
check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "  ${RED}✗${NC} $2 ${YELLOW}(installing...)${NC}"
        pip install $3 -q
        return 1
    fi
}

check_package "fastapi" "FastAPI" "fastapi"
check_package "uvicorn" "Uvicorn" "uvicorn[standard]"
check_package "PIL" "Pillow" "pillow"
check_package "torch" "PyTorch" "torch"
check_package "transformers" "Transformers" "transformers"
check_package "scipy" "SciPy" "scipy"
check_package "numpy" "NumPy" "numpy"

# Optional OCR
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                    📝 OCR Support (Optional)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if python3 -c "import easyocr" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} EasyOCR (Text Forensics enabled)"
else
    echo -e "  ${YELLOW}○${NC} EasyOCR not installed (Text Forensics disabled)"
    echo -e "    ${YELLOW}→ Install with: pip install easyocr${NC}"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                    🧠 AI Models${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BLUE}○${NC} Primary: umm-maybe/AI-image-detector"
echo -e "  ${BLUE}○${NC} Secondary: Organika/sdxl-detector"
echo -e "  ${BLUE}○${NC} CLIP: openai/clip-vit-base-patch32"
echo -e "  ${BLUE}○${NC} BLIP: Salesforce/blip-image-captioning-base"
echo ""
echo -e "  ${YELLOW}ℹ️  Models will be downloaded on first run (~500MB)${NC}"
echo ""

# Start backend
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 Starting Backend Server...${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC} http://localhost:8000/docs"
echo ""

# Start frontend in background and open browser
(
    sleep 3
    # Start frontend server
    cd ..
    python3 -m http.server 3000 &>/dev/null &
    FRONTEND_PID=$!
    
    # Wait for backend to be ready
    echo -e "${YELLOW}⏳ Waiting for backend to initialize...${NC}"
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${GREEN}✅ All systems ready!${NC}"
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo -e "  ${CYAN}🌐 Frontend:${NC} http://localhost:3000"
            echo -e "  ${CYAN}🔧 Backend:${NC}  http://localhost:8000"
            echo -e "  ${CYAN}📚 API Docs:${NC} http://localhost:8000/docs"
            echo ""
            
            # Open browser
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open http://localhost:3000
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                xdg-open http://localhost:3000 2>/dev/null || sensible-browser http://localhost:3000 2>/dev/null
            fi
            break
        fi
        sleep 1
    done
) &

# Run uvicorn
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
