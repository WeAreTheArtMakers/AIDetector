# AI Image Detector - Sanal Ortam Kurulumu

## 🚀 Hızlı Başlangıç (Önerilen)

### 1. Sanal Ortam Kurulumu
```bash
# macOS/Linux
./setup_venv.sh

# Windows
setup_venv.bat
```

### 2. Uygulamayı Çalıştırma
```bash
# Terminal 1 - Backend (Sanal Ortam)
./run_backend_venv.sh     # macOS/Linux
run_backend_venv.bat      # Windows

# Terminal 2 - Frontend
./run_frontend.sh         # macOS/Linux  
run_frontend.bat          # Windows
```

## 🌐 Erişim Adresleri
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## ⏱️ İlk Çalıştırma
- **Model İndirme**: İlk çalıştırmada ~346MB model indirilir (2-5 dakika)
- **Sonraki Başlatmalar**: ~10-15 saniye

## 🔧 Manuel Kurulum (Alternatif)

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend
```bash
python3 -m http.server 3000
```

## 🧪 Test Etme
1. Frontend'e git: http://localhost:3000
2. Bir görsel yükle (JPG, PNG, WebP)
3. "Analiz Et" butonuna bas
4. Backend'den gelen AI analiz sonuçlarını gör

## ⚠️ Sorun Giderme

### "Address already in use" Hatası:
```bash
# Port 8001 kullanımda ise farklı port dene
python3 -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# script.js'de BACKEND_URL'i güncelle
const BACKEND_URL = 'http://localhost:8002';
```

### Model İndirme Hatası:
- İnternet bağlantısını kontrol et
- Disk alanını kontrol et (min 1GB boş alan)
- Firewall/antivirus ayarlarını kontrol et

### CORS Hatası:
- Backend'in çalıştığından emin ol
- Browser console'da hata detaylarını kontrol et
- Backend loglarını kontrol et

### Dependency Conflict:
- Sanal ortam kullan (önerilen)
- Eski Python paketlerini temizle
- Python 3.8+ kullan

## 📊 Performans
- **İlk model yükleme**: 2-5 dakika
- **Analiz süresi**: 1-3 saniye
- **Desteklenen formatlar**: JPG, PNG, WebP, BMP
- **Max dosya boyutu**: 10MB
- **Önerilen görsel boyutu**: 1024x1024 ve altı

## 🔒 Güvenlik
- Bu bir MVP/demo uygulamasıdır
- Production kullanımı için ek güvenlik gereklidir
- Sonuçlar olasılık tahminidir, kesin değildir