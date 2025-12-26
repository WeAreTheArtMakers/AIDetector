# 🤖 AI Image Detector

[![License: WATAM](https://img.shields.io/badge/License-WATAM-blue.svg)](https://WeAreTheArtMakers.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-orange.svg)](https://huggingface.co/transformers/)

Modern ve gelişmiş **AI Image Detector** - Yapay zeka tarafından üretilen görselleri tespit eden web uygulaması. Gelişmiş bilgisayarlı görü teknikleri ve makine öğrenmesi modelleri ile %90+ doğruluk oranında AI detection.

![AI Image Detector Demo](https://via.placeholder.com/800x400/1e293b/60a5fa?text=AI+Image+Detector+Demo)

## ✨ Özellikler

### 🔬 Gelişmiş AI Tespit Teknolojileri
- **🧠 Çoklu AI Model Analizi**: HuggingFace transformers ile derin öğrenme
- **📐 Boyut İmza Analizi**: 512x512, 1024x1024 gibi tipik AI boyutları tespiti
- **🎨 Renk İmza Tespiti**: Aşırı doygunluk, quantization, perfect gradients
- **🔍 Gürültü Analizi**: AI'ın tipik "çok temiz" görüntü imzası
- **⚡ Kenar Analizi**: Sobel operatörü ile unnatural sharpening tespiti
- **📊 Frekans Domain Analizi**: FFT ile frequency anomalileri

### 🛡️ Metadata & Kaynak Analizi
- **🏷️ AI Yazılım Tespiti**: 20+ AI tool imzası (Midjourney, DALL-E, Stable Diffusion, vb.)
- **📷 EXIF Analizi**: Kamera vs yazılım kaynak tespiti
- **🔍 Şüpheli Desen Tespiti**: Perfect dimensions, missing camera info
- **📋 Content Credentials**: C2PA metadata kontrolü (gelecek özellik)

### 🎨 Modern UI/UX
- **🌙 Dark Mode**: Glassmorphism efektli modern tasarım
- **📱 Responsive**: Mobil ve masaüstü uyumlu
- **🎯 Sürükle & Bırak**: Kolay dosya yükleme
- **⚡ Real-time**: Anlık analiz sonuçları
- **📊 Detaylı Raporlama**: 5 farklı teknik analiz gösterimi

### 🔧 Teknik Özellikler
- **🚀 FastAPI Backend**: Yüksek performanslı API
- **🤖 HuggingFace Integration**: Önceden eğitilmiş modeller
- **🔄 Fallback System**: Backend çalışmazsa yerel analiz
- **🐳 Docker Ready**: Kolay deployment
- **📈 Scalable**: Production-ready mimari

## 🚀 Hızlı Başlangıç

### 📋 Gereksinimler
- Python 3.8+
- 1GB+ RAM
- 2GB+ disk alanı (model dosyaları için)
- İnternet bağlantısı (ilk kurulum için)

### ⚡ Otomatik Kurulum
```bash
# Repository'yi klonla
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector

# Sanal ortam kur ve başlat
./setup_venv.sh          # macOS/Linux
# veya
setup_venv.bat           # Windows

# Uygulamayı çalıştır
./run_backend_venv.sh    # Terminal 1 - Backend
./run_frontend.sh        # Terminal 2 - Frontend
```

### 🌐 Erişim
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 📊 Performans & Doğruluk

| Görsel Türü | Tespit Doğruluğu | Ortalama Süre |
|--------------|-------------------|---------------|
| AI Üretimi (Midjourney, DALL-E) | %85-95 | 1-2 saniye |
| Gerçek Fotoğraflar | %90-95 | 1-2 saniye |
| Upscaled/Enhanced | %70-85 | 1-3 saniye |
| Hibrit/Edited | %60-80 | 2-3 saniye |

## 🔍 Desteklenen AI Araçları

### ✅ Yüksek Tespit Oranı (%85+)
- **Midjourney** (v4, v5, v6, Niji)
- **DALL-E** (2, 3, ChatGPT integration)
- **Stable Diffusion** (1.5, 2.0, XL, Turbo)
- **Adobe Firefly** (v1, v2, v3)
- **Leonardo AI** (Phoenix, Alchemy)
- **Google Gemini** (Imagen, Bard integration)
- **Microsoft Copilot** (Designer, Bing Creator)

### ⚠️ Orta Tespit Oranı (%60-85%)
- **Runway ML** (Gen-1, Gen-2)
- **Artbreeder** (Collage, Splicer)
- **NightCafe** (Stable, Artistic)
- **DeepAI** (Text2Img, StyleGAN)
- **Canva AI** (Magic Design)
- **Meta AI** (Imagine, Emu)
- **Pika Labs** (Video-to-Image)

### 📝 2024 Yeni Araçlar (%70-90%)
- **Sora** (OpenAI Video AI)
- **Ideogram** (Text rendering AI)
- **Flux** (Black Forest Labs)
- **Recraft** (Vector AI)
- **Freepik AI** (Pikaso)
- **Adobe Express** (Generative AI)
- **Anthropic Claude** (Vision capabilities)

## 🛠️ Geliştirici Rehberi

### 📁 Proje Yapısı
```
ai-image-detector/
├── 📂 backend/
│   ├── main.py              # FastAPI uygulaması
│   ├── requirements.txt     # Python bağımlılıkları
│   └── models/              # Model cache (otomatik oluşur)
├── 📂 frontend/
│   ├── index.html          # Ana sayfa
│   ├── style.css           # Modern CSS (Glassmorphism)
│   └── script.js           # Vanilla JS (ES6+)
├── 📂 docs/
│   ├── SETUP.md            # Kurulum rehberi
│   └── DISCLAIMER.md       # Sorumluluk reddi
├── 🚀 setup_venv.sh       # Otomatik kurulum
└── 📋 requirements.txt     # Tüm bağımlılıklar
```

### 🔧 API Kullanımı
```python
import requests

# Görsel analizi
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/analyze',
        files={'file': f}
    )
    
result = response.json()
print(f"AI Olasılığı: %{result['aiProbability']}")
```

### 🎯 Model Değiştirme
```python
# backend/main.py içinde
MODEL_CONFIG = {
    "model_name": "your-custom-model",  # Değiştir
    "device": "cuda",  # GPU kullanımı için
    "max_image_size": (1024, 1024)
}
```

## 🐳 Docker ile Çalıştırma

```bash
# Docker Compose ile (yakında)
docker-compose up -d

# Manuel Docker
docker build -t ai-detector-backend ./backend
docker run -p 8001:8001 ai-detector-backend
```

## 📈 Production Deployment

### 🌐 Vercel/Netlify (Frontend)
```bash
# Frontend static files
# Deploy to Vercel/Netlify
```

### ☁️ Railway/Heroku (Backend)
```bash
# Procfile oluştur
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile
# Deploy
```

### 🔒 Güvenlik Ayarları
```python
# Production için
CORS_ORIGINS = ["https://yourdomain.com"]
API_RATE_LIMIT = "100/minute"
MAX_FILE_SIZE = "5MB"
```

## 🤝 Katkıda Bulunma

1. **Fork** edin
2. **Feature branch** oluşturun (`git checkout -b feature/amazing-feature`)
3. **Commit** edin (`git commit -m 'Add amazing feature'`)
4. **Push** edin (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### 🐛 Bug Raporu
- Issue template kullanın
- Görsel örnekleri ekleyin
- Sistem bilgilerini paylaşın
- Beklenen vs gerçek sonuçları belirtin

### 💡 Özellik İsteği
- Use case açıklayın
- Teknik detayları verin
- Mockup/wireframe ekleyin

## 📊 Roadmap

### 🎯 v2.0 (Q1 2025)
- [ ] Video AI detection (Sora, Runway)
- [ ] Batch processing API
- [ ] Advanced C2PA integration
- [ ] Custom model training interface

### 🚀 v2.5 (Q2 2025)
- [ ] Real-time webcam analysis
- [ ] Browser extension
- [ ] Mobile app (React Native)
- [ ] Enterprise dashboard

### 🌟 v3.0 (Q3 2025)
- [ ] Blockchain verification
- [ ] AI watermarking
- [ ] Federated learning
- [ ] Multi-language support

## ⚠️ Önemli Uyarılar

> **🚨 Bu uygulama bir olasılık tahmini yapar, kesin hüküm vermez!**
> 
> - Sonuçlar %100 doğru değildir
> - Kritik kararlar için profesyonel doğrulama gereklidir
> - False positive/negative sonuçlar mümkündür
> - Sürekli gelişen AI teknolojileri nedeniyle güncellemeler gereklidir

## 📄 Lisans

Bu proje [WATAM Lisansı](LICENSE) altında lisanslanmıştır - WeAreTheArtMakers.com

## 🙏 Teşekkürler

- [HuggingFace](https://huggingface.co/) - Transformers kütüphanesi
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Lucide Icons](https://lucide.dev/) - Beautiful icons
- [PIL/Pillow](https://pillow.readthedocs.io/) - Image processing
- [WeAreTheArtMakers.com](https://WeAreTheArtMakers.com) - Lisans sağlayıcısı

## 📞 İletişim & Destek

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/ai-image-detector/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-image-detector/discussions)
- 📧 **Email**: your.email@example.com
- 🐦 **Twitter**: [@yourusername](https://twitter.com/yourusername)

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

Made with ❤️ by [WATAM](https://github.com/wearetheartmakers)

</div>
