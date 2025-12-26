# 🤝 Katkıda Bulunma Rehberi

AI Image Detector projesine katkıda bulunmak istediğiniz için teşekkürler! Bu rehber, projeye nasıl katkıda bulunabileceğinizi açıklar.

## 🚀 Başlangıç

### 1. Repository'yi Fork Edin
```bash
# GitHub'da "Fork" butonuna tıklayın
# Sonra kendi fork'unuzu klonlayın
git clone https://github.com/YOURUSERNAME/ai-image-detector.git
cd ai-image-detector
```

### 2. Geliştirme Ortamını Kurun
```bash
# Sanal ortam oluşturun
./setup_venv.sh  # macOS/Linux
# veya
setup_venv.bat   # Windows

# Uygulamayı test edin
./run_backend_venv.sh    # Terminal 1
./run_frontend.sh        # Terminal 2
```

## 🐛 Bug Raporu

### Bug Raporu Şablonu
```markdown
**Bug Açıklaması**
Kısa ve net bug açıklaması.

**Yeniden Üretme Adımları**
1. '...' sayfasına git
2. '....' butonuna tıkla
3. '....' alanını doldur
4. Hatayı gör

**Beklenen Davranış**
Ne olmasını bekliyordunuz?

**Gerçek Davranış**
Ne oldu?

**Ekran Görüntüleri**
Varsa ekran görüntüleri ekleyin.

**Sistem Bilgileri**
- OS: [e.g. macOS 14.0]
- Browser: [e.g. Chrome 120]
- Python: [e.g. 3.11]
- Backend Status: [çalışıyor/çalışmıyor]

**Test Edilen Görsel**
- Görsel türü: [AI/Gerçek/Belirsiz]
- Dosya formatı: [JPG/PNG/WebP]
- Dosya boyutu: [MB]
- Beklenen sonuç: [%XX AI olasılığı]
- Gerçek sonuç: [%XX AI olasılığı]
```

## 💡 Özellik İsteği

### Özellik İsteği Şablonu
```markdown
**Özellik Açıklaması**
Yeni özelliğin kısa açıklaması.

**Problem/İhtiyaç**
Bu özellik hangi problemi çözüyor?

**Önerilen Çözüm**
Nasıl çözülmesini öneriyorsunuz?

**Alternatifler**
Başka hangi çözümler düşündünüz?

**Teknik Detaylar**
- Frontend/Backend/Both
- Yeni bağımlılık gerekiyor mu?
- API değişikliği gerekiyor mu?

**Mockup/Wireframe**
Varsa tasarım örnekleri ekleyin.
```

## 🔧 Kod Katkısı

### 1. Branch Oluşturun
```bash
# Ana branch'ten yeni branch oluşturun
git checkout -b feature/amazing-feature
# veya
git checkout -b bugfix/fix-detection-issue
```

### 2. Kod Standartları

#### Python (Backend)
```python
# PEP 8 standartlarını takip edin
# Type hints kullanın
def analyze_image(image: Image.Image) -> Dict[str, Any]:
    """
    Analyze image for AI detection.
    
    Args:
        image: PIL Image object
        
    Returns:
        Analysis results dictionary
    """
    pass

# Docstring'leri ekleyin
# Error handling yapın
# Logging kullanın
```

#### JavaScript (Frontend)
```javascript
// ES6+ syntax kullanın
// JSDoc comments ekleyin
/**
 * Calculate AI probability from analysis results
 * @param {Object} analyses - Analysis results
 * @returns {number} AI probability (0-100)
 */
function calculateAIProbability(analyses) {
    // Clear variable names
    // Consistent formatting
    // Error handling
}
```

### 3. Test Edin
```bash
# Backend testleri
cd backend
python -m pytest tests/

# Frontend testleri (manuel)
# Farklı görsel türleri test edin:
# - AI üretimi görseller
# - Gerçek fotoğraflar
# - Edge cases (çok küçük/büyük dosyalar)

# API testleri
curl -X POST http://localhost:8001/analyze \
  -F "file=@test_image.jpg"
```

### 4. Commit Mesajları
```bash
# Conventional Commits formatı kullanın
git commit -m "feat: add video AI detection support"
git commit -m "fix: resolve CORS issue in production"
git commit -m "docs: update API documentation"
git commit -m "style: improve mobile responsive design"
git commit -m "refactor: optimize noise analysis algorithm"
git commit -m "test: add unit tests for color analysis"
```

### 5. Pull Request
```bash
# Değişikliklerinizi push edin
git push origin feature/amazing-feature

# GitHub'da Pull Request oluşturun
# PR template'i doldurun
```

## 📋 Pull Request Şablonu

```markdown
## 📝 Değişiklik Açıklaması
Bu PR'da neler değişti?

## 🎯 İlgili Issue
Fixes #123

## 🧪 Test Edildi
- [ ] Backend testleri geçiyor
- [ ] Frontend manuel testleri yapıldı
- [ ] Farklı görsel türleri test edildi
- [ ] Mobile responsive kontrol edildi

## 📸 Ekran Görüntüleri
Varsa UI değişikliklerinin ekran görüntüleri.

## ✅ Checklist
- [ ] Kod standartlarına uygun
- [ ] Dokümantasyon güncellendi
- [ ] Tests eklendi/güncellendi
- [ ] Breaking change yok
- [ ] Commit mesajları düzgün
```

## 🏗️ Geliştirme Alanları

### 🔥 Yüksek Öncelik
- [ ] Video AI detection
- [ ] Batch processing
- [ ] Performance optimizations
- [ ] Mobile app

### 🚀 Orta Öncelik
- [ ] Custom model training
- [ ] Advanced metadata analysis
- [ ] Real-time webcam analysis
- [ ] Browser extension

### 💡 Düşük Öncelik
- [ ] Multi-language support
- [ ] Dark/Light theme toggle
- [ ] Export/Import settings
- [ ] Analytics dashboard

## 🎨 UI/UX Katkıları

### Tasarım Prensipleri
- **Minimalist**: Sade ve temiz tasarım
- **Accessible**: Erişilebilir renkler ve fontlar
- **Responsive**: Tüm cihazlarda çalışmalı
- **Fast**: Hızlı yükleme ve etkileşim

### Figma/Design Files
- Tasarım dosyalarını `designs/` klasörüne ekleyin
- Component library'yi güncel tutun
- Design system'e uygun olun

## 📚 Dokümantasyon

### Güncellenmesi Gerekenler
- API dokümantasyonu
- Kurulum rehberi
- Troubleshooting guide
- Performance benchmarks

### Yazım Kuralları
- Türkçe: Teknik terimler İngilizce kalabilir
- İngilizce: README ve kod yorumları
- Markdown formatı kullanın
- Kod örnekleri ekleyin

## 🤔 Sorularınız mı Var?

- 💬 [GitHub Discussions](https://github.com/yourusername/ai-image-detector/discussions)
- 📧 Email: your.email@example.com
- 🐦 Twitter: [@yourusername](https://twitter.com/yourusername)

## 🙏 Teşekkürler

Katkılarınız için teşekkür ederiz! Her katkı, projeyi daha iyi hale getirir.

---

**Happy Coding! 🚀**