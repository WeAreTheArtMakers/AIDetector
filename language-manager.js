/**
 * Language Manager for AI Image Detector
 * Supports Turkish (tr) and English (en)
 */

const translations = {
    tr: {
        // Header
        "app_title": "AI Image Detector",
        "app_subtitle": "Kanıta Dayalı Adli Görsel Analizi • Multi-Model Ensemble",
        
        // Hero Section
        "hero_title": "Kanıta Dayalı AI Görsel Analizi",
        "hero_subtitle": "Multi-model ensemble, CLIP semantik analizi ve adli metadata doğrulaması ile görsellerin AI tarafından üretilip üretilmediğini tespit edin.",
        
        // Features
        "feature_ensemble": "Multi-Model Ensemble",
        "feature_ensemble_desc": "Birden fazla AI tespit modeli ile güvenilir sonuçlar ve belirsizlik ölçümü.",
        "feature_forensic": "Adli Metadata Analizi",
        "feature_forensic_desc": "EXIF, XMP, IPTC, GPS ve kamera bilgileri ile kapsamlı doğrulama.",
        "feature_content": "İçerik Türü Tespiti",
        "feature_content_desc": "Fotoğraf, CGI, illüstrasyon ve dijital sanat ayrımı.",
        
        // Stats
        "stat_models": "AI Modeli",
        "stat_metadata": "Metadata Alanı",
        "stat_semantic": "Semantik Analiz",
        "stat_gps": "Konum Doğrulama",
        
        // Upload
        "upload_title": "Görsel Yükle",
        "upload_hint": "Sürükle bırak veya tıkla",
        "select_file": "Dosya Seç",
        "file_types": "JPG, PNG, WebP • Max 10MB",
        "analyze_btn": "Forensic Analiz",
        "new_analysis": "Yeni Analiz",
        
        // Upload warning
        "upload_warning": "Orijinal dosyayı yükleyin. WhatsApp/Instagram yeniden sıkıştırır ve EXIF'i silebilir.",
        
        // File chain
        "file_chain_label": "Dosya Kaynağı",
        "chain_original": "Kamera / Orijinal",
        "chain_whatsapp": "WhatsApp",
        "chain_instagram": "Instagram",
        "chain_screenshot": "Ekran Görüntüsü",
        "chain_unknown": "Bilinmiyor",
        
        // Progress
        "connecting": "Bağlantı kuruluyor...",
        "extracting_metadata": "Metadata çıkarılıyor...",
        "running_ai": "AI modelleri çalışıyor...",
        "completed": "Tamamlandı!",
        
        // Results
        "ai_probability": "AI %",
        "confidence": "Güven",
        "processing_time": "Süre",
        "report_id": "Rapor ID",
        "evidence_level": "Kanıt Seviyesi",
        "evaluation": "Değerlendirme",
        
        // Evidence levels
        "DEFINITIVE_AI": "KESİN: AI Tarafından Üretilmiş",
        "DEFINITIVE_REAL": "KESİN: Doğrulanmış Gerçek Fotoğraf",
        "STRONG_AI_EVIDENCE": "GÜÇLÜ KANIT: AI Üretimi Muhtemel",
        "STRONG_REAL_EVIDENCE": "GÜÇLÜ KANIT: Gerçek Fotoğraf",
        "STATISTICAL_AI": "İSTATİSTİKSEL: AI Olabilir (Belirsiz)",
        "STATISTICAL_REAL": "İSTATİSTİKSEL: Gerçek Olabilir",
        "INCONCLUSIVE": "BELİRSİZ: Yeterli Kanıt Yok",
        
        // Level labels
        "level_definitive": "Kesin Kanıt",
        "level_strong": "Güçlü Kanıt",
        "level_statistical": "İstatistiksel",
        "level_inconclusive": "Belirsiz",
        "level_content_type": "İçerik Türü",
        
        // Content Type
        "content_type": "İçerik Türü",
        "content_type_photo": "Fotoğraf Benzeri",
        "content_type_non_photo": "CGI/İllüstrasyon",
        "content_type_composite": "Olası Kompozit/Düzenlenmiş",
        "content_type_uncertain": "Belirsiz",
        "content_type_high": "Yüksek Güven",
        "content_type_medium": "Orta Güven",
        "content_type_low": "Düşük Güven",
        "content_type_warning": "Bu görsel CGI/İllüstrasyon gibi görünüyor. AI olasılığı sadece bilgi amaçlıdır.",
        "top_matches": "En İyi Eşleşmeler",
        
        // Two-Axis Output
        "ai_likelihood": "AI Olasılığı",
        "evidential_quality": "Kanıt Kalitesi",
        "ai_likelihood_high": "Yüksek AI Olasılığı",
        "ai_likelihood_medium": "Orta AI Olasılığı",
        "ai_likelihood_low": "Düşük AI Olasılığı",
        "quality_high": "Yüksek Kalite Kanıt",
        "quality_medium": "Orta Kalite Kanıt",
        "quality_low": "Düşük Kalite Kanıt",
        "quality_score": "Kalite Skoru",
        "quality_factors": "Faktörler",
        "quality_warning": "Kanıt kalitesi düşük. Yeniden sıkıştırma, eksik metadata veya model uyuşmazlığı nedeniyle sonuçlar daha az güvenilir olabilir.",
        
        // Sections
        "model_scores": "Model Skorları (Ensemble)",
        "findings": "Bulgular",
        "gps_location": "GPS / Konum (EXIF)",
        "camera_info": "Kamera",
        "ai_software": "AI Yazılım",
        "domain_info": "Domain",
        "image_info": "Görsel Bilgileri",
        "jpeg_forensics": "JPEG Forensics",
        "manipulation_section": "Manipülasyon Tespiti",
        "metadata_section": "Metadata (EXIF/XMP/IPTC)",
        "uncertainty_section": "Belirsizlik Analizi",
        "ai_type_section": "AI Üretim Tipi",
        
        // GPS
        "gps_found": "GPS Konumu Bulundu",
        "gps_not_found": "GPS verisi bulunamadı",
        "latitude": "Enlem",
        "longitude": "Boylam",
        "altitude": "Yükseklik",
        "gps_time": "GPS Zamanı",
        "gps_warning": "EXIF GPS düzenlenebilir - kanıt değil, ipucu olarak değerlendirin",
        
        // Camera
        "camera_found": "Kamera Bilgisi",
        "camera_not_found": "Kamera Bilgisi Yok",
        "camera_make": "Marka",
        "camera_model": "Model",
        "lens": "Lens",
        "exposure": "Pozlama",
        "aperture": "Diyafram",
        "iso": "ISO",
        "focal_length": "Odak",
        
        // AI Detection
        "ai_detected": "AI Yazılım Tespit Edildi!",
        "ai_not_detected": "AI yazılım imzası bulunamadı",
        "software": "Yazılım",
        "vendor": "Üretici",
        "type": "Tür",
        
        // Domain
        "domain_original": "Orijinal",
        "domain_screenshot": "Ekran Görüntüsü",
        "domain_recompressed": "Yeniden Sıkıştırılmış",
        "domain_modified": "Değiştirilmiş",
        "domain_unknown": "Bilinmiyor",
        
        // Image info
        "dimensions": "Boyut",
        "format": "Format",
        "file_size": "Dosya",
        "color_mode": "Mod",
        
        // Findings
        "definitive_findings": "Kesin Bulgular",
        "strong_evidence": "Güçlü Kanıtlar",
        "authenticity_indicators": "Gerçeklik Göstergeleri",
        "technical_indicators": "Teknik Göstergeler",
        "no_findings": "Belirgin bulgu yok",
        "suggests_real": "→ Gerçek",
        "suggests_ai": "→ AI",
        
        // Uncertainty
        "confidence_interval": "Güven Aralığı",
        "model_disagreement": "Model Uyuşmazlığı",
        "expected_error": "Beklenen Hata Oranı",
        "ece": "ECE (Kalibrasyon Hatası)",
        
        // AI Type
        "ai_type_guess": "Tahmin",
        "text_to_image": "Text-to-Image",
        "img2img": "Image-to-Image",
        "inpainting": "Inpainting",
        "unknown": "Bilinmiyor",
        "ai_type_disclaimer": "Bu heuristik bir tahmindir, kesin kanıt değildir.",
        
        // JPEG Forensics
        "jpeg_quality": "Tahmini Kalite",
        "subsampling": "Alt Örnekleme",
        "double_compression": "Çift Sıkıştırma Olasılığı",
        "quant_fingerprint": "Quantization Parmak İzi",
        
        // Warnings
        "warning_not_definitive": "⚠️ Bu sonuç kesin değildir. İstatistiksel tahminler hata payı içerir. Kesin sonuç için AI yazılım imzası veya C2PA Content Credentials gereklidir.",
        "warning_statistical": "⚠️ İstatistiksel skorlar kanıt değildir, olasılık tahminidir.",
        
        // Visualization / Heatmap
        "visualization_section": "Görselleştirme",
        "show_overlay": "Analiz Katmanını Göster",
        "hide_overlay": "Analiz Katmanını Gizle",
        "global_intensity_overlay": "Global İşleme Yoğunluğu Katmanı",
        "global_intensity_note": "Bu katman global işleme yoğunluğunu gösterir (yapıştırma kanıtı değil)",
        "local_suspicion_overlay": "Şüpheli Bölgeler Katmanı",
        "local_suspicion_note": "Bu katman sınır doğrulamalı şüpheli bölgeleri gösterir",
        "visualization_legend": "Renk Açıklaması",
        "low_intensity": "Düşük yoğunluk",
        "medium_intensity": "Orta yoğunluk",
        "high_intensity": "Yüksek yoğunluk",
        "low_suspicion": "Düşük şüphe",
        "medium_suspicion": "Orta şüphe",
        "high_suspicion": "Yüksek şüphe (sınır doğrulamalı)",
        "recompression_warning": "Sosyal medya yeniden sıkıştırması artefaktlar oluşturabilir",
        "no_visualization": "Görselleştirme için yeterli düzenleme tespit edilmedi",
        
        // Edit Assessment
        "edit_type": "Düzenleme Türü",
        "none_detected": "Düzenleme Tespit Edilmedi",
        "global_postprocess": "Global Düzenleme (Filtre/Renk)",
        "local_manipulation": "Yerel Manipülasyon Tespit Edildi",
        "generator_artifacts": "AI Üretim Artefaktları",
        "boundary_corroboration": "Sınır Doğrulaması",
        "global_edit_score": "Global Düzenleme Skoru",
        "local_manipulation_score": "Yerel Manipülasyon Skoru",
        "generator_artifacts_score": "Üretici Artefakt Skoru",
        "method_scores": "Yöntem Skorları (Teknik)",
        "noise_inconsistency": "Gürültü Tutarsızlığı",
        "blur_mismatch": "Bulanıklık Uyumsuzluğu",
        "copy_move": "Kopyala-Yapıştır",
        "edge_matte": "Kenar Halo",
        "suspicious_regions": "Şüpheli Bölgeler",
        "assessment_reasons": "Değerlendirme Nedenleri",
        "region_suppressed_global": "Bölge gösterimi bastırıldı (global işlemeden kaynaklanan artifaktlar, yerel manipülasyon değil)",
        "region_suppressed_ai": "Bölge gösterimi bastırıldı (AI üretim artefaktları, manipülasyon değil)",
        "global_edit_explanation": "Global düzenlemeler tespit edildi (renk düzeltme, filtreler, platform işleme). Bu yerel manipülasyon/yapıştırma DEĞİLDİR.",
        "local_manipulation_explanation": "Yerel manipülasyon kanıtı tespit edildi (sınır doğrulamalı)",
        "generator_artifacts_explanation": "Doku/gürültü tutarsızlıkları AI üretim sürecinden kaynaklanıyor, görsel manipülasyonu veya yapıştırmadan değil.",
        
        // Pathway
        "pathway_section": "Üretim Yolu",
        "real_photo": "Gerçek Fotoğraf",
        "t2i": "Text-to-Image (T2I)",
        "i2i": "Image-to-Image (I2I)",
        "inpainting": "Inpainting",
        "stylization": "Stilizasyon",
        "generator_family": "Üretici Ailesi",
        "diffusion_fingerprint": "Diffusion İzi",
        "i2i_evidence": "I2I Kanıtı",
        "camera_evidence": "Kamera Kanıtı",
        "detection_signals": "Tespit Sinyalleri",
        
        // More details
        "more_details": "Detayları Göster",
        "hide_details": "Detayları Gizle",
        
        // Errors
        "invalid_file": "Lütfen geçerli bir görsel dosyası seçin.",
        "file_too_large": "Dosya boyutu 10MB'dan küçük olmalıdır.",
        "analysis_error": "Analiz hatası"
    },
    
    en: {
        // Header
        "app_title": "AI Image Detector",
        "app_subtitle": "Evidence-Based Forensic Image Analysis • Multi-Model Ensemble",
        
        // Hero Section
        "hero_title": "Evidence-Based AI Image Analysis",
        "hero_subtitle": "Detect whether images are AI-generated using multi-model ensemble, CLIP semantic analysis, and forensic metadata verification.",
        
        // Features
        "feature_ensemble": "Multi-Model Ensemble",
        "feature_ensemble_desc": "Reliable results with multiple AI detection models and uncertainty measurement.",
        "feature_forensic": "Forensic Metadata Analysis",
        "feature_forensic_desc": "Comprehensive verification with EXIF, XMP, IPTC, GPS and camera data.",
        "feature_content": "Content Type Detection",
        "feature_content_desc": "Distinguish between photos, CGI, illustrations and digital art.",
        
        // Stats
        "stat_models": "AI Models",
        "stat_metadata": "Metadata Fields",
        "stat_semantic": "Semantic Analysis",
        "stat_gps": "Location Verification",
        
        // Upload
        "upload_title": "Upload Image",
        "upload_hint": "Drag & drop or click",
        "select_file": "Select File",
        "file_types": "JPG, PNG, WebP • Max 10MB",
        "analyze_btn": "Forensic Analysis",
        "new_analysis": "New Analysis",
        
        // Upload warning
        "upload_warning": "Upload the original file. WhatsApp/Instagram may recompress and strip EXIF.",
        
        // File chain
        "file_chain_label": "File Source",
        "chain_original": "Camera / Original",
        "chain_whatsapp": "WhatsApp",
        "chain_instagram": "Instagram",
        "chain_screenshot": "Screenshot",
        "chain_unknown": "Unknown",
        
        // Progress
        "connecting": "Connecting...",
        "extracting_metadata": "Extracting metadata...",
        "running_ai": "Running AI models...",
        "completed": "Completed!",
        
        // Results
        "ai_probability": "AI %",
        "confidence": "Confidence",
        "processing_time": "Time",
        "report_id": "Report ID",
        "evidence_level": "Evidence Level",
        "evaluation": "Evaluation",
        
        // Evidence levels
        "DEFINITIVE_AI": "DEFINITIVE: AI Generated",
        "DEFINITIVE_REAL": "DEFINITIVE: Verified Real Photo",
        "STRONG_AI_EVIDENCE": "STRONG EVIDENCE: Likely AI Generated",
        "STRONG_REAL_EVIDENCE": "STRONG EVIDENCE: Likely Real Photo",
        "STATISTICAL_AI": "STATISTICAL: Possibly AI (Uncertain)",
        "STATISTICAL_REAL": "STATISTICAL: Possibly Real",
        "INCONCLUSIVE": "INCONCLUSIVE: Insufficient Evidence",
        
        // Level labels
        "level_definitive": "Definitive",
        "level_strong": "Strong Evidence",
        "level_statistical": "Statistical",
        "level_inconclusive": "Inconclusive",
        "level_content_type": "Content Type",
        
        // Content Type
        "content_type": "Content Type",
        "content_type_photo": "Photo-like",
        "content_type_non_photo": "CGI/Illustration",
        "content_type_composite": "Possible Composite/Edited",
        "content_type_uncertain": "Uncertain",
        "content_type_high": "High Confidence",
        "content_type_medium": "Medium Confidence",
        "content_type_low": "Low Confidence",
        "content_type_warning": "This image appears to be CGI/Illustration. AI probability is informational only.",
        "top_matches": "Top Matches",
        
        // Two-Axis Output
        "ai_likelihood": "AI Likelihood",
        "evidential_quality": "Evidential Quality",
        "ai_likelihood_high": "High AI Likelihood",
        "ai_likelihood_medium": "Medium AI Likelihood",
        "ai_likelihood_low": "Low AI Likelihood",
        "quality_high": "High Quality Evidence",
        "quality_medium": "Medium Quality Evidence",
        "quality_low": "Low Quality Evidence",
        "quality_score": "Quality Score",
        "quality_factors": "Factors",
        "quality_warning": "Evidence quality is low. Results may be less reliable due to recompression, missing metadata, or model disagreement.",
        
        // Sections
        "model_scores": "Model Scores (Ensemble)",
        "findings": "Findings",
        "gps_location": "GPS / Location (EXIF)",
        "camera_info": "Camera",
        "ai_software": "AI Software",
        "domain_info": "Domain",
        "image_info": "Image Info",
        "jpeg_forensics": "JPEG Forensics",
        "manipulation_section": "Manipulation Detection",
        "metadata_section": "Metadata (EXIF/XMP/IPTC)",
        "uncertainty_section": "Uncertainty Analysis",
        "ai_type_section": "AI Generation Type",
        
        // GPS
        "gps_found": "GPS Location Found",
        "gps_not_found": "No GPS data found",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "altitude": "Altitude",
        "gps_time": "GPS Time",
        "gps_warning": "EXIF GPS can be edited - treat as clue, not proof",
        
        // Camera
        "camera_found": "Camera Info",
        "camera_not_found": "No Camera Info",
        "camera_make": "Make",
        "camera_model": "Model",
        "lens": "Lens",
        "exposure": "Exposure",
        "aperture": "Aperture",
        "iso": "ISO",
        "focal_length": "Focal Length",
        
        // AI Detection
        "ai_detected": "AI Software Detected!",
        "ai_not_detected": "No AI software signature found",
        "software": "Software",
        "vendor": "Vendor",
        "type": "Type",
        
        // Domain
        "domain_original": "Original",
        "domain_screenshot": "Screenshot",
        "domain_recompressed": "Recompressed",
        "domain_modified": "Modified",
        "domain_unknown": "Unknown",
        
        // Image info
        "dimensions": "Dimensions",
        "format": "Format",
        "file_size": "File Size",
        "color_mode": "Mode",
        
        // Findings
        "definitive_findings": "Definitive Findings",
        "strong_evidence": "Strong Evidence",
        "authenticity_indicators": "Authenticity Indicators",
        "technical_indicators": "Technical Indicators",
        "no_findings": "No significant findings",
        "suggests_real": "→ Real",
        "suggests_ai": "→ AI",
        
        // Uncertainty
        "confidence_interval": "Confidence Interval",
        "model_disagreement": "Model Disagreement",
        "expected_error": "Expected Error Rate",
        "ece": "ECE (Calibration Error)",
        
        // AI Type
        "ai_type_guess": "Guess",
        "text_to_image": "Text-to-Image",
        "img2img": "Image-to-Image",
        "inpainting": "Inpainting",
        "unknown": "Unknown",
        "ai_type_disclaimer": "This is a heuristic guess, not definitive proof.",
        
        // JPEG Forensics
        "jpeg_quality": "Estimated Quality",
        "subsampling": "Subsampling",
        "double_compression": "Double Compression Probability",
        "quant_fingerprint": "Quantization Fingerprint",
        
        // Warnings
        "warning_not_definitive": "⚠️ This result is not definitive. Statistical predictions have error margins. Definitive results require AI software signature or C2PA Content Credentials.",
        "warning_statistical": "⚠️ Statistical scores are probability estimates, not proof.",
        
        // Visualization / Heatmap
        "visualization_section": "Visualization",
        "show_overlay": "Show Analysis Overlay",
        "hide_overlay": "Hide Analysis Overlay",
        "global_intensity_overlay": "Global Processing Intensity Overlay",
        "global_intensity_note": "This overlay shows global processing intensity (not proof of splice)",
        "local_suspicion_overlay": "Suspicious Regions Overlay",
        "local_suspicion_note": "This overlay shows boundary-corroborated suspicious regions",
        "visualization_legend": "Color Legend",
        "low_intensity": "Low intensity",
        "medium_intensity": "Medium intensity",
        "high_intensity": "High intensity",
        "low_suspicion": "Low suspicion",
        "medium_suspicion": "Medium suspicion",
        "high_suspicion": "High suspicion (boundary-corroborated)",
        "recompression_warning": "Social-media recompression can create artifacts",
        "no_visualization": "No significant edits detected for visualization",
        
        // Edit Assessment
        "edit_type": "Edit Type",
        "none_detected": "No Edit Detected",
        "global_postprocess": "Global Edit (Filter/Color)",
        "local_manipulation": "Local Manipulation Detected",
        "generator_artifacts": "AI Generation Artifacts",
        "boundary_corroboration": "Boundary Corroboration",
        "global_edit_score": "Global Edit Score",
        "local_manipulation_score": "Local Manipulation Score",
        "generator_artifacts_score": "Generator Artifacts Score",
        "method_scores": "Method Scores (Technical)",
        "noise_inconsistency": "Noise Inconsistency",
        "blur_mismatch": "Blur Mismatch",
        "copy_move": "Copy-Move",
        "edge_matte": "Edge Matte",
        "suspicious_regions": "Suspicious Regions",
        "assessment_reasons": "Assessment Reasons",
        "region_suppressed_global": "Region display suppressed (artifacts from global processing, not local manipulation)",
        "region_suppressed_ai": "Region display suppressed (AI generation artifacts, not manipulation)",
        "global_edit_explanation": "Global adjustments detected (color grading, filters, platform processing). This is NOT local manipulation/splice.",
        "local_manipulation_explanation": "Local manipulation evidence detected (boundary-corroborated)",
        "generator_artifacts_explanation": "Texture/noise inconsistencies are from AI generation process, not from image manipulation or splicing.",
        
        // Pathway
        "pathway_section": "Generation Pathway",
        "real_photo": "Real Photo",
        "t2i": "Text-to-Image (T2I)",
        "i2i": "Image-to-Image (I2I)",
        "inpainting": "Inpainting",
        "stylization": "Stylization",
        "generator_family": "Generator Family",
        "diffusion_fingerprint": "Diffusion Fingerprint",
        "i2i_evidence": "I2I Evidence",
        "camera_evidence": "Camera Evidence",
        "detection_signals": "Detection Signals",
        
        // More details
        "more_details": "Show Details",
        "hide_details": "Hide Details",
        
        // Errors
        "invalid_file": "Please select a valid image file.",
        "file_too_large": "File size must be less than 10MB.",
        "analysis_error": "Analysis error"
    }
};

class LanguageManager {
    constructor() {
        this.currentLang = localStorage.getItem('ai_detector_lang') || 'tr';
        this.listeners = [];
    }
    
    get(key) {
        return translations[this.currentLang]?.[key] || translations['en']?.[key] || key;
    }
    
    setLanguage(lang) {
        if (translations[lang]) {
            this.currentLang = lang;
            localStorage.setItem('ai_detector_lang', lang);
            this.notifyListeners();
            this.updateUI();
        }
    }
    
    toggleLanguage() {
        const newLang = this.currentLang === 'tr' ? 'en' : 'tr';
        this.setLanguage(newLang);
    }
    
    getCurrentLanguage() {
        return this.currentLang;
    }
    
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    notifyListeners() {
        this.listeners.forEach(cb => cb(this.currentLang));
    }
    
    updateUI() {
        // Update all elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = this.get(key);
        });
        
        // Update placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.get(key);
        });
        
        // Update titles
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = this.get(key);
        });
    }
    
    // Format verdict based on evidence level
    formatVerdict(evidenceLevel) {
        return this.get(evidenceLevel) || this.get('INCONCLUSIVE');
    }
    
    // Format evidence level label
    formatEvidenceLevel(level) {
        if (level.includes('DEFINITIVE')) return this.get('level_definitive');
        if (level.includes('STRONG')) return this.get('level_strong');
        if (level.includes('STATISTICAL')) return this.get('level_statistical');
        return this.get('level_inconclusive');
    }
    
    // Format domain type
    formatDomain(type) {
        const map = {
            'original': 'domain_original',
            'screenshot': 'domain_screenshot',
            'recompressed': 'domain_recompressed',
            'modified': 'domain_modified',
            'unknown': 'domain_unknown'
        };
        return this.get(map[type] || 'domain_unknown');
    }
    
    // Format AI type
    formatAIType(type) {
        const map = {
            'text_to_image': 'text_to_image',
            'img2img': 'img2img',
            'inpainting': 'inpainting',
            'unknown': 'unknown'
        };
        return this.get(map[type] || 'unknown');
    }
}

// Global instance
window.i18n = new LanguageManager();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.i18n.updateUI();
});
