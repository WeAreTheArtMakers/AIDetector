/**
 * AI Image Detector - Forensic Analysis Frontend v6.0
 * Multi-model ensemble with uncertainty, JPEG forensics, AI type classification
 * Integrated with language manager (i18n)
 */

class AIImageDetector {
    constructor() {
        this.currentFile = null;
        this.isAnalyzing = false;
        this.backendUrl = 'http://localhost:8000';
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.initIcons();
        this.initLanguageToggle();
        this.initAccordions();
    }

    bindElements() {
        this.uploadZone = document.getElementById('uploadZone');
        this.uploadContent = document.getElementById('uploadContent');
        this.previewContent = document.getElementById('previewContent');
        this.previewImg = document.getElementById('previewImg');
        this.fileName = document.getElementById('fileName');
        this.fileInput = document.getElementById('fileInput');
        this.selectBtn = document.getElementById('selectBtn');
        this.removeBtn = document.getElementById('removeBtn');
        this.analyzeSection = document.getElementById('analyzeSection');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressSection = document.getElementById('progressSection');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.resultsSection = document.getElementById('resultsSection');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
        this.fileChainSection = document.getElementById('fileChainSection');
        this.fileChainSelect = document.getElementById('fileChainSelect');
    }

    bindEvents() {
        this.selectBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.fileInput.click();
        });

        this.uploadZone?.addEventListener('click', () => {
            if (!this.currentFile) this.fileInput.click();
        });
        
        this.fileInput?.addEventListener('change', (e) => {
            if (e.target.files.length) this.processFile(e.target.files[0]);
        });
        
        ['dragover', 'dragleave', 'drop'].forEach(event => {
            this.uploadZone?.addEventListener(event, (e) => {
                e.preventDefault();
                if (event === 'dragover') this.uploadZone.classList.add('drag-over');
                else this.uploadZone.classList.remove('drag-over');
                if (event === 'drop' && e.dataTransfer.files.length) {
                    this.processFile(e.dataTransfer.files[0]);
                }
            });
        });
        
        this.removeBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.resetUpload();
        });
        
        this.analyzeBtn?.addEventListener('click', () => this.startAnalysis());
        this.newAnalysisBtn?.addEventListener('click', () => this.resetAll());
    }

    initIcons() {
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }

    initLanguageToggle() {
        const langTR = document.getElementById('langTR');
        const langEN = document.getElementById('langEN');
        
        const updateToggleUI = (lang) => {
            langTR?.classList.toggle('active', lang === 'tr');
            langEN?.classList.toggle('active', lang === 'en');
        };
        
        langTR?.addEventListener('click', () => {
            window.i18n?.setLanguage('tr');
            updateToggleUI('tr');
        });
        
        langEN?.addEventListener('click', () => {
            window.i18n?.setLanguage('en');
            updateToggleUI('en');
        });
        
        // Set initial state
        updateToggleUI(window.i18n?.getCurrentLanguage() || 'tr');
    }

    initAccordions() {
        document.querySelectorAll('.accordion-toggle').forEach(toggle => {
            toggle.addEventListener('click', () => {
                const targetId = toggle.getAttribute('data-target');
                const content = document.getElementById(targetId);
                if (content) {
                    content.classList.toggle('open');
                    toggle.classList.toggle('open');
                }
            });
        });
    }

    t(key) {
        return window.i18n?.get(key) || key;
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert(this.t('invalid_file') || 'Lütfen geçerli bir görsel dosyası seçin.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert(this.t('file_too_large') || 'Dosya boyutu 10MB\'dan küçük olmalıdır.');
            return;
        }
        
        this.currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.fileName.textContent = file.name;
            this.uploadContent.classList.add('hidden');
            this.previewContent.classList.remove('hidden');
            this.analyzeSection.classList.remove('hidden');
            this.fileChainSection?.classList.remove('hidden');
            this.resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetUpload() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.uploadContent.classList.remove('hidden');
        this.previewContent.classList.add('hidden');
        this.analyzeSection.classList.add('hidden');
        this.fileChainSection?.classList.add('hidden');
    }

    resetAll() {
        this.resetUpload();
        this.resultsSection.classList.add('hidden');
        this.progressSection.classList.add('hidden');
        
        // Show hero section again
        const heroSection = document.getElementById('heroSection');
        if (heroSection) heroSection.classList.remove('hidden');
    }

    async startAnalysis() {
        if (!this.currentFile || this.isAnalyzing) return;
        
        this.isAnalyzing = true;
        this.analyzeBtn.disabled = true;
        this.progressSection.classList.remove('hidden');
        this.resultsSection.classList.add('hidden');
        
        try {
            await this.updateProgress(10, this.t('connecting'));
            
            const formData = new FormData();
            formData.append('file', this.currentFile);
            
            const fileChain = this.fileChainSelect?.value || 'unknown';
            
            await this.updateProgress(20, this.t('extracting_metadata'));
            
            const response = await fetch(`${this.backendUrl}/analyze?file_chain=${fileChain}`, {
                method: 'POST',
                body: formData
            });
            
            await this.updateProgress(60, this.t('running_ai'));
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            await this.updateProgress(100, this.t('completed'));
            
            await new Promise(r => setTimeout(r, 300));
            this.displayResults(result);
            
        } catch (error) {
            console.error('Analysis error:', error);
            alert(`Analiz hatası: ${error.message}`);
        }
        
        this.isAnalyzing = false;
        this.analyzeBtn.disabled = false;
        this.progressSection.classList.add('hidden');
    }

    async updateProgress(percent, text) {
        this.progressBar.style.width = `${percent}%`;
        this.progressText.textContent = text;
        await new Promise(r => setTimeout(r, 50));
    }

    displayResults(data) {
        // Hide hero section when showing results
        const heroSection = document.getElementById('heroSection');
        if (heroSection) heroSection.classList.add('hidden');
        
        this.resultsSection.classList.remove('hidden');
        
        // Main score
        this.animateScore(data.ai_probability || 50);
        
        // Verdict - use new verdict_text layer if available
        if (data.verdict_text) {
            this.displayVerdictText(data.verdict_text, data.ai_probability_informational);
        } else {
            this.updateVerdict(data.verdict, data.evidence_level, data.ai_probability_informational);
        }
        
        // Two-Axis Summary (from verdict_text or summary_axes)
        const summaryAxes = data.verdict_text?.summary_axes || data.summary_axes;
        this.displaySummaryAxes(summaryAxes, data.ai_probability_informational);
        
        // Content Type
        this.displayContentType(data.content_type, data.ai_probability_informational);
        
        // Stats
        document.getElementById('confidenceValue').textContent = `${data.confidence || 0}%`;
        document.getElementById('processingTime').textContent = `${data.processing_time || 0}s`;
        document.getElementById('reportId').textContent = data.report_id || '-';
        
        // Uncertainty
        this.displayUncertainty(data.uncertainty || {});
        
        // Recommendation
        document.getElementById('recommendation').textContent = data.recommendation || '';
        
        // Warning/Banner - use verdict_text banner if available
        this.displayWarningBanner(data);
        
        // Global Footer
        this.displayGlobalFooter(data.verdict_text);
        
        // Model scores
        this.displayModelScores(data.models || []);
        
        // Findings
        this.displayFindings(data);
        
        // GPS
        this.displayGPS(data.gps || {});
        
        // Camera
        this.displayCamera(data.camera || {}, data.metadata || {});
        
        // AI Detection
        this.displayAIDetection(data.ai_detection || {});
        
        // Domain
        this.displayDomain(data.domain || {});
        
        // Image info
        this.displayImageInfo(data.image_info || {});
        
        // New sections
        this.displayJPEGForensics(data.jpeg_forensics);
        this.displayAIType(data.ai_generation_type);
        this.displayPathway(data.pathway, data.diffusion_fingerprint);
        this.displayStatistics(data.statistics);
        this.displayExtendedMetadata(data.metadata_extended);
        this.displayManipulation(data.manipulation, data.localization, data.edit_assessment, data.visualization);
        
        this.initIcons();
    }

    displayVisualization(visualization, editAssessment) {
        const container = document.getElementById('visualizationContainer');
        if (!container) return;
        
        if (!visualization || visualization.mode === 'none') {
            container.innerHTML = `<p class="text-slate-400 text-sm">${this.t('no_visualization')}</p>`;
            return;
        }
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const mode = visualization.mode;
        const heatmap = visualization.heatmap;
        const regions = visualization.regions || [];
        const legend = visualization.legend || [];
        const notes = visualization.notes || [];
        
        // Mode labels
        const modeLabels = {
            'global_edit': { 
                tr: 'Global İşleme Yoğunluğu', 
                en: 'Global Processing Intensity',
                color: 'blue',
                icon: 'sliders'
            },
            'local_manipulation': { 
                tr: 'Şüpheli Bölgeler', 
                en: 'Suspicious Regions',
                color: 'red',
                icon: 'alert-triangle'
            }
        };
        
        const modeInfo = modeLabels[mode] || modeLabels.global_edit;
        
        let html = `
            <div class="space-y-4">
                <!-- Mode Header -->
                <div class="bg-${modeInfo.color}-500/10 border border-${modeInfo.color}-500/30 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="${modeInfo.icon}" class="w-4 h-4 text-${modeInfo.color}-400"></i>
                        <span class="font-semibold text-${modeInfo.color}-400">
                            ${lang === 'en' ? modeInfo.en : modeInfo.tr}
                        </span>
                    </div>
                    ${notes.length > 0 ? `
                    <p class="text-xs text-slate-400">${notes[0]}</p>
                    ` : ''}
                </div>
        `;
        
        // Overlay Toggle and Image
        if (heatmap && heatmap.overlay_base64) {
            html += `
                <div class="space-y-2">
                    <button id="toggleOverlayBtn" 
                            class="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-slate-200 flex items-center justify-center gap-2 transition-colors">
                        <i data-lucide="layers" class="w-4 h-4"></i>
                        <span id="toggleOverlayText">${this.t('show_overlay')}</span>
                    </button>
                    
                    <div id="overlayImageContainer" class="hidden relative rounded-lg overflow-hidden">
                        <img id="overlayImage" 
                             src="data:image/png;base64,${heatmap.overlay_base64}" 
                             alt="Analysis Overlay"
                             class="w-full h-auto rounded-lg" />
                        <div class="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs text-slate-300">
                            ${heatmap.type === 'global_intensity' 
                                ? (lang === 'en' ? 'Global Intensity' : 'Global Yoğunluk')
                                : (lang === 'en' ? 'Local Suspicion' : 'Yerel Şüphe')}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Legend
        if (legend.length > 0) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-slate-400 cursor-pointer">${this.t('visualization_legend')}</summary>
                    <ul class="mt-2 text-xs text-slate-400 space-y-1">
                        ${legend.map(l => `<li>• ${l}</li>`).join('')}
                    </ul>
                </details>
            `;
        }
        
        // Regions (only for local_manipulation)
        if (mode === 'local_manipulation' && regions.length > 0) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-red-400 cursor-pointer">
                        <i data-lucide="map-pin" class="w-3 h-3 inline mr-1"></i>
                        ${this.t('suspicious_regions')} (${regions.length})
                    </summary>
                    <div class="mt-2 space-y-1 text-xs">
            `;
            
            regions.slice(0, 5).forEach((r, i) => {
                const reasonLabels = {
                    'edge_matte': lang === 'en' ? 'Edge Matte' : 'Kenar Halo',
                    'copy_move': lang === 'en' ? 'Copy-Move' : 'Kopyala-Yapıştır',
                    'splice_noise': lang === 'en' ? 'Splice Noise' : 'Yapıştırma Gürültüsü',
                    'inpaint_boundary': lang === 'en' ? 'Inpaint Boundary' : 'Inpaint Sınırı'
                };
                const reasonLabel = reasonLabels[r.reason] || r.reason;
                
                html += `
                    <div class="flex justify-between text-slate-300 bg-slate-800/30 rounded p-1">
                        <span>#${i+1}: ${reasonLabel} @ (${r.x}, ${r.y})</span>
                        <span class="font-mono">${(r.score * 100).toFixed(0)}%</span>
                    </div>
                `;
            });
            
            html += '</div></details>';
        }
        
        // Hash info (collapsible)
        if (heatmap) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-slate-500 cursor-pointer">${lang === 'en' ? 'Integrity Hashes' : 'Bütünlük Hash\'leri'}</summary>
                    <div class="mt-2 text-xs text-slate-500 font-mono break-all">
                        <p>Overlay: ${heatmap.hash_overlay_sha256?.substring(0, 16)}...</p>
                        <p>Raw: ${heatmap.hash_raw_sha256?.substring(0, 16)}...</p>
                    </div>
                </details>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
        
        // Bind toggle button
        const toggleBtn = document.getElementById('toggleOverlayBtn');
        const overlayContainer = document.getElementById('overlayImageContainer');
        const toggleText = document.getElementById('toggleOverlayText');
        
        if (toggleBtn && overlayContainer) {
            toggleBtn.addEventListener('click', () => {
                const isHidden = overlayContainer.classList.contains('hidden');
                overlayContainer.classList.toggle('hidden');
                toggleText.textContent = isHidden ? this.t('hide_overlay') : this.t('show_overlay');
            });
        }
    }

    displayManipulation(manipulation, localization, editAssessment, visualization) {
        const container = document.getElementById('manipulationContainer');
        if (!container) return;
        
        if (!manipulation || !manipulation.enabled) {
            container.innerHTML = '<p class="text-slate-400 text-sm">Manipülasyon tespiti devre dışı</p>';
            return;
        }
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const overallScore = manipulation.overall_score || 0;
        
        // Use edit_assessment for proper classification
        const editType = editAssessment?.edit_type || 'none_detected';
        const editConfidence = editAssessment?.confidence || 'low';
        const globalScore = editAssessment?.global_adjustment_score || 0;
        const localScore = editAssessment?.local_manipulation_score || 0;
        const genArtifactsScore = editAssessment?.generator_artifacts_score || 0;
        const boundaryCorroborated = editAssessment?.boundary_corroborated || false;
        const showRegions = localization?.show_regions || (editType === 'local_manipulation');
        
        // Edit type labels
        const editTypeLabels = {
            'none_detected': { 
                tr: 'Düzenleme Tespit Edilmedi', 
                en: 'No Edit Detected',
                color: 'green',
                icon: 'check-circle'
            },
            'global_postprocess': { 
                tr: 'Global Düzenleme (Filtre/Renk)', 
                en: 'Global Edit (Filter/Color)',
                color: 'blue',
                icon: 'sliders'
            },
            'local_manipulation': { 
                tr: 'Yerel Manipülasyon Tespit Edildi', 
                en: 'Local Manipulation Detected',
                color: 'red',
                icon: 'alert-triangle'
            },
            'generator_artifacts': { 
                tr: 'AI Üretim Artefaktları', 
                en: 'AI Generation Artifacts',
                color: 'purple',
                icon: 'cpu'
            }
        };
        
        const editInfo = editTypeLabels[editType] || editTypeLabels.none_detected;
        
        let html = `
            <div class="space-y-4">
                <!-- Edit Type Classification (NEW) -->
                <div class="bg-${editInfo.color}-500/10 border border-${editInfo.color}-500/30 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="${editInfo.icon}" class="w-4 h-4 text-${editInfo.color}-400"></i>
                        <span class="font-semibold text-${editInfo.color}-400">
                            ${lang === 'en' ? 'Edit Type' : 'Düzenleme Türü'}: ${lang === 'en' ? editInfo.en : editInfo.tr}
                        </span>
                    </div>
                    <div class="text-xs text-slate-400">
                        ${lang === 'en' ? 'Confidence' : 'Güven'}: ${editConfidence}
                    </div>
                </div>
        `;
        
        // Show explanation based on edit type
        if (editType === 'global_postprocess') {
            html += `
                <div class="bg-blue-500/5 border border-blue-500/20 rounded-lg p-3">
                    <p class="text-xs text-blue-300">
                        ${lang === 'en' 
                            ? 'Global adjustments detected (color grading, filters, platform processing). This is NOT local manipulation/splice.'
                            : 'Global düzenlemeler tespit edildi (renk düzeltme, filtreler, platform işleme). Bu yerel manipülasyon/yapıştırma DEĞİLDİR.'}
                    </p>
                </div>
            `;
        } else if (editType === 'local_manipulation') {
            html += `
                <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                    <div class="flex items-center gap-2 text-red-400">
                        <i data-lucide="alert-triangle" class="w-4 h-4"></i>
                        <span class="font-semibold text-sm">
                            ${lang === 'en' ? 'Local manipulation evidence detected (boundary-corroborated)' : 'Yerel manipülasyon kanıtı tespit edildi (sınır doğrulamalı)'}
                        </span>
                    </div>
                </div>
            `;
        } else if (editType === 'generator_artifacts') {
            html += `
                <div class="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                    <div class="flex items-center gap-2 text-purple-400">
                        <i data-lucide="cpu" class="w-4 h-4"></i>
                        <span class="font-semibold text-sm">
                            ${lang === 'en' ? 'AI generation artifacts detected (not manipulation)' : 'AI üretim artefaktları tespit edildi (manipülasyon değil)'}
                        </span>
                    </div>
                    <p class="text-xs text-purple-300 mt-2">
                        ${lang === 'en' 
                            ? 'Texture/noise inconsistencies are from AI generation process, not from image manipulation or splicing.'
                            : 'Doku/gürültü tutarsızlıkları AI üretim sürecinden kaynaklanıyor, görsel manipülasyonu veya yapıştırmadan değil.'}
                    </p>
                </div>
            `;
        }
        
        // Score breakdown
        html += `
            <div class="grid grid-cols-2 gap-3">
                <div class="bg-slate-800/50 rounded-lg p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Global Edit Score' : 'Global Düzenleme Skoru'}</span>
                    <p class="font-mono text-blue-400">${(globalScore * 100).toFixed(0)}%</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Local Manipulation Score' : 'Yerel Manipülasyon Skoru'}</span>
                    <p class="font-mono ${localScore >= 0.6 ? 'text-red-400' : 'text-slate-400'}">${(localScore * 100).toFixed(0)}%</p>
                </div>
                ${genArtifactsScore > 0 ? `
                <div class="bg-slate-800/50 rounded-lg p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Generator Artifacts Score' : 'Üretici Artefakt Skoru'}</span>
                    <p class="font-mono text-purple-400">${(genArtifactsScore * 100).toFixed(0)}%</p>
                </div>
                ` : ''}
                ${boundaryCorroborated ? `
                <div class="bg-slate-800/50 rounded-lg p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Boundary Corroboration' : 'Sınır Doğrulaması'}</span>
                    <p class="font-mono text-green-400">✓</p>
                </div>
                ` : ''}
            </div>
        `;
        
        // Show individual method scores (collapsed by default)
        const methods = ['splice', 'blur_noise_mismatch', 'copy_move', 'edge_matte'];
        const methodLabels = {
            'splice': { tr: 'Gürültü Tutarsızlığı', en: 'Noise Inconsistency' },
            'blur_noise_mismatch': { tr: 'Bulanıklık Uyumsuzluğu', en: 'Blur Mismatch' },
            'copy_move': { tr: 'Kopyala-Yapıştır', en: 'Copy-Move' },
            'edge_matte': { tr: 'Kenar Halo', en: 'Edge Matte' }
        };
        
        html += `
            <details class="mt-2">
                <summary class="text-xs text-slate-400 cursor-pointer">${lang === 'en' ? 'Method Scores (Technical)' : 'Yöntem Skorları (Teknik)'}</summary>
                <div class="grid grid-cols-2 gap-2 mt-2">
        `;
        
        for (const method of methods) {
            const data = manipulation[method] || {};
            const score = data.score || 0;
            const color = score >= 0.6 ? 'text-red-400' : score >= 0.3 ? 'text-yellow-400' : 'text-slate-400';
            const label = methodLabels[method] || { tr: method, en: method };
            
            html += `
                <div class="bg-slate-800/30 rounded p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? label.en : label.tr}</span>
                    <p class="font-mono text-sm ${color}">${(score * 100).toFixed(0)}%</p>
                </div>
            `;
        }
        html += '</div></details>';
        
        // Show regions ONLY if edit_type is local_manipulation
        if (showRegions && localization?.top_regions?.length > 0) {
            html += `
                <details class="mt-3">
                    <summary class="text-xs text-red-400 cursor-pointer">
                        <i data-lucide="map-pin" class="w-3 h-3 inline mr-1"></i>
                        ${lang === 'en' ? 'Suspicious Regions' : 'Şüpheli Bölgeler'} (${localization.top_regions.length})
                    </summary>
                    <div class="mt-2 space-y-1 text-xs">
            `;
            
            localization.top_regions.slice(0, 5).forEach((r, i) => {
                const methodLabel = methodLabels[r.method]?.[lang === 'en' ? 'en' : 'tr'] || r.method;
                html += `
                    <div class="flex justify-between text-slate-300 bg-slate-800/30 rounded p-1">
                        <span>#${i+1}: ${methodLabel} @ (${r.x}, ${r.y})</span>
                        <span class="font-mono">${(r.score * 100).toFixed(0)}%</span>
                    </div>
                `;
            });
            
            html += '</div></details>';
        } else if (editType !== 'local_manipulation' && manipulation.overall_score > 0.3) {
            // Show note that regions are suppressed for global edits or generator artifacts
            const suppressionReason = editType === 'generator_artifacts' 
                ? (lang === 'en' 
                    ? 'Region display suppressed (AI generation artifacts, not manipulation)'
                    : 'Bölge gösterimi bastırıldı (AI üretim artefaktları, manipülasyon değil)')
                : (lang === 'en' 
                    ? 'Region display suppressed (artifacts from global processing, not local manipulation)'
                    : 'Bölge gösterimi bastırıldı (global işlemeden kaynaklanan artifaktlar, yerel manipülasyon değil)');
            html += `
                <div class="text-xs text-slate-500 mt-2">
                    <i data-lucide="info" class="w-3 h-3 inline mr-1"></i>
                    ${suppressionReason}
                </div>
            `;
        }
        
        // Show edit assessment reasons if available
        if (editAssessment?.reasons?.length > 0) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-slate-500 cursor-pointer">${lang === 'en' ? 'Assessment Reasons' : 'Değerlendirme Nedenleri'}</summary>
                    <ul class="mt-1 text-xs text-slate-400 space-y-0.5">
                        ${editAssessment.reasons.map(r => `<li>• ${r}</li>`).join('')}
                    </ul>
                </details>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
        
        // Display visualization overlay
        this.displayVisualization(visualization, editAssessment);
    }

    displayVerdictText(verdictText, isInformational = false) {
        const badge = document.getElementById('verdictBadge');
        const icon = document.getElementById('verdictIcon');
        const text = document.getElementById('verdictText');
        const levelText = document.getElementById('evidenceLevelText');
        const subtitleEl = document.getElementById('verdictSubtitle');
        const tagsContainer = document.getElementById('verdictTags');
        
        if (!verdictText) return;
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        
        // Title
        const title = lang === 'en' ? verdictText.title_en : verdictText.title_tr;
        text.textContent = title;
        
        // Subtitle
        if (subtitleEl) {
            const subtitle = lang === 'en' ? verdictText.subtitle_en : verdictText.subtitle_tr;
            subtitleEl.textContent = subtitle;
            subtitleEl.classList.remove('hidden');
        }
        
        // Tags
        if (tagsContainer && verdictText.tags?.length) {
            tagsContainer.innerHTML = verdictText.tags.map(tag => 
                `<span class="px-2 py-0.5 bg-slate-700/50 rounded text-xs text-slate-300">${tag}</span>`
            ).join('');
            tagsContainer.classList.remove('hidden');
        }
        
        // Verdict key to color/icon mapping
        const verdictConfig = {
            'NON_PHOTO_HIGH': { color: 'purple', icon: 'palette' },
            'COMPOSITE_LIKELY': { color: 'orange', icon: 'layers' },
            'COMPOSITE_POSSIBLE': { color: 'yellow', icon: 'layers' },
            'PHOTO_GLOBAL_EDIT': { color: 'blue', icon: 'sliders' },
            'AI_T2I_HIGH': { color: 'purple', icon: 'wand-2' },
            'AI_T2I_MEDIUM': { color: 'purple', icon: 'wand-2' },
            'AI_PHOTOREAL_HIGH': { color: 'purple', icon: 'sparkles' },
            'AI_PHOTOREAL_MEDIUM': { color: 'purple', icon: 'sparkles' },
            'GENERATOR_ARTIFACTS': { color: 'blue', icon: 'cpu' },
            'AI_HIGH': { color: 'red', icon: 'alert-triangle' },
            'AI_MEDIUM': { color: 'orange', icon: 'alert-triangle' },
            'REAL_LIKELY': { color: 'green', icon: 'camera' },
            'REAL_MEDIUM': { color: 'blue', icon: 'image' },
            'INCONCLUSIVE': { color: 'gray', icon: 'help-circle' }
        };
        
        const config = verdictConfig[verdictText.verdict_key] || verdictConfig.INCONCLUSIVE;
        
        badge.className = 'verdict-badge';
        badge.classList.add(`verdict-${config.color}`);
        icon.setAttribute('data-lucide', config.icon);
        
        // Evidence level text based on verdict key
        const levelLabels = {
            'NON_PHOTO_HIGH': { tr: 'İçerik Türü', en: 'Content Type' },
            'COMPOSITE_LIKELY': { tr: 'Kompozit Tespit', en: 'Composite Detection' },
            'COMPOSITE_POSSIBLE': { tr: 'Olası Kompozit', en: 'Possible Composite' },
            'PHOTO_GLOBAL_EDIT': { tr: 'Global Düzenleme', en: 'Global Edit' },
            'AI_T2I_HIGH': { tr: 'T2I Yüksek Güven', en: 'T2I High Confidence' },
            'AI_T2I_MEDIUM': { tr: 'T2I Orta Güven', en: 'T2I Medium Confidence' },
            'AI_PHOTOREAL_HIGH': { tr: 'Fotogerçekçi AI', en: 'Photoreal AI' },
            'AI_PHOTOREAL_MEDIUM': { tr: 'Olası Fotogerçekçi AI', en: 'Possible Photoreal AI' },
            'GENERATOR_ARTIFACTS': { tr: 'AI Artefaktları', en: 'AI Artifacts' },
            'AI_HIGH': { tr: 'Yüksek Güven', en: 'High Confidence' },
            'AI_MEDIUM': { tr: 'Orta Güven', en: 'Medium Confidence' },
            'REAL_LIKELY': { tr: 'Yüksek Güven', en: 'High Confidence' },
            'REAL_MEDIUM': { tr: 'Orta Güven', en: 'Medium Confidence' },
            'INCONCLUSIVE': { tr: 'Belirsiz', en: 'Inconclusive' }
        };
        
        const levelLabel = levelLabels[verdictText.verdict_key] || levelLabels.INCONCLUSIVE;
        levelText.textContent = lang === 'en' ? levelLabel.en : levelLabel.tr;
    }

    displayWarningBanner(data) {
        const warningBanner = document.getElementById('warningBanner');
        const warningText = document.getElementById('warningText');
        
        if (!warningBanner) return;
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const verdictText = data.verdict_text;
        
        // Use verdict_text banner if available
        if (verdictText?.banner_key) {
            warningBanner.classList.remove('hidden');
            warningText.textContent = lang === 'en' ? verdictText.banner_en : verdictText.banner_tr;
        } else if (data.warning) {
            // Fallback to old warning field
            warningBanner.classList.remove('hidden');
            warningText.textContent = data.warning;
        } else {
            warningBanner.classList.add('hidden');
        }
    }

    displayGlobalFooter(verdictText) {
        const footerContainer = document.getElementById('globalFooter');
        if (!footerContainer) return;
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        
        if (verdictText?.footer_tr) {
            footerContainer.classList.remove('hidden');
            footerContainer.textContent = lang === 'en' ? verdictText.footer_en : verdictText.footer_tr;
        } else {
            // Default footer
            footerContainer.classList.remove('hidden');
            footerContainer.textContent = lang === 'en' 
                ? 'Note: This is a probabilistic analysis. Definitive verification requires C2PA/Content Credentials or trusted provenance.'
                : 'Not: Bu sonuç olasılıksal analizdir. Kesin doğrulama için C2PA/Content Credentials veya güvenilir provenance gerekir.';
        }
    }

    displaySummaryAxes(summaryAxes, isInformational = false) {
        const container = document.getElementById('summaryAxesContainer');
        if (!container) return;
        
        if (!summaryAxes) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const aiLikelihood = summaryAxes.ai_likelihood || {};
        const evidentialQuality = summaryAxes.evidential_quality || {};
        
        // AI Likelihood level colors and labels
        const aiLevelConfig = {
            'high': { 
                color: 'red', 
                label: lang === 'en' ? 'High AI Likelihood' : 'Yüksek AI Olasılığı',
                icon: 'alert-triangle'
            },
            'medium': { 
                color: 'yellow', 
                label: lang === 'en' ? 'Medium AI Likelihood' : 'Orta AI Olasılığı',
                icon: 'help-circle'
            },
            'low': { 
                color: 'green', 
                label: lang === 'en' ? 'Low AI Likelihood' : 'Düşük AI Olasılığı',
                icon: 'check-circle'
            }
        };
        
        // Evidential Quality level colors and labels
        const qualityLevelConfig = {
            'high': { 
                color: 'green', 
                label: lang === 'en' ? 'High Quality Evidence' : 'Yüksek Kalite Kanıt',
                icon: 'shield-check'
            },
            'medium': { 
                color: 'yellow', 
                label: lang === 'en' ? 'Medium Quality Evidence' : 'Orta Kalite Kanıt',
                icon: 'shield'
            },
            'low': { 
                color: 'orange', 
                label: lang === 'en' ? 'Low Quality Evidence' : 'Düşük Kalite Kanıt',
                icon: 'shield-alert'
            }
        };
        
        const aiConfig = aiLevelConfig[aiLikelihood.level] || aiLevelConfig.medium;
        const qualityConfig = qualityLevelConfig[evidentialQuality.level] || qualityLevelConfig.medium;
        
        // Quality reasons translation
        const reasonLabels = {
            'domain_penalty_high': lang === 'en' ? 'High domain penalty (recompression/screenshot)' : 'Yüksek domain cezası (yeniden sıkıştırma/ekran görüntüsü)',
            'domain_penalty_moderate': lang === 'en' ? 'Moderate domain penalty' : 'Orta domain cezası',
            'high_model_disagreement': lang === 'en' ? 'High model disagreement' : 'Yüksek model uyuşmazlığı',
            'moderate_model_disagreement': lang === 'en' ? 'Moderate model disagreement' : 'Orta model uyuşmazlığı',
            'strong_metadata': lang === 'en' ? 'Strong metadata present' : 'Güçlü metadata mevcut',
            'weak_metadata': lang === 'en' ? 'Weak/missing metadata' : 'Zayıf/eksik metadata',
            'wide_confidence_interval': lang === 'en' ? 'Wide confidence interval' : 'Geniş güven aralığı',
            'moderate_confidence_interval': lang === 'en' ? 'Moderate confidence interval' : 'Orta güven aralığı'
        };
        
        const ci = aiLikelihood.confidence_interval || [0.3, 0.7];
        const prob = aiLikelihood.probability || 0.5;
        
        let html = `
            <div class="grid md:grid-cols-2 gap-4">
                <!-- AI Likelihood Axis -->
                <div class="bg-${aiConfig.color}-500/10 border border-${aiConfig.color}-500/30 rounded-xl p-4">
                    <div class="flex items-center gap-2 mb-3">
                        <i data-lucide="${aiConfig.icon}" class="w-5 h-5 text-${aiConfig.color}-400"></i>
                        <span class="font-semibold text-${aiConfig.color}-400">${aiConfig.label}</span>
                    </div>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-slate-400">${lang === 'en' ? 'Probability' : 'Olasılık'}</span>
                            <span class="font-mono text-white">${(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">${lang === 'en' ? 'CI' : 'Güven Aralığı'}</span>
                            <span class="font-mono text-slate-300">%${(ci[0] * 100).toFixed(0)} - %${(ci[1] * 100).toFixed(0)}</span>
                        </div>
                        ${isInformational ? `
                        <div class="mt-2 text-xs text-amber-400">
                            <i data-lucide="info" class="w-3 h-3 inline mr-1"></i>
                            ${lang === 'en' ? 'Informational only (CGI/Illustration)' : 'Sadece bilgi amaçlı (CGI/İllüstrasyon)'}
                        </div>` : ''}
                    </div>
                </div>
                
                <!-- Evidential Quality Axis -->
                <div class="bg-${qualityConfig.color}-500/10 border border-${qualityConfig.color}-500/30 rounded-xl p-4">
                    <div class="flex items-center gap-2 mb-3">
                        <i data-lucide="${qualityConfig.icon}" class="w-5 h-5 text-${qualityConfig.color}-400"></i>
                        <span class="font-semibold text-${qualityConfig.color}-400">${qualityConfig.label}</span>
                    </div>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-slate-400">${lang === 'en' ? 'Quality Score' : 'Kalite Skoru'}</span>
                            <span class="font-mono text-white">${evidentialQuality.score || 0}/100</span>
                        </div>
                        ${evidentialQuality.reasons?.length ? `
                        <div class="mt-2">
                            <span class="text-xs text-slate-500">${lang === 'en' ? 'Factors:' : 'Faktörler:'}</span>
                            <ul class="text-xs text-slate-400 mt-1 space-y-0.5">
                                ${evidentialQuality.reasons.slice(0, 3).map(r => 
                                    `<li>• ${reasonLabels[r] || r}</li>`
                                ).join('')}
                            </ul>
                        </div>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        // Low quality warning banner
        if (evidentialQuality.level === 'low') {
            html += `
                <div class="mt-3 bg-amber-500/5 border border-amber-500/20 rounded-lg p-3 flex items-start gap-2">
                    <i data-lucide="alert-triangle" class="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5"></i>
                    <p class="text-xs text-amber-200/80">
                        ${lang === 'en' 
                            ? 'Evidence quality is low. Results may be less reliable due to recompression, missing metadata, or model disagreement.'
                            : 'Kanıt kalitesi düşük. Yeniden sıkıştırma, eksik metadata veya model uyuşmazlığı nedeniyle sonuçlar daha az güvenilir olabilir.'}
                    </p>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }

    animateScore(score) {
        const circle = document.getElementById('scoreCircle');
        const scoreValue = document.getElementById('scoreValue');
        const circumference = 339.292;
        
        let current = 0;
        const duration = 1000;
        const start = performance.now();
        
        const animate = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            
            current = Math.round(score * progress);
            scoreValue.textContent = current;
            
            const offset = circumference - (circumference * (current / 100));
            circle.style.strokeDashoffset = offset;
            
            // Color
            if (current < 35) circle.style.stroke = '#22c55e';
            else if (current < 55) circle.style.stroke = '#3b82f6';
            else if (current < 70) circle.style.stroke = '#eab308';
            else circle.style.stroke = '#ef4444';
            
            if (progress < 1) requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }

    updateVerdict(verdict, evidenceLevel, isInformational = false) {
        const badge = document.getElementById('verdictBadge');
        const icon = document.getElementById('verdictIcon');
        const text = document.getElementById('verdictText');
        const levelText = document.getElementById('evidenceLevelText');
        
        if (!verdict) return;
        
        // Use i18n for verdict text if available
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        text.textContent = lang === 'en' ? (verdict.text_en || verdict.text) : verdict.text;
        
        // Handle content type evidence level
        if (evidenceLevel === 'CONTENT_TYPE_NON_PHOTO') {
            levelText.textContent = lang === 'en' ? 'Content Type' : 'İçerik Türü';
        } else {
            levelText.textContent = window.i18n?.formatEvidenceLevel(evidenceLevel) || evidenceLevel;
        }
        
        badge.className = 'verdict-badge';
        const colors = {
            'red': 'verdict-red', 'green': 'verdict-green',
            'orange': 'verdict-orange', 'yellow': 'verdict-yellow',
            'blue': 'verdict-blue', 'gray': 'verdict-gray',
            'purple': 'verdict-purple'
        };
        badge.classList.add(colors[verdict.color] || 'verdict-gray');
        icon.setAttribute('data-lucide', verdict.icon || 'help-circle');
    }

    displayContentType(contentType, isInformational = false) {
        const container = document.getElementById('contentTypeContainer');
        if (!container) return;
        
        if (!contentType || !contentType.enabled) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        
        const typeLabels = {
            'photo_like': { tr: 'Fotoğraf Benzeri', en: 'Photo-like', color: 'green' },
            'non_photo_like': { tr: 'CGI/İllüstrasyon', en: 'CGI/Illustration', color: 'purple' },
            'possible_composite': { tr: 'Olası Kompozit/Düzenlenmiş', en: 'Possible Composite/Edited', color: 'orange' },
            'uncertain': { tr: 'Belirsiz', en: 'Uncertain', color: 'gray' }
        };
        
        const typeInfo = typeLabels[contentType.type] || typeLabels.uncertain;
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const label = lang === 'en' ? typeInfo.en : typeInfo.tr;
        
        const confidenceLabels = {
            'high': { tr: 'Yüksek Güven', en: 'High Confidence' },
            'medium': { tr: 'Orta Güven', en: 'Medium Confidence' },
            'low': { tr: 'Düşük Güven', en: 'Low Confidence' }
        };
        const confLabel = confidenceLabels[contentType.confidence] || confidenceLabels.low;
        
        const iconMap = {
            'non_photo_like': 'palette',
            'photo_like': 'camera',
            'possible_composite': 'layers',
            'uncertain': 'help-circle'
        };
        
        let html = `
            <div class="bg-${typeInfo.color}-500/10 border border-${typeInfo.color}-500/30 rounded-lg p-3">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                        <i data-lucide="${iconMap[contentType.type] || 'help-circle'}" 
                           class="w-4 h-4 text-${typeInfo.color}-400"></i>
                        <span class="font-semibold text-${typeInfo.color}-400">${lang === 'en' ? 'Content Type' : 'İçerik Türü'}: ${label}</span>
                    </div>
                    <span class="text-xs text-slate-400">${lang === 'en' ? confLabel.en : confLabel.tr}</span>
                </div>
        `;
        
        if (contentType.type === 'non_photo_like' && contentType.confidence === 'high') {
            html += `
                <p class="text-xs text-purple-300 mb-2">
                    ${lang === 'en' 
                        ? 'This image appears to be CGI/Illustration. AI probability is informational only.'
                        : 'Bu görsel CGI/İllüstrasyon gibi görünüyor. AI olasılığı sadece bilgi amaçlıdır.'}
                </p>
            `;
        }
        
        if (contentType.type === 'possible_composite') {
            html += `
                <p class="text-xs text-orange-300 mb-2">
                    ${lang === 'en' 
                        ? 'This image may be a composite or edited photo. Elements may have been added, removed, or manipulated.'
                        : 'Bu görsel kompozit veya düzenlenmiş bir fotoğraf olabilir. Öğeler eklenmiş, kaldırılmış veya manipüle edilmiş olabilir.'}
                </p>
            `;
        }
        
        // Show scores
        html += `
            <div class="grid grid-cols-4 gap-2 text-xs mt-2">
                <div>
                    <span class="text-slate-500">Non-Photo</span>
                    <p class="font-mono text-slate-300">${(contentType.non_photo_score * 100).toFixed(0)}%</p>
                </div>
                <div>
                    <span class="text-slate-500">Photo</span>
                    <p class="font-mono text-slate-300">${(contentType.photo_score * 100).toFixed(0)}%</p>
                </div>
                <div>
                    <span class="text-slate-500">Composite</span>
                    <p class="font-mono text-slate-300">${((contentType.composite_score || 0) * 100).toFixed(0)}%</p>
                </div>
                <div>
                    <span class="text-slate-500">Margin</span>
                    <p class="font-mono text-slate-300">${(contentType.margin * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
        
        // Top matches (collapsible)
        if (contentType.top_matches) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-slate-400 cursor-pointer">${lang === 'en' ? 'Top Matches' : 'En İyi Eşleşmeler'}</summary>
                    <div class="mt-2 text-xs space-y-1">
            `;
            
            if (contentType.top_matches.non_photo?.length) {
                html += `<p class="text-purple-400">Non-Photo:</p>`;
                contentType.top_matches.non_photo.slice(0, 2).forEach(m => {
                    html += `<p class="text-slate-400 pl-2">• ${m.prompt}: ${(m.sim * 100).toFixed(0)}%</p>`;
                });
            }
            
            if (contentType.top_matches.photo?.length) {
                html += `<p class="text-green-400 mt-1">Photo:</p>`;
                contentType.top_matches.photo.slice(0, 2).forEach(m => {
                    html += `<p class="text-slate-400 pl-2">• ${m.prompt}: ${(m.sim * 100).toFixed(0)}%</p>`;
                });
            }
            
            if (contentType.top_matches.composite?.length) {
                html += `<p class="text-orange-400 mt-1">Composite:</p>`;
                contentType.top_matches.composite.slice(0, 2).forEach(m => {
                    html += `<p class="text-slate-400 pl-2">• ${m.prompt}: ${(m.sim * 100).toFixed(0)}%</p>`;
                });
            }
            
            html += `</div></details>`;
        }
        
        html += `</div>`;
        container.innerHTML = html;
    }

    displayUncertainty(uncertainty) {
        const container = document.getElementById('uncertaintyContainer');
        if (!container) return;
        
        const ci = uncertainty.confidence_interval || [0.3, 0.7];
        const disagreement = uncertainty.disagreement || 0;
        
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-3 text-sm">
                <div>
                    <span class="text-slate-400">${this.t('confidence_interval')}</span>
                    <p class="font-mono text-slate-200">%${Math.round(ci[0]*100)} - %${Math.round(ci[1]*100)}</p>
                </div>
                <div>
                    <span class="text-slate-400">${this.t('model_disagreement')}</span>
                    <p class="font-mono text-slate-200">${(disagreement * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
    }

    displayModelScores(models) {
        const container = document.getElementById('modelScoresContainer');
        if (!container || !models.length) {
            if (container) container.innerHTML = '<p class="text-slate-400 text-sm">Model skoru yok</p>';
            return;
        }
        
        container.innerHTML = models.map(m => `
            <div class="mb-3">
                <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-300">${m.name}</span>
                    <span class="font-mono ${m.raw_score > 0.6 ? 'text-orange-400' : 'text-blue-400'}">
                        ${Math.round(m.raw_score * 100)}%
                    </span>
                </div>
                <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div class="h-full ${m.raw_score > 0.6 ? 'bg-orange-500' : 'bg-blue-500'} rounded-full transition-all"
                         style="width: ${m.raw_score * 100}%"></div>
                </div>
            </div>
        `).join('');
    }

    displayFindings(data) {
        const container = document.getElementById('findingsContainer');
        if (!container) return;
        
        let html = '';
        
        if (data.definitive_findings?.length) {
            html += this.createFindingSection(this.t('definitive_findings'), data.definitive_findings, 'shield-check', 'red');
        }
        
        if (data.strong_evidence?.length) {
            const items = data.strong_evidence.map(e => e.description || e);
            html += this.createFindingSection(this.t('strong_evidence'), items, 'check-circle', 'green');
        }
        
        if (data.authenticity_indicators?.length) {
            const items = data.authenticity_indicators.map(e => e.description || e);
            html += this.createFindingSection(this.t('authenticity_indicators'), items, 'camera', 'emerald');
        }
        
        if (data.statistical_indicators?.length) {
            const items = data.statistical_indicators.map(e => 
                `${e.description} (${e.suggests === 'real' ? this.t('suggests_real') : this.t('suggests_ai')})`
            );
            html += this.createFindingSection(this.t('technical_indicators'), items, 'bar-chart-2', 'yellow');
        }
        
        container.innerHTML = html || `<p class="text-slate-400 text-sm">${this.t('no_findings')}</p>`;
    }

    createFindingSection(title, items, icon, color) {
        return `
            <div class="mb-4">
                <h5 class="text-sm font-medium text-${color}-400 mb-2 flex items-center gap-2">
                    <i data-lucide="${icon}" class="w-4 h-4"></i>${title}
                </h5>
                <ul class="space-y-1">
                    ${items.map(item => `
                        <li class="text-sm text-slate-300 flex items-start gap-2">
                            <i data-lucide="chevron-right" class="w-3 h-3 mt-1 text-slate-500"></i>
                            <span>${item}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    displayGPS(gps) {
        const container = document.getElementById('gpsContainer');
        if (!container) return;
        
        if (gps.present && gps.latitude && gps.longitude) {
            const confidence = gps.confidence || 'unknown';
            const confidenceColors = {
                'high': 'text-green-400',
                'medium': 'text-yellow-400',
                'low': 'text-orange-400'
            };
            
            container.innerHTML = `
                <div class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                    <div class="flex items-center gap-2 text-blue-400 mb-3">
                        <i data-lucide="map-pin" class="w-5 h-5"></i>
                        <span class="font-semibold">${this.t('gps_found')}</span>
                        <span class="text-xs ${confidenceColors[confidence] || 'text-slate-400'}">(${confidence})</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-sm mb-3">
                        <div>
                            <span class="text-slate-400">${this.t('latitude')}</span>
                            <p class="font-mono text-slate-200">${gps.latitude.toFixed(6)}</p>
                        </div>
                        <div>
                            <span class="text-slate-400">${this.t('longitude')}</span>
                            <p class="font-mono text-slate-200">${gps.longitude.toFixed(6)}</p>
                        </div>
                        ${gps.altitude_m ? `
                        <div>
                            <span class="text-slate-400">${this.t('altitude')}</span>
                            <p class="font-mono text-slate-200">${gps.altitude_m}m</p>
                        </div>` : ''}
                        ${gps.timestamp_utc ? `
                        <div>
                            <span class="text-slate-400">${this.t('gps_time')}</span>
                            <p class="font-mono text-slate-200 text-xs">${gps.timestamp_utc}</p>
                        </div>` : ''}
                    </div>
                    <div class="text-xs text-amber-400 mt-2">
                        <i data-lucide="alert-triangle" class="w-3 h-3 inline mr-1"></i>
                        ${this.t('gps_warning')}
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="bg-slate-700/50 rounded-lg p-4 text-center">
                    <i data-lucide="map-pin-off" class="w-8 h-8 text-slate-500 mx-auto mb-2"></i>
                    <p class="text-sm text-slate-400">${this.t('gps_not_found')}</p>
                </div>
            `;
        }
    }

    displayCamera(camera, metadata) {
        const container = document.getElementById('cameraContainer');
        if (!container) return;
        
        if (camera.has_camera_info) {
            const tech = metadata.technical || {};
            container.innerHTML = `
                <div class="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                    <div class="flex items-center gap-2 text-green-400 mb-3">
                        <i data-lucide="camera" class="w-5 h-5"></i>
                        <span class="font-semibold">${this.t('camera_found')}</span>
                    </div>
                    <div class="space-y-2 text-sm">
                        ${camera.make ? `<p><span class="text-slate-400">${this.t('camera_make')}:</span> <span class="text-white">${camera.make}</span></p>` : ''}
                        ${camera.model ? `<p><span class="text-slate-400">${this.t('camera_model')}:</span> <span class="text-white">${camera.model}</span></p>` : ''}
                        ${camera.lens ? `<p><span class="text-slate-400">${this.t('lens')}:</span> <span class="text-white">${camera.lens}</span></p>` : ''}
                        ${tech.exposure_time ? `<p><span class="text-slate-400">${this.t('exposure')}:</span> <span class="text-white">${tech.exposure_time}</span></p>` : ''}
                        ${tech.f_number ? `<p><span class="text-slate-400">${this.t('aperture')}:</span> <span class="text-white">f/${tech.f_number}</span></p>` : ''}
                        ${tech.iso ? `<p><span class="text-slate-400">${this.t('iso')}:</span> <span class="text-white">${tech.iso}</span></p>` : ''}
                        ${tech.focal_length ? `<p><span class="text-slate-400">${this.t('focal_length')}:</span> <span class="text-white">${tech.focal_length}</span></p>` : ''}
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                    <div class="flex items-center gap-2 text-yellow-400 mb-2">
                        <i data-lucide="camera-off" class="w-5 h-5"></i>
                        <span class="font-semibold">${this.t('camera_not_found')}</span>
                    </div>
                    <p class="text-sm text-slate-400">EXIF'te kamera verisi bulunamadı</p>
                </div>
            `;
        }
    }

    displayAIDetection(aiDetection) {
        const container = document.getElementById('aiDetectionContainer');
        if (!container) return;
        
        if (aiDetection.detected) {
            container.innerHTML = `
                <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                    <div class="flex items-center gap-2 text-red-400 mb-3">
                        <i data-lucide="alert-octagon" class="w-5 h-5"></i>
                        <span class="font-semibold">${this.t('ai_detected')}</span>
                    </div>
                    <div class="space-y-1 text-sm">
                        <p><span class="text-slate-400">${this.t('software')}:</span> <span class="text-white">${aiDetection.software}</span></p>
                        <p><span class="text-slate-400">${this.t('vendor')}:</span> <span class="text-white">${aiDetection.vendor}</span></p>
                        <p><span class="text-slate-400">${this.t('type')}:</span> <span class="text-white">${aiDetection.type}</span></p>
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = `<p class="text-sm text-slate-400">${this.t('ai_not_detected')}</p>`;
        }
    }

    displayDomain(domain) {
        const container = document.getElementById('domainContainer');
        if (!container) return;
        
        const typeLabel = window.i18n?.formatDomain(domain.type) || domain.type;
        
        container.innerHTML = `
            <div class="text-sm">
                <p class="mb-2"><span class="text-slate-400">${this.t('type')}:</span> 
                    <span class="text-slate-200">${typeLabel}</span>
                </p>
                ${domain.characteristics?.length ? `
                <p class="text-slate-400 mb-1">Özellikler:</p>
                <ul class="text-slate-300 text-xs space-y-1">
                    ${domain.characteristics.map(c => `<li>• ${c}</li>`).join('')}
                </ul>` : ''}
                ${domain.warnings?.length ? `
                <div class="mt-2 text-xs text-amber-400">
                    ${domain.warnings.map(w => `<p>⚠️ ${w}</p>`).join('')}
                </div>` : ''}
            </div>
        `;
    }

    displayImageInfo(info) {
        const container = document.getElementById('imageInfoContainer');
        if (!container) return;
        
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-2 text-sm">
                <div><span class="text-slate-400">${this.t('dimensions')}</span><p class="text-slate-200">${info.width} x ${info.height}</p></div>
                <div><span class="text-slate-400">${this.t('format')}</span><p class="text-slate-200">${info.format || 'Bilinmiyor'}</p></div>
                <div><span class="text-slate-400">${this.t('file_size')}</span><p class="text-slate-200">${info.file_size_kb} KB</p></div>
                <div><span class="text-slate-400">${this.t('color_mode')}</span><p class="text-slate-200">${info.mode}</p></div>
            </div>
        `;
    }

    // === NEW FORENSIC SECTIONS ===
    
    displayJPEGForensics(jpegData) {
        const container = document.getElementById('jpegForensicsContainer');
        if (!container) return;
        
        if (!jpegData) {
            container.innerHTML = `<p class="text-slate-400 text-sm">JPEG forensics devre dışı veya veri yok</p>`;
            return;
        }
        
        const qualityColor = jpegData.estimated_quality > 85 ? 'text-green-400' : 
                            jpegData.estimated_quality > 70 ? 'text-yellow-400' : 'text-orange-400';
        
        const dcProb = jpegData.double_compression_probability || 0;
        const dcColor = dcProb > 0.6 ? 'text-red-400' : dcProb > 0.3 ? 'text-yellow-400' : 'text-green-400';
        
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('jpeg_quality')}</span>
                    <p class="font-mono text-lg ${qualityColor}">${jpegData.estimated_quality || 'N/A'}%</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('subsampling')}</span>
                    <p class="font-mono text-lg text-slate-200">${jpegData.subsampling || 'N/A'}</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('double_compression')}</span>
                    <p class="font-mono text-lg ${dcColor}">${(dcProb * 100).toFixed(0)}%</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('quant_fingerprint')}</span>
                    <p class="font-mono text-xs text-slate-200 truncate">${jpegData.quantization_fingerprint || 'N/A'}</p>
                </div>
            </div>
            ${jpegData.notes?.length ? `
            <div class="mt-3 text-xs text-slate-400">
                ${jpegData.notes.map(n => `<p>• ${n}</p>`).join('')}
            </div>` : ''}
        `;
    }

    displayAIType(aiTypeData) {
        const container = document.getElementById('aiTypeContainer');
        if (!container) return;
        
        if (!aiTypeData) {
            container.innerHTML = `<p class="text-slate-400 text-sm">AI tip sınıflandırması devre dışı veya veri yok</p>`;
            return;
        }
        
        const typeLabel = window.i18n?.formatAIType(aiTypeData.predicted_type) || aiTypeData.predicted_type;
        const confidence = aiTypeData.confidence || 0;
        
        const typeColors = {
            'text_to_image': 'text-purple-400',
            'img2img': 'text-blue-400',
            'inpainting': 'text-orange-400',
            'unknown': 'text-slate-400'
        };
        
        container.innerHTML = `
            <div class="bg-slate-800/50 rounded-lg p-4">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-slate-400 text-sm">${this.t('ai_type_guess')}</span>
                    <span class="font-mono text-xs text-slate-500">${(confidence * 100).toFixed(0)}% güven</span>
                </div>
                <p class="text-xl font-semibold ${typeColors[aiTypeData.predicted_type] || 'text-slate-200'}">${typeLabel}</p>
                ${aiTypeData.indicators?.length ? `
                <div class="mt-3 text-xs text-slate-400">
                    <p class="mb-1">Göstergeler:</p>
                    ${aiTypeData.indicators.map(i => `<p>• ${i}</p>`).join('')}
                </div>` : ''}
                <p class="mt-3 text-xs text-amber-400/70">
                    <i data-lucide="info" class="w-3 h-3 inline mr-1"></i>
                    ${this.t('ai_type_disclaimer')}
                </p>
            </div>
        `;
    }

    displayPathway(pathwayData, diffusionData) {
        const container = document.getElementById('pathwayContainer');
        if (!container) return;
        
        if (!pathwayData) {
            container.innerHTML = `<p class="text-slate-400 text-sm">Pathway sınıflandırması devre dışı</p>`;
            return;
        }
        
        const lang = window.i18n?.getCurrentLanguage() || 'tr';
        const pred = pathwayData.pred || 'unknown';
        const confidence = pathwayData.confidence || 'low';
        const probs = pathwayData.probs || {};
        const evidence = pathwayData.evidence || {};
        const generatorFamily = pathwayData.generator_family || {};
        
        // Pathway labels
        const pathwayLabels = {
            'real_photo': { tr: 'Gerçek Fotoğraf', en: 'Real Photo', color: 'green', icon: 'camera' },
            't2i': { tr: 'Text-to-Image (T2I)', en: 'Text-to-Image (T2I)', color: 'purple', icon: 'wand-2' },
            'i2i': { tr: 'Image-to-Image (I2I)', en: 'Image-to-Image (I2I)', color: 'blue', icon: 'layers' },
            'inpainting': { tr: 'Inpainting', en: 'Inpainting', color: 'orange', icon: 'eraser' },
            'stylization': { tr: 'Stilizasyon', en: 'Stylization', color: 'pink', icon: 'palette' },
            'unknown': { tr: 'Belirsiz', en: 'Unknown', color: 'gray', icon: 'help-circle' }
        };
        
        const pathwayInfo = pathwayLabels[pred] || pathwayLabels.unknown;
        const label = lang === 'en' ? pathwayInfo.en : pathwayInfo.tr;
        
        // Generator family labels
        const familyLabels = {
            'diffusion_t2i_modern': { tr: 'Modern Diffusion T2I', en: 'Modern Diffusion T2I' },
            'diffusion_t2i_legacy': { tr: 'Eski Diffusion T2I', en: 'Legacy Diffusion T2I' },
            'diffusion_i2i': { tr: 'Diffusion I2I', en: 'Diffusion I2I' },
            'gan_legacy': { tr: 'GAN (Eski)', en: 'GAN (Legacy)' },
            'render_3d': { tr: '3D Render', en: '3D Render' },
            'camera_photo': { tr: 'Kamera Fotoğrafı', en: 'Camera Photo' },
            'unknown': { tr: 'Bilinmiyor', en: 'Unknown' }
        };
        
        const familyInfo = familyLabels[generatorFamily.pred] || familyLabels.unknown;
        const familyLabel = lang === 'en' ? familyInfo.en : familyInfo.tr;
        
        // Confidence labels
        const confLabels = {
            'high': { tr: 'Yüksek', en: 'High' },
            'medium': { tr: 'Orta', en: 'Medium' },
            'low': { tr: 'Düşük', en: 'Low' }
        };
        const confLabel = confLabels[confidence] || confLabels.low;
        
        let html = `
            <div class="space-y-4">
                <!-- Main Pathway -->
                <div class="bg-${pathwayInfo.color}-500/10 border border-${pathwayInfo.color}-500/30 rounded-lg p-4">
                    <div class="flex items-center gap-3 mb-2">
                        <i data-lucide="${pathwayInfo.icon}" class="w-6 h-6 text-${pathwayInfo.color}-400"></i>
                        <div>
                            <p class="text-lg font-semibold text-${pathwayInfo.color}-400">${label}</p>
                            <p class="text-xs text-slate-400">${lang === 'en' ? 'Confidence' : 'Güven'}: ${lang === 'en' ? confLabel.en : confLabel.tr}</p>
                        </div>
                    </div>
                </div>
        `;
        
        // Generator Family
        if (generatorFamily.pred && generatorFamily.pred !== 'unknown') {
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Generator Family' : 'Üretici Ailesi'}</span>
                    <p class="font-semibold text-slate-200">${familyLabel}</p>
                    <p class="text-xs text-slate-400">${lang === 'en' ? 'Confidence' : 'Güven'}: ${generatorFamily.confidence || 'low'}</p>
                </div>
            `;
        }
        
        // Diffusion Score
        if (diffusionData && diffusionData.diffusion_score !== undefined) {
            const diffScore = diffusionData.diffusion_score;
            const diffColor = diffScore >= 0.7 ? 'text-purple-400' : diffScore >= 0.5 ? 'text-yellow-400' : 'text-green-400';
            
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-slate-500">${lang === 'en' ? 'Diffusion Fingerprint' : 'Diffusion İzi'}</span>
                        <span class="font-mono ${diffColor}">${(diffScore * 100).toFixed(0)}%</span>
                    </div>
                    <div class="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div class="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-purple-500 rounded-full" 
                             style="width: ${diffScore * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        // Evidence Scores
        html += `
            <div class="grid grid-cols-2 gap-2">
                <div class="bg-slate-800/30 rounded p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'I2I Evidence' : 'I2I Kanıtı'}</span>
                    <p class="font-mono text-sm ${evidence.i2i_evidence_score >= 0.6 ? 'text-blue-400' : 'text-slate-400'}">
                        ${((evidence.i2i_evidence_score || 0) * 100).toFixed(0)}%
                    </p>
                </div>
                <div class="bg-slate-800/30 rounded p-2">
                    <span class="text-xs text-slate-500">${lang === 'en' ? 'Camera Evidence' : 'Kamera Kanıtı'}</span>
                    <p class="font-mono text-sm ${evidence.camera_evidence_score >= 0.5 ? 'text-green-400' : 'text-slate-400'}">
                        ${((evidence.camera_evidence_score || 0) * 100).toFixed(0)}%
                    </p>
                </div>
            </div>
        `;
        
        // Signals
        if (evidence.signals?.length > 0) {
            html += `
                <details class="mt-2">
                    <summary class="text-xs text-slate-400 cursor-pointer">${lang === 'en' ? 'Detection Signals' : 'Tespit Sinyalleri'}</summary>
                    <ul class="mt-2 text-xs text-slate-400 space-y-1">
                        ${evidence.signals.slice(0, 5).map(s => `<li>• ${s}</li>`).join('')}
                    </ul>
                </details>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }

    displayStatistics(statsData) {
        const container = document.getElementById('statisticsContainer');
        if (!container) return;
        
        if (!statsData) {
            container.innerHTML = `<p class="text-slate-400 text-sm">İstatistik analizi devre dışı veya veri yok</p>`;
            return;
        }
        
        const ci = statsData.confidence_interval || [0.3, 0.7];
        
        // Handle ECE - check for null/undefined/NaN
        const ece = statsData.ece;
        let eceValue = null;
        let eceColor = 'text-slate-400';
        let eceDisplay = 'N/A';
        
        if (ece && ece.value !== null && ece.value !== undefined && !isNaN(ece.value)) {
            eceValue = ece.value;
            eceColor = eceValue < 0.05 ? 'text-green-400' : eceValue < 0.1 ? 'text-yellow-400' : 'text-red-400';
            eceDisplay = `${(eceValue * 100).toFixed(1)}%`;
        }
        
        // Handle expected error rate
        const errorRate = statsData.expected_error_rate;
        let errorDisplay = 'N/A';
        if (errorRate && errorRate.overall !== null && errorRate.overall !== undefined && !isNaN(errorRate.overall)) {
            errorDisplay = `${(errorRate.overall * 100).toFixed(1)}%`;
        }
        
        // Handle model agreement
        const agreement = statsData.model_agreement;
        let disagreementDisplay = 'N/A';
        if (agreement !== null && agreement !== undefined && !isNaN(agreement)) {
            disagreementDisplay = `${((1 - agreement) * 100).toFixed(1)}%`;
        }
        
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('confidence_interval')}</span>
                    <p class="font-mono text-lg text-slate-200">%${Math.round(ci[0]*100)} - %${Math.round(ci[1]*100)}</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('ece')}</span>
                    <p class="font-mono text-lg ${eceColor}">${eceDisplay}</p>
                    ${eceValue === null ? '<p class="text-xs text-slate-500">(kalibrasyon yok)</p>' : ''}
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('expected_error')}</span>
                    <p class="font-mono text-lg text-slate-200">${errorDisplay}</p>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <span class="text-slate-400 text-xs">${this.t('model_disagreement')}</span>
                    <p class="font-mono text-lg text-slate-200">${disagreementDisplay}</p>
                </div>
            </div>
            ${statsData.inconclusive_reason ? `
            <div class="mt-3 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
                <p class="text-sm text-amber-200">⚠️ ${statsData.inconclusive_reason}</p>
            </div>` : ''}
        `;
    }

    displayExtendedMetadata(metadataExt) {
        const container = document.getElementById('metadataContainer');
        if (!container) return;
        
        if (!metadataExt) {
            container.innerHTML = `<p class="text-slate-400 text-sm">Genişletilmiş metadata yok</p>`;
            return;
        }
        
        let html = '<div class="space-y-4">';
        
        // EXIF summary
        const exif = metadataExt.exif || {};
        const exifCount = Object.keys(exif).length;
        html += `
            <div class="bg-slate-800/50 rounded-lg p-3">
                <h5 class="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                    <i data-lucide="file-text" class="w-4 h-4 text-blue-400"></i>EXIF
                </h5>
                <p class="text-xs text-slate-400">${exifCount} alan bulundu</p>
                ${exifCount > 0 ? `
                <details class="mt-2">
                    <summary class="text-xs text-blue-400 cursor-pointer">${this.t('more_details')}</summary>
                    <div class="mt-2 text-xs text-slate-300 max-h-40 overflow-y-auto">
                        ${Object.entries(exif).slice(0, 20).map(([k, v]) => 
                            `<p><span class="text-slate-500">${k}:</span> ${String(v).substring(0, 50)}</p>`
                        ).join('')}
                        ${exifCount > 20 ? `<p class="text-slate-500">... ve ${exifCount - 20} daha</p>` : ''}
                    </div>
                </details>` : ''}
            </div>
        `;
        
        // XMP
        if (metadataExt.xmp && Object.keys(metadataExt.xmp).length > 0) {
            const xmpCount = Object.keys(metadataExt.xmp).length;
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <h5 class="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <i data-lucide="code" class="w-4 h-4 text-purple-400"></i>XMP
                    </h5>
                    <p class="text-xs text-slate-400">${xmpCount} alan bulundu</p>
                    <details class="mt-2">
                        <summary class="text-xs text-purple-400 cursor-pointer">${this.t('more_details')}</summary>
                        <div class="mt-2 text-xs text-slate-300 max-h-40 overflow-y-auto">
                            ${Object.entries(metadataExt.xmp).slice(0, 15).map(([k, v]) => 
                                `<p><span class="text-slate-500">${k}:</span> ${String(v).substring(0, 50)}</p>`
                            ).join('')}
                        </div>
                    </details>
                </div>
            `;
        }
        
        // IPTC
        if (metadataExt.iptc && Object.keys(metadataExt.iptc).length > 0) {
            const iptcCount = Object.keys(metadataExt.iptc).length;
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <h5 class="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <i data-lucide="tag" class="w-4 h-4 text-green-400"></i>IPTC
                    </h5>
                    <p class="text-xs text-slate-400">${iptcCount} alan bulundu</p>
                    <details class="mt-2">
                        <summary class="text-xs text-green-400 cursor-pointer">${this.t('more_details')}</summary>
                        <div class="mt-2 text-xs text-slate-300 max-h-40 overflow-y-auto">
                            ${Object.entries(metadataExt.iptc).slice(0, 15).map(([k, v]) => 
                                `<p><span class="text-slate-500">${k}:</span> ${String(v).substring(0, 50)}</p>`
                            ).join('')}
                        </div>
                    </details>
                </div>
            `;
        }
        
        // Software history
        if (metadataExt.software_history?.length > 0) {
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <h5 class="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <i data-lucide="history" class="w-4 h-4 text-orange-400"></i>Yazılım Geçmişi
                    </h5>
                    <ul class="text-xs text-slate-300 space-y-1">
                        ${metadataExt.software_history.map(s => `<li>• ${s}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Format info
        if (metadataExt.format_info) {
            const fi = metadataExt.format_info;
            html += `
                <div class="bg-slate-800/50 rounded-lg p-3">
                    <h5 class="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <i data-lucide="file-type" class="w-4 h-4 text-cyan-400"></i>Format Bilgisi
                    </h5>
                    <div class="text-xs text-slate-300 space-y-1">
                        ${fi.format ? `<p><span class="text-slate-500">Format:</span> ${fi.format}</p>` : ''}
                        ${fi.is_heic ? `<p><span class="text-slate-500">HEIC:</span> Evet</p>` : ''}
                        ${fi.color_profile ? `<p><span class="text-slate-500">Renk Profili:</span> ${fi.color_profile}</p>` : ''}
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.detector = new AIImageDetector();
});
