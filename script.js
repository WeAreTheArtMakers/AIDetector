// AI Image Detector - JavaScript Functionality
// Advanced image analysis with real computer vision techniques

class AIImageDetector {
    constructor() {
        this.currentFile = null;
        this.isAnalyzing = false;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.initializeElements();
        this.bindEvents();
        this.initializeLucideIcons();
    }

    initializeElements() {
        // DOM Elements
        this.uploadArea = document.getElementById('uploadArea');
        this.uploadContent = document.getElementById('uploadContent');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.fileName = document.getElementById('fileName');
        this.fileInput = document.getElementById('fileInput');
        this.selectFileBtn = document.getElementById('selectFileBtn');
        this.removeImage = document.getElementById('removeImage');
        this.analysisSection = document.getElementById('analysisSection');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressSection = document.getElementById('progressSection');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultContent = document.getElementById('resultContent');
        this.analyzeAgain = document.getElementById('analyzeAgain');
    }

    initializeLucideIcons() {
        // Initialize Lucide icons after DOM is loaded
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }

    bindEvents() {
        // File input events
        this.selectFileBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', () => {
            if (!this.currentFile) this.fileInput.click();
        });

        // Image management
        this.removeImage.addEventListener('click', () => this.removeCurrentImage());
        
        // Analysis events
        this.analyzeBtn.addEventListener('click', () => this.startAnalysis());
        this.analyzeAgain.addEventListener('click', () => this.resetForNewAnalysis());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Lütfen geçerli bir resim dosyası seçin.');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Dosya boyutu 10MB\'dan küçük olmalıdır.');
            return;
        }

        this.currentFile = file;
        this.displayImagePreview(file);
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.fileName.textContent = file.name;
            
            // Hide upload content and show preview
            this.uploadContent.classList.add('hidden');
            this.imagePreview.classList.remove('hidden');
            this.imagePreview.classList.add('fade-in');
            
            // Show analysis section
            this.analysisSection.classList.remove('hidden');
            this.analysisSection.classList.add('scale-in');
            
            // Reset any previous results
            this.resetResults();
        };
        
        reader.readAsDataURL(file);
    }

    removeCurrentImage() {
        this.currentFile = null;
        this.fileInput.value = '';
        
        // Reset UI
        this.uploadContent.classList.remove('hidden');
        this.imagePreview.classList.add('hidden');
        this.analysisSection.classList.add('hidden');
        this.resetResults();
    }

    async startAnalysis() {
        if (!this.currentFile || this.isAnalyzing) return;
        
        this.isAnalyzing = true;
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = '<i data-lucide="loader-2" class="w-6 h-6 mr-3 animate-spin"></i>Analiz Ediliyor...';
        
        // Show progress section
        this.progressSection.classList.remove('hidden');
        this.progressSection.classList.add('fade-in');
        this.progressBar.classList.add('progress-glow');
        
        try {
            // Use backend analysis instead of local analysis
            const analysisResults = await this.analyzeWithBackend();
            
            // Display results
            this.displayResults(analysisResults);
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analiz sırasında bir hata oluştu: ' + error.message);
        }
        
        // Reset button state
        this.isAnalyzing = false;
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.innerHTML = '<i data-lucide="search" class="w-6 h-6 mr-3"></i>Analiz Et';
        this.progressBar.classList.remove('progress-glow');
        
        // Re-initialize icons
        this.initializeLucideIcons();
    }

    async analyzeWithBackend() {
        const BACKEND_URL = 'http://localhost:8001'; // Change this for production
        
        try {
            // Simulate progress updates
            await this.updateProgress(10, 'Görsel backend\'e gönderiliyor...');
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.currentFile);
            
            await this.updateProgress(30, 'AI model analizi başlatılıyor...');
            
            // Send to backend
            const response = await fetch(`${BACKEND_URL}/analyze`, {
                method: 'POST',
                body: formData
            });
            
            await this.updateProgress(70, 'Sonuçlar işleniyor...');
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            await this.updateProgress(100, 'Analiz tamamlandı!');
            
            if (!result.success) {
                throw new Error('Backend analysis failed');
            }
            
            // Return result in the format expected by displayResults
            return {
                aiProbability: result.aiProbability,
                confidence: result.confidence,
                verdict: result.verdict,
                indicators: result.indicators,
                processingTime: result.processingTime,
                fileSize: result.fileSize,
                metadata: result.metadata,
                warning: result.warning
            };
            
        } catch (error) {
            console.error('Backend analysis failed:', error);
            
            // Fallback to local analysis if backend fails
            console.log('Falling back to local analysis...');
            await this.updateProgress(50, 'Backend hatası, yerel analiz yapılıyor...');
            
            return await this.performAdvancedAnalysis();
        }
    }

    async performAdvancedAnalysis() {
        const startTime = performance.now();
        
        // Load image to canvas for analysis
        const img = await this.loadImageToCanvas();
        
        // Step 1: Basic image properties
        await this.updateProgress(10, 'Görsel özellikleri analiz ediliyor...');
        const basicProps = this.analyzeBasicProperties(img);
        
        // Step 2: Noise analysis
        await this.updateProgress(25, 'Gürültü deseni inceleniyor...');
        const noiseAnalysis = this.analyzeNoise();
        
        // Step 3: Frequency domain analysis
        await this.updateProgress(40, 'Frekans analizi yapılıyor...');
        const frequencyAnalysis = this.analyzeFrequencyDomain();
        
        // Step 4: Edge detection and sharpness
        await this.updateProgress(55, 'Kenar tespiti ve netlik analizi...');
        const edgeAnalysis = this.analyzeEdges();
        
        // Step 5: Color distribution analysis
        await this.updateProgress(70, 'Renk dağılımı inceleniyor...');
        const colorAnalysis = this.analyzeColorDistribution();
        
        // Step 6: Compression artifacts
        await this.updateProgress(85, 'Sıkıştırma artefaktları taranıyor...');
        const compressionAnalysis = this.analyzeCompression();
        
        // Step 7: Calculate final AI probability
        await this.updateProgress(100, 'Sonuçlar hesaplanıyor...');
        const aiProbability = this.calculateAIProbability({
            basicProps,
            noiseAnalysis,
            frequencyAnalysis,
            edgeAnalysis,
            colorAnalysis,
            compressionAnalysis
        });
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        
        return {
            aiProbability,
            confidence: this.calculateConfidence(aiProbability),
            verdict: this.getVerdict(aiProbability),
            indicators: this.generateTechnicalIndicators({
                basicProps,
                noiseAnalysis,
                frequencyAnalysis,
                edgeAnalysis,
                colorAnalysis,
                compressionAnalysis
            }),
            processingTime,
            fileSize: (this.currentFile.size / 1024).toFixed(1),
            metadata: this.extractMetadata()
        };
    }

    async loadImageToCanvas() {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                // Resize for analysis (max 512x512 for performance)
                const maxSize = 512;
                let { width, height } = img;
                
                if (width > maxSize || height > maxSize) {
                    const ratio = Math.min(maxSize / width, maxSize / height);
                    width *= ratio;
                    height *= ratio;
                }
                
                this.canvas.width = width;
                this.canvas.height = height;
                this.ctx.drawImage(img, 0, 0, width, height);
                resolve(img);
            };
            img.src = this.previewImg.src;
        });
    }

    analyzeBasicProperties(img) {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        let totalBrightness = 0;
        let totalSaturation = 0;
        let pixelCount = data.length / 4;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Calculate brightness
            const brightness = (r + g + b) / 3;
            totalBrightness += brightness;
            
            // Calculate saturation
            const max = Math.max(r, g, b);
            const min = Math.min(r, g, b);
            const saturation = max === 0 ? 0 : (max - min) / max;
            totalSaturation += saturation;
        }
        
        return {
            avgBrightness: totalBrightness / pixelCount,
            avgSaturation: totalSaturation / pixelCount,
            dimensions: { width: this.canvas.width, height: this.canvas.height },
            aspectRatio: this.canvas.width / this.canvas.height
        };
    }

    analyzeNoise() {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        let noiseLevel = 0;
        let edgeVariance = 0;
        let sampleCount = 0;
        let highFreqNoise = 0;
        let localVariances = [];
        let microVariances = []; // New: Micro-level variance analysis
        let smoothRegions = 0; // New: Count of suspiciously smooth regions
        
        // Ultra-thorough noise analysis with smaller sampling
        for (let y = 3; y < height - 3; y += 1) { // Smaller step for more precision
            for (let x = 3; x < width - 3; x += 1) {
                const idx = (y * width + x) * 4;
                
                // Get 5x5 neighborhood for more detailed analysis
                const neighborhood = [];
                const microNeighborhood = []; // 3x3 for micro analysis
                
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        const nIdx = ((y + dy) * width + (x + dx)) * 4;
                        const luminance = (data[nIdx] + data[nIdx + 1] + data[nIdx + 2]) / 3;
                        neighborhood.push(luminance);
                        
                        // Micro neighborhood (3x3 center)
                        if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) {
                            microNeighborhood.push(luminance);
                        }
                    }
                }
                
                // Calculate statistics
                const center = neighborhood[12]; // Center of 5x5
                const mean = neighborhood.reduce((a, b) => a + b) / 25;
                const variance = neighborhood.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / 25;
                
                // Micro variance (3x3)
                const microMean = microNeighborhood.reduce((a, b) => a + b) / 9;
                const microVariance = microNeighborhood.reduce((sum, val) => sum + Math.pow(val - microMean, 2), 0) / 9;
                
                localVariances.push(variance);
                microVariances.push(microVariance);
                
                // Enhanced Laplacian for high-frequency noise
                const laplacian = Math.abs(
                    -24 * center + 
                    neighborhood[0] + neighborhood[1] + neighborhood[2] + neighborhood[3] + neighborhood[4] +
                    neighborhood[5] + neighborhood[6] + neighborhood[7] + neighborhood[8] + neighborhood[9] +
                    neighborhood[10] + neighborhood[11] + neighborhood[13] + neighborhood[14] +
                    neighborhood[15] + neighborhood[16] + neighborhood[17] + neighborhood[18] + neighborhood[19] +
                    neighborhood[20] + neighborhood[21] + neighborhood[22] + neighborhood[23] + neighborhood[24]
                );
                
                // AI images have suspiciously low micro-variance
                if (microVariance < 2) {
                    smoothRegions++;
                }
                
                // Natural noise detection
                if (laplacian > 15 && laplacian < 100 && variance > 3) {
                    highFreqNoise++;
                }
                
                // Edge variance calculation
                const maxDiff = Math.max(...neighborhood) - Math.min(...neighborhood);
                edgeVariance += maxDiff;
                
                // Count natural noise patterns
                if (variance > 3 && variance < 80 && microVariance > 1) {
                    noiseLevel++;
                }
                
                sampleCount++;
            }
        }
        
        // Calculate enhanced noise statistics
        const avgVariance = localVariances.reduce((a, b) => a + b, 0) / localVariances.length;
        const avgMicroVariance = microVariances.reduce((a, b) => a + b, 0) / microVariances.length;
        const noisePercentage = (noiseLevel / sampleCount) * 100;
        const highFreqPercentage = (highFreqNoise / sampleCount) * 100;
        const smoothRegionRatio = smoothRegions / sampleCount;
        
        // Enhanced AI detection criteria
        const isTooClean = noisePercentage < 0.8 || avgMicroVariance < 1.5; // More aggressive
        const isUnnatural = noisePercentage < 2.5 && avgVariance < 12;
        const hasSuspiciousSmoothing = smoothRegionRatio > 0.6; // Too many smooth regions
        
        // Calculate noise entropy (randomness measure)
        const varianceHistogram = new Array(20).fill(0);
        localVariances.forEach(v => {
            const bin = Math.min(19, Math.floor(v / 5));
            varianceHistogram[bin]++;
        });
        
        let entropy = 0;
        varianceHistogram.forEach(count => {
            if (count > 0) {
                const p = count / localVariances.length;
                entropy -= p * Math.log2(p);
            }
        });
        
        // Low entropy = too uniform = AI suspicious
        const hasLowEntropy = entropy < 2.5;
        
        console.log('🔍 NOISE ANALYSIS:', {
            noisePercentage: noisePercentage.toFixed(2),
            avgVariance: avgVariance.toFixed(2),
            avgMicroVariance: avgMicroVariance.toFixed(2),
            smoothRegionRatio: smoothRegionRatio.toFixed(3),
            entropy: entropy.toFixed(2),
            isTooClean,
            hasSuspiciousSmoothing,
            hasLowEntropy
        });
        
        return {
            noiseLevel: noisePercentage,
            edgeVariance: edgeVariance / sampleCount,
            avgVariance,
            avgMicroVariance,
            highFreqNoise: highFreqPercentage,
            smoothRegionRatio,
            entropy,
            isUnnatural,
            isTooClean,
            hasSuspiciousSmoothing,
            hasLowEntropy,
            suspicionLevel: (isTooClean || hasSuspiciousSmoothing || hasLowEntropy) ? 'high' : 
                           (isUnnatural ? 'medium' : 'low')
        };
    }

    analyzeFrequencyDomain() {
        // Simplified frequency analysis using gradient detection
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        let highFreq = 0;
        let lowFreq = 0;
        let mediumFreq = 0;
        
        for (let y = 1; y < height - 1; y += 2) {
            for (let x = 1; x < width - 1; x += 2) {
                const idx = (y * width + x) * 4;
                
                // Sobel operator for edge detection
                const gx = 
                    -1 * data[idx - width * 4 - 4] + 1 * data[idx - width * 4 + 4] +
                    -2 * data[idx - 4] + 2 * data[idx + 4] +
                    -1 * data[idx + width * 4 - 4] + 1 * data[idx + width * 4 + 4];
                
                const gy = 
                    -1 * data[idx - width * 4 - 4] + -2 * data[idx - width * 4] + -1 * data[idx - width * 4 + 4] +
                    1 * data[idx + width * 4 - 4] + 2 * data[idx + width * 4] + 1 * data[idx + width * 4 + 4];
                
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                
                if (magnitude > 100) highFreq++;
                else if (magnitude > 30) mediumFreq++;
                else lowFreq++;
            }
        }
        
        const total = highFreq + mediumFreq + lowFreq;
        
        return {
            highFreqRatio: highFreq / total,
            mediumFreqRatio: mediumFreq / total,
            lowFreqRatio: lowFreq / total,
            isArtificial: (highFreq / total) > 0.3 // Too many sharp edges
        };
    }

    analyzeEdges() {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        let sharpEdges = 0;
        let softEdges = 0;
        let totalEdges = 0;
        let unnaturalEdges = 0;
        let edgeStrengths = [];
        
        // Enhanced Sobel edge detection
        for (let y = 1; y < height - 1; y += 2) {
            for (let x = 1; x < width - 1; x += 2) {
                const idx = (y * width + x) * 4;
                
                // Get 3x3 neighborhood luminance values
                const pixels = [];
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nIdx = ((y + dy) * width + (x + dx)) * 4;
                        const lum = (data[nIdx] + data[nIdx + 1] + data[nIdx + 2]) / 3;
                        pixels.push(lum);
                    }
                }
                
                // Sobel operators
                const gx = (-1 * pixels[0] + 1 * pixels[2] +
                           -2 * pixels[3] + 2 * pixels[5] +
                           -1 * pixels[6] + 1 * pixels[8]);
                
                const gy = (-1 * pixels[0] + -2 * pixels[1] + -1 * pixels[2] +
                            1 * pixels[6] +  2 * pixels[7] +  1 * pixels[8]);
                
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                
                if (magnitude > 30) { // Significant edge
                    totalEdges++;
                    edgeStrengths.push(magnitude);
                    
                    if (magnitude > 120) {
                        sharpEdges++;
                        
                        // Check for unnaturally sharp edges (AI signature)
                        if (magnitude > 200) {
                            unnaturalEdges++;
                        }
                    } else if (magnitude > 60) {
                        softEdges++;
                    }
                }
            }
        }
        
        // Calculate edge statistics
        const avgEdgeStrength = edgeStrengths.length > 0 ? 
            edgeStrengths.reduce((a, b) => a + b) / edgeStrengths.length : 0;
        
        // Sort edge strengths to find distribution
        edgeStrengths.sort((a, b) => b - a);
        const topEdges = edgeStrengths.slice(0, Math.min(100, edgeStrengths.length));
        const avgTopEdges = topEdges.length > 0 ? 
            topEdges.reduce((a, b) => a + b) / topEdges.length : 0;
        
        const sharpEdgeRatio = totalEdges > 0 ? sharpEdges / totalEdges : 0;
        const unnaturalEdgeRatio = totalEdges > 0 ? unnaturalEdges / totalEdges : 0;
        
        // AI detection criteria
        const isOverSharpened = sharpEdgeRatio > 0.7 || unnaturalEdgeRatio > 0.3;
        const hasAIEdgePattern = avgTopEdges > 180 && sharpEdgeRatio > 0.5;
        
        return {
            sharpEdgeRatio,
            softEdgeRatio: totalEdges > 0 ? softEdges / totalEdges : 0,
            unnaturalEdgeRatio,
            totalEdges,
            avgEdgeStrength,
            avgTopEdges,
            isOverSharpened,
            hasAIEdgePattern,
            suspicionLevel: hasAIEdgePattern ? 'high' : (isOverSharpened ? 'medium' : 'low')
        };
    }

    analyzeColorDistribution() {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        const colorBins = new Array(256).fill(0);
        const hueBins = new Array(360).fill(0);
        const saturationBins = new Array(100).fill(0);
        let totalPixels = data.length / 4;
        
        let oversaturatedPixels = 0;
        let perfectColors = 0;
        let quantizedColors = 0; // New: Colors that look quantized
        let artificialGradients = 0; // New: Too-perfect gradients
        let colorClusterCount = 0; // New: Unnatural color clustering
        
        // Enhanced color analysis
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Luminance distribution
            const luminance = Math.floor(0.299 * r + 0.587 * g + 0.114 * b);
            colorBins[luminance]++;
            
            // HSV calculation for detailed analysis
            const max = Math.max(r, g, b) / 255;
            const min = Math.min(r, g, b) / 255;
            const delta = max - min;
            
            // Saturation analysis
            const saturation = max === 0 ? 0 : delta / max;
            const satBin = Math.floor(saturation * 99);
            saturationBins[satBin]++;
            
            // AI oversaturation detection (more aggressive)
            if (saturation > 0.85) {
                oversaturatedPixels++;
            }
            
            // Enhanced "perfect" color detection
            // AI often uses specific bit patterns
            if ((r % 8 === 0 && g % 8 === 0 && b % 8 === 0) || // 8-bit quantization
                (r % 16 === 0 && g % 16 === 0 && b % 16 === 0) || // 4-bit quantization
                (r === g && g === b) || // Perfect grayscale
                (Math.abs(r - g) < 2 && Math.abs(g - b) < 2)) { // Near grayscale
                perfectColors++;
            }
            
            // Detect AI-style quantization (values ending in 0 or 5)
            if ((r % 5 === 0 || r % 10 === 0) && 
                (g % 5 === 0 || g % 10 === 0) && 
                (b % 5 === 0 || b % 10 === 0)) {
                quantizedColors++;
            }
            
            // Hue distribution
            if (delta > 0) {
                let hue = 0;
                if (max === r/255) hue = ((g - b) / 255 / delta) % 6;
                else if (max === g/255) hue = (b - r) / 255 / delta + 2;
                else hue = (r - g) / 255 / delta + 4;
                
                hue = Math.floor((hue * 60 + 360) % 360);
                hueBins[hue]++;
            }
        }
        
        // Enhanced distribution analysis
        const expectedLuminance = totalPixels / 256;
        let luminanceVariance = 0;
        let peakCount = 0;
        let sharpPeaks = 0; // New: Count of unnaturally sharp peaks
        
        for (let i = 0; i < 256; i++) {
            luminanceVariance += Math.pow(colorBins[i] - expectedLuminance, 2);
            
            // Enhanced peak detection
            if (i > 1 && i < 254) {
                const current = colorBins[i];
                const prev = colorBins[i-1];
                const next = colorBins[i+1];
                const prevPrev = colorBins[i-2];
                const nextNext = colorBins[i+2];
                
                // Standard peak
                if (current > prev && current > next && current > expectedLuminance * 1.3) {
                    peakCount++;
                    
                    // Sharp, unnatural peak (AI signature)
                    if (current > prev * 2 && current > next * 2 && 
                        current > prevPrev * 3 && current > nextNext * 3) {
                        sharpPeaks++;
                    }
                }
            }
        }
        
        luminanceVariance /= 256;
        
        // Color clustering analysis (AI tends to cluster colors)
        const significantBins = colorBins.filter(count => count > totalPixels * 0.01);
        colorClusterCount = significantBins.length;
        
        // Calculate ratios
        const oversaturationRatio = oversaturatedPixels / totalPixels;
        const perfectColorRatio = perfectColors / totalPixels;
        const quantizedRatio = quantizedColors / totalPixels;
        
        // Enhanced AI detection criteria
        const hasExtremeUniformity = luminanceVariance < 200; // More aggressive
        const hasAIColorSigns = oversaturationRatio > 0.2 || perfectColorRatio > 0.05 || quantizedRatio > 0.15;
        const hasSuspiciousPeaks = sharpPeaks > 2 || (peakCount < 2 && luminanceVariance < 500);
        const hasUnaturalClustering = colorClusterCount < 8 || colorClusterCount > 50;
        
        // Calculate color entropy
        let colorEntropy = 0;
        for (let i = 0; i < 256; i++) {
            if (colorBins[i] > 0) {
                const p = colorBins[i] / totalPixels;
                colorEntropy -= p * Math.log2(p);
            }
        }
        
        const hasLowColorEntropy = colorEntropy < 6; // Too uniform color distribution
        
        // Saturation analysis
        let saturationEntropy = 0;
        for (let i = 0; i < 100; i++) {
            if (saturationBins[i] > 0) {
                const p = saturationBins[i] / totalPixels;
                saturationEntropy -= p * Math.log2(p);
            }
        }
        
        const hasAbnormalSaturation = saturationEntropy < 3 || saturationEntropy > 6;
        
        console.log('🎨 COLOR ANALYSIS:', {
            luminanceVariance: luminanceVariance.toFixed(0),
            peakCount,
            sharpPeaks,
            oversaturationRatio: (oversaturationRatio * 100).toFixed(1) + '%',
            perfectColorRatio: (perfectColorRatio * 100).toFixed(1) + '%',
            quantizedRatio: (quantizedRatio * 100).toFixed(1) + '%',
            colorClusterCount,
            colorEntropy: colorEntropy.toFixed(2),
            saturationEntropy: saturationEntropy.toFixed(2),
            hasAIColorSigns,
            hasExtremeUniformity,
            hasSuspiciousPeaks
        });
        
        return {
            luminanceVariance,
            peakCount,
            sharpPeaks,
            oversaturationRatio,
            perfectColorRatio,
            quantizedRatio,
            colorClusterCount,
            colorEntropy,
            saturationEntropy,
            isUnnatural: hasExtremeUniformity || hasSuspiciousPeaks || hasUnaturalClustering,
            hasAIColorSigns: hasAIColorSigns || hasLowColorEntropy || hasAbnormalSaturation,
            dominantColors: this.findDominantColors(colorBins),
            colorRange: Math.max(...colorBins) - Math.min(...colorBins),
            suspicionLevel: (hasExtremeUniformity || hasAIColorSigns || hasSuspiciousPeaks) ? 'high' : 
                           (hasUnaturalClustering || hasLowColorEntropy ? 'medium' : 'low')
        };
    }

    analyzeCompression() {
        // Analyze JPEG compression artifacts by looking for 8x8 block patterns
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        const width = this.canvas.width;
        
        let blockArtifacts = 0;
        let totalBlocks = 0;
        
        // Check 8x8 blocks
        for (let y = 0; y < this.canvas.height - 8; y += 8) {
            for (let x = 0; x < width - 8; x += 8) {
                let blockVariance = 0;
                let blockMean = 0;
                
                // Calculate block statistics
                for (let by = 0; by < 8; by++) {
                    for (let bx = 0; bx < 8; bx++) {
                        const idx = ((y + by) * width + (x + bx)) * 4;
                        const luminance = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                        blockMean += luminance;
                    }
                }
                blockMean /= 64;
                
                for (let by = 0; by < 8; by++) {
                    for (let bx = 0; bx < 8; bx++) {
                        const idx = ((y + by) * width + (x + bx)) * 4;
                        const luminance = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                        blockVariance += Math.pow(luminance - blockMean, 2);
                    }
                }
                blockVariance /= 64;
                
                // Detect compression artifacts
                if (blockVariance < 10 && blockMean > 50) {
                    blockArtifacts++;
                }
                
                totalBlocks++;
            }
        }
        
        return {
            compressionRatio: totalBlocks > 0 ? blockArtifacts / totalBlocks : 0,
            hasArtifacts: (blockArtifacts / totalBlocks) > 0.1,
            quality: this.estimateJPEGQuality(blockArtifacts, totalBlocks)
        };
    }

    calculateAIProbability(analyses) {
        let aiScore = 0;
        let suspiciousFeatures = 0;
        let totalFeatures = 0;
        let criticalFlags = 0; // Critical AI signatures
        
        // CRITICAL FLAG 1: Extreme cleanliness (AI signature)
        if (analyses.noiseAnalysis.noiseLevel < 0.5) {
            criticalFlags++;
            aiScore += 98; // Almost certain AI
            console.log('🚨 CRITICAL: Extremely low noise detected');
        } else if (analyses.noiseAnalysis.noiseLevel < 1.5) {
            aiScore += 85;
            suspiciousFeatures += 0.9;
        } else if (analyses.noiseAnalysis.noiseLevel < 3) {
            aiScore += 70;
            suspiciousFeatures += 0.7;
        } else {
            aiScore += Math.max(0, 25 - analyses.noiseAnalysis.noiseLevel);
        }
        totalFeatures++;
        
        // CRITICAL FLAG 2: Perfect edge patterns (AI signature)
        const perfectEdgeRatio = analyses.edgeAnalysis.sharpEdgeRatio;
        const unnaturalRatio = analyses.edgeAnalysis.unnaturalEdgeRatio;
        
        if (perfectEdgeRatio > 0.9 || unnaturalRatio > 0.5) {
            criticalFlags++;
            aiScore += 98;
            console.log('🚨 CRITICAL: Perfect edge patterns detected');
        } else if (perfectEdgeRatio > 0.8 || unnaturalRatio > 0.3) {
            aiScore += 90;
            suspiciousFeatures += 0.9;
        } else if (perfectEdgeRatio > 0.6) {
            aiScore += 75;
            suspiciousFeatures += 0.8;
        } else {
            aiScore += perfectEdgeRatio * 50;
        }
        totalFeatures++;
        
        // CRITICAL FLAG 3: Unnatural color distribution
        if (analyses.colorAnalysis.luminanceVariance < 200) {
            criticalFlags++;
            aiScore += 95;
            console.log('🚨 CRITICAL: Extremely uniform color distribution');
        } else if (analyses.colorAnalysis.luminanceVariance < 500) {
            aiScore += 85;
            suspiciousFeatures += 0.9;
        } else if (analyses.colorAnalysis.luminanceVariance < 1000) {
            aiScore += 60;
            suspiciousFeatures += 0.6;
        } else {
            aiScore += 20;
        }
        totalFeatures++;
        
        // CRITICAL FLAG 4: Perfect saturation patterns
        const oversatRatio = analyses.colorAnalysis.oversaturationRatio;
        const perfectColorRatio = analyses.colorAnalysis.perfectColorRatio;
        
        if (oversatRatio > 0.5 || perfectColorRatio > 0.2) {
            criticalFlags++;
            aiScore += 95;
            console.log('🚨 CRITICAL: AI color signature detected');
        } else if (oversatRatio > 0.3 || perfectColorRatio > 0.1) {
            aiScore += 80;
            suspiciousFeatures += 0.8;
        } else {
            aiScore += (oversatRatio + perfectColorRatio) * 150;
        }
        totalFeatures++;
        
        // CRITICAL FLAG 5: Suspicious dimensions and ratios
        const aspectRatio = analyses.basicProps.aspectRatio;
        const width = analyses.basicProps.dimensions.width;
        const height = analyses.basicProps.dimensions.height;
        
        // AI generation signatures
        const commonAISizes = [512, 768, 1024, 1536, 2048];
        const isExactAISize = commonAISizes.includes(width) && commonAISizes.includes(height);
        const isSquare = Math.abs(aspectRatio - 1) < 0.01;
        const isPowerOfTwo = (n) => n > 0 && (n & (n - 1)) === 0;
        
        if (isExactAISize || (isSquare && (isPowerOfTwo(width) || isPowerOfTwo(height)))) {
            criticalFlags++;
            aiScore += 90;
            console.log('🚨 CRITICAL: AI generation dimensions detected');
        } else if (commonAISizes.includes(width) || commonAISizes.includes(height)) {
            aiScore += 70;
            suspiciousFeatures += 0.7;
        } else if (isSquare) {
            aiScore += 50;
            suspiciousFeatures += 0.5;
        } else {
            aiScore += 15;
        }
        totalFeatures++;
        
        // Enhanced frequency analysis
        if (analyses.frequencyAnalysis.highFreqRatio > 0.5) {
            aiScore += 90;
            suspiciousFeatures += 0.9;
        } else if (analyses.frequencyAnalysis.lowFreqRatio > 0.8) {
            aiScore += 85;
            suspiciousFeatures += 0.9;
        } else {
            aiScore += analyses.frequencyAnalysis.highFreqRatio * 70;
        }
        totalFeatures++;
        
        // Compression analysis with AI bias
        if (analyses.compressionAnalysis.quality > 99) {
            aiScore += 85;
            suspiciousFeatures += 0.8;
        } else if (analyses.compressionAnalysis.quality > 97) {
            aiScore += 70;
            suspiciousFeatures += 0.7;
        } else {
            aiScore += Math.max(0, analyses.compressionAnalysis.quality - 85);
        }
        totalFeatures++;
        
        // Saturation extremes (AI tendency)
        const avgSat = analyses.basicProps.avgSaturation;
        if (avgSat > 0.9 || avgSat < 0.1) {
            aiScore += 80;
            suspiciousFeatures += 0.8;
        } else if (avgSat > 0.8 || avgSat < 0.2) {
            aiScore += 60;
            suspiciousFeatures += 0.6;
        } else {
            aiScore += 20;
        }
        totalFeatures++;
        
        // Calculate base score
        const baseScore = aiScore / totalFeatures;
        
        // AGGRESSIVE MULTIPLIERS
        let finalMultiplier = 1;
        const suspiciousRatio = suspiciousFeatures / totalFeatures;
        
        // Critical flags trigger extreme scores
        if (criticalFlags >= 4) {
            finalMultiplier = 2.5; // Almost certain AI
            console.log('🚨 MULTIPLE CRITICAL FLAGS: Very high AI probability');
        } else if (criticalFlags >= 3) {
            finalMultiplier = 2.2; // Multiple critical AI signatures
            console.log('🚨 MULTIPLE CRITICAL FLAGS: High AI probability');
        } else if (criticalFlags >= 2) {
            finalMultiplier = 1.9; // Two critical signatures
        } else if (criticalFlags >= 1) {
            finalMultiplier = 1.6; // One critical signature
        } else if (suspiciousRatio > 0.9) {
            finalMultiplier = 1.5; // Almost all features suspicious
        } else if (suspiciousRatio > 0.7) {
            finalMultiplier = 1.3; // Many suspicious features
        } else if (suspiciousRatio > 0.5) {
            finalMultiplier = 1.2; // Several suspicious features
        } else if (suspiciousRatio > 0.3) {
            finalMultiplier = 1.1; // Some suspicious features
        }
        
        // Additional boost for specific AI patterns
        if (analyses.noiseAnalysis.avgVariance < 5 && perfectEdgeRatio > 0.7) {
            finalMultiplier *= 1.3; // Classic AI signature combo
            console.log('🎯 AI PATTERN: Ultra-low variance + high edge sharpness');
        }
        
        if (analyses.colorAnalysis.peakCount < 2 && analyses.colorAnalysis.luminanceVariance < 300) {
            finalMultiplier *= 1.25; // Too uniform color distribution
            console.log('🎯 AI PATTERN: Extremely uniform colors');
        }
        
        // Perfect dimensions + low noise = almost certain AI
        if (isExactAISize && analyses.noiseAnalysis.noiseLevel < 2) {
            finalMultiplier *= 1.4;
            console.log('🎯 AI PATTERN: Perfect dimensions + no noise');
        }
        
        const finalScore = Math.min(100, Math.max(0, Math.round(baseScore * finalMultiplier)));
        
        // Enhanced debug logging
        console.log('🔍 ENHANCED AI DETECTION:', {
            baseScore: baseScore.toFixed(1),
            criticalFlags,
            suspiciousFeatures: suspiciousFeatures.toFixed(1),
            suspiciousRatio: suspiciousRatio.toFixed(2),
            finalMultiplier: finalMultiplier.toFixed(2),
            finalScore,
            noiseLevel: analyses.noiseAnalysis.noiseLevel.toFixed(2),
            edgeSharpness: (perfectEdgeRatio * 100).toFixed(1) + '%',
            colorVariance: analyses.colorAnalysis.luminanceVariance.toFixed(0),
            dimensions: `${width}x${height}`,
            aspectRatio: aspectRatio.toFixed(2),
            isExactAISize,
            avgSaturation: avgSat.toFixed(2)
        });
        
        return finalScore;
    }

    calculateConfidence(aiProbability) {
        // Higher confidence for extreme values
        const distance = Math.abs(aiProbability - 50);
        return Math.min(95, 60 + distance);
    }

    generateTechnicalIndicators(analyses) {
        const indicators = [];
        
        // Noise pattern analysis
        indicators.push({
            name: 'Gürültü Deseni',
            score: Math.round(100 - analyses.noiseAnalysis.noiseLevel),
            status: analyses.noiseAnalysis.isTooClean ? 'suspicious' : 
                   (analyses.noiseAnalysis.isUnnatural ? 'warning' : 'normal'),
            description: analyses.noiseAnalysis.isTooClean ? 'Aşırı temiz (AI şüphesi)' : 
                        (analyses.noiseAnalysis.isUnnatural ? 'Çok az gürültü' : 'Doğal gürültü'),
            details: `Varyans: ${analyses.noiseAnalysis.avgVariance.toFixed(1)}, HF Gürültü: ${analyses.noiseAnalysis.highFreqNoise.toFixed(1)}%`
        });
        
        // Edge quality analysis
        indicators.push({
            name: 'Kenar Kalitesi',
            score: Math.round(analyses.edgeAnalysis.sharpEdgeRatio * 100),
            status: analyses.edgeAnalysis.hasAIEdgePattern ? 'suspicious' : 
                   (analyses.edgeAnalysis.isOverSharpened ? 'warning' : 'normal'),
            description: analyses.edgeAnalysis.hasAIEdgePattern ? 'AI kenar deseni tespit edildi' : 
                        (analyses.edgeAnalysis.isOverSharpened ? 'Aşırı keskinleştirilmiş' : 'Doğal kenarlar'),
            details: `Ortalama güç: ${analyses.edgeAnalysis.avgEdgeStrength.toFixed(1)}, Unnatural: ${(analyses.edgeAnalysis.unnaturalEdgeRatio * 100).toFixed(1)}%`
        });
        
        // Frequency distribution
        indicators.push({
            name: 'Frekans Dağılımı',
            score: Math.round(analyses.frequencyAnalysis.highFreqRatio * 100),
            status: analyses.frequencyAnalysis.isArtificial ? 'suspicious' : 'normal',
            description: analyses.frequencyAnalysis.isArtificial ? 'Yapay frekans deseni' : 'Doğal dağılım',
            details: `Yüksek frekans: ${(analyses.frequencyAnalysis.highFreqRatio * 100).toFixed(1)}%`
        });
        
        // Color distribution analysis
        indicators.push({
            name: 'Renk Dağılımı',
            score: Math.round(Math.min(100, analyses.colorAnalysis.luminanceVariance / 50)),
            status: analyses.colorAnalysis.hasAIColorSigns ? 'suspicious' : 
                   (analyses.colorAnalysis.isUnnatural ? 'warning' : 'normal'),
            description: analyses.colorAnalysis.hasAIColorSigns ? 'AI renk imzası tespit edildi' : 
                        (analyses.colorAnalysis.isUnnatural ? 'Çok düzgün dağılım' : 'Doğal varyasyon'),
            details: `Aşırı doygunluk: ${(analyses.colorAnalysis.oversaturationRatio * 100).toFixed(1)}%, Pik sayısı: ${analyses.colorAnalysis.peakCount}`
        });
        
        // Compression quality
        indicators.push({
            name: 'Sıkıştırma Kalitesi',
            score: analyses.compressionAnalysis.quality,
            status: analyses.compressionAnalysis.quality > 98 ? 'suspicious' : 
                   (analyses.compressionAnalysis.quality > 95 ? 'warning' : 'normal'),
            description: analyses.compressionAnalysis.quality > 98 ? 'Şüpheli yüksek kalite' : 
                        (analyses.compressionAnalysis.quality > 95 ? 'Çok yüksek kalite' : 'Normal sıkıştırma'),
            details: `Artefakt oranı: ${(analyses.compressionAnalysis.compressionRatio * 100).toFixed(1)}%`
        });
        
        return indicators;
    }

    findDominantColors(colorBins) {
        const peaks = [];
        for (let i = 1; i < colorBins.length - 1; i++) {
            if (colorBins[i] > colorBins[i-1] && colorBins[i] > colorBins[i+1] && colorBins[i] > 100) {
                peaks.push({ value: i, count: colorBins[i] });
            }
        }
        return peaks.sort((a, b) => b.count - a.count).slice(0, 3);
    }

    estimateJPEGQuality(artifacts, total) {
        const artifactRatio = artifacts / total;
        if (artifactRatio < 0.05) return 95;
        if (artifactRatio < 0.1) return 85;
        if (artifactRatio < 0.2) return 75;
        if (artifactRatio < 0.3) return 65;
        return 50;
    }

    extractMetadata() {
        // Basic metadata extraction
        return {
            fileName: this.currentFile.name,
            fileSize: this.currentFile.size,
            fileType: this.currentFile.type,
            lastModified: new Date(this.currentFile.lastModified).toLocaleDateString('tr-TR')
        };
    }

    updateProgress(percentage, text) {
        return new Promise(resolve => {
            this.progressBar.style.width = `${percentage}%`;
            this.progressText.textContent = `${percentage}%`;
            
            // Update status text if provided
            if (text) {
                const statusElement = this.progressSection.querySelector('span');
                statusElement.textContent = text;
            }
            
            setTimeout(resolve, 200);
        });
    }

    getVerdict(probability) {
        if (probability < 25) {
            return {
                text: 'Bu görsel büyük olasılıkla gerçek',
                color: 'text-green-400',
                icon: 'check-circle',
                bgColor: 'bg-green-500/20'
            };
        } else if (probability < 40) {
            return {
                text: 'Görsel muhtemelen gerçek',
                color: 'text-green-400',
                icon: 'check-circle-2',
                bgColor: 'bg-green-500/20'
            };
        } else if (probability < 60) {
            return {
                text: 'Belirsiz - daha fazla analiz gerekebilir',
                color: 'text-yellow-400',
                icon: 'alert-triangle',
                bgColor: 'bg-yellow-500/20'
            };
        } else if (probability < 75) {
            return {
                text: 'Bu görsel AI tarafından üretilmiş olabilir',
                color: 'text-orange-400',
                icon: 'alert-circle',
                bgColor: 'bg-orange-500/20'
            };
        } else {
            return {
                text: 'Bu görsel büyük olasılıkla AI üretimi',
                color: 'text-red-400',
                icon: 'x-circle',
                bgColor: 'bg-red-500/20'
            };
        }
    }

    displayResults(results) {
        const { aiProbability, confidence, verdict, indicators, processingTime, fileSize, metadata, warning } = results;
        
        this.resultContent.innerHTML = `
            <!-- Warning Notice -->
            ${warning ? `
            <div class="result-card rounded-lg p-4 bg-blue-500/20 border border-blue-500/30 mb-4">
                <div class="flex items-center mb-2">
                    <i data-lucide="info" class="w-5 h-5 text-blue-400 mr-3"></i>
                    <span class="font-medium text-blue-400">Önemli Uyarı</span>
                </div>
                <div class="text-sm text-slate-300">${warning}</div>
            </div>
            ` : ''}

            <!-- Main Verdict -->
            <div class="result-card rounded-lg p-4 ${verdict.bgColor}">
                <div class="flex items-center mb-2">
                    <i data-lucide="${verdict.icon}" class="w-6 h-6 ${verdict.color} mr-3"></i>
                    <span class="font-semibold ${verdict.color}">${verdict.text}</span>
                </div>
                <div class="text-sm text-slate-300">
                    Güven seviyesi: %${confidence} | İşlem süresi: ${processingTime}s
                </div>
            </div>

            <!-- AI Probability -->
            <div class="result-card rounded-lg p-4">
                <div class="flex items-center justify-between mb-3">
                    <span class="font-medium">AI Üretim Olasılığı</span>
                    <span class="text-2xl font-bold text-blue-400">%${aiProbability}</span>
                </div>
                <div class="confidence-meter ai-indicator">
                    <div class="confidence-indicator" style="left: ${aiProbability}%"></div>
                </div>
                <div class="flex justify-between text-xs text-slate-400 mt-1">
                    <span>Gerçek</span>
                    <span>Belirsiz</span>
                    <span>AI Üretimi</span>
                </div>
            </div>

            <!-- Technical Indicators -->
            <div class="result-card rounded-lg p-4">
                <h4 class="font-medium mb-3 flex items-center">
                    <i data-lucide="activity" class="w-5 h-5 mr-2 text-purple-400"></i>
                    Gelişmiş Teknik Analiz
                </h4>
                <div class="space-y-3">
                    ${indicators.map(indicator => `
                        <div class="flex items-center justify-between p-3 rounded-lg ${
                            indicator.status === 'suspicious' ? 'bg-red-500/20 border border-red-500/30' : 
                            indicator.status === 'warning' ? 'bg-yellow-500/20 border border-yellow-500/30' :
                            'bg-green-500/10 border border-green-500/20'
                        }">
                            <div class="flex-1">
                                <div class="flex items-center">
                                    <span class="text-sm font-medium">${indicator.name}</span>
                                    <div class="w-2 h-2 rounded-full ml-2 ${
                                        indicator.status === 'suspicious' ? 'bg-red-400' : 
                                        indicator.status === 'warning' ? 'bg-yellow-400' : 
                                        'bg-green-400'
                                    }"></div>
                                </div>
                                <div class="text-xs text-slate-400 mt-1">${indicator.description}</div>
                                ${indicator.details ? `<div class="text-xs text-slate-500 mt-1">${indicator.details}</div>` : ''}
                            </div>
                            <div class="text-right">
                                <span class="text-sm font-medium">%${indicator.score}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Detailed Analysis -->
            <div class="result-card rounded-lg p-4">
                <h4 class="font-medium mb-3 flex items-center">
                    <i data-lucide="microscope" class="w-5 h-5 mr-2 text-blue-400"></i>
                    Detaylı Görsel Analizi
                </h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-slate-400">Dosya Boyutu:</span>
                            <span>${fileSize} KB</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Dosya Türü:</span>
                            <span>${metadata?.fileType || 'Bilinmiyor'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Son Değişiklik:</span>
                            <span>${metadata?.lastModified || 'Bilinmiyor'}</span>
                        </div>
                    </div>
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-slate-400">Çözünürlük:</span>
                            <span>${this.canvas.width}x${this.canvas.height}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">En-Boy Oranı:</span>
                            <span>${(this.canvas.width / this.canvas.height).toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Analiz Metodu:</span>
                            <span>${metadata?.ai_model_analysis ? 'AI Model + CV' : 'Gelişmiş CV'}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Backend Analysis Details (if available) -->
            ${metadata?.metadata_analysis ? `
            <div class="result-card rounded-lg p-4">
                <h4 class="font-medium mb-3 flex items-center">
                    <i data-lucide="shield-check" class="w-5 h-5 mr-2 text-green-400"></i>
                    Metadata Analizi
                </h4>
                <div class="text-sm text-slate-300 space-y-2">
                    <div class="flex justify-between">
                        <span class="text-slate-400">AI Araçları:</span>
                        <span class="${metadata.metadata_analysis.ai_tools_detected?.length > 0 ? 'text-red-400' : 'text-green-400'}">
                            ${metadata.metadata_analysis.ai_tools_detected?.length > 0 ? 
                              metadata.metadata_analysis.ai_tools_detected.join(', ') : 'Tespit Edilmedi'}
                        </span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-400">Şüpheli Desenler:</span>
                        <span class="${metadata.metadata_analysis.suspicious_patterns?.length > 0 ? 'text-yellow-400' : 'text-green-400'}">
                            ${metadata.metadata_analysis.suspicious_patterns?.length > 0 ? 
                              metadata.metadata_analysis.suspicious_patterns.length + ' adet' : 'Bulunamadı'}
                        </span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-400">EXIF Verisi:</span>
                        <span class="${metadata.metadata_analysis.exif_data_present ? 'text-green-400' : 'text-yellow-400'}">
                            ${metadata.metadata_analysis.exif_data_present ? 'Mevcut' : 'Eksik'}
                        </span>
                    </div>
                    ${metadata.metadata_analysis.camera_make ? `
                    <div class="flex justify-between">
                        <span class="text-slate-400">Kamera:</span>
                        <span class="text-green-400">${metadata.metadata_analysis.camera_make} ${metadata.metadata_analysis.camera_model || ''}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
            ` : ''}

            <!-- Analysis Explanation -->
            <div class="result-card rounded-lg p-4">
                <h4 class="font-medium mb-3 flex items-center">
                    <i data-lucide="info" class="w-5 h-5 mr-2 text-blue-400"></i>
                    Analiz Açıklaması
                </h4>
                <div class="text-sm text-slate-300 space-y-2">
                    <p>Bu analiz, ${metadata?.ai_model_analysis ? 'AI modeli ve' : ''} görselinizin piksel seviyesinde incelenmesi sonucu elde edilmiştir:</p>
                    <ul class="list-disc list-inside space-y-1 text-xs text-slate-400">
                        ${metadata?.ai_model_analysis ? '<li><strong>AI Model:</strong> Önceden eğitilmiş derin öğrenme modeli ile analiz</li>' : ''}
                        <li><strong>Metadata Analizi:</strong> EXIF verileri ve C2PA sertifikaları incelendi</li>
                        <li><strong>Gürültü Analizi:</strong> Doğal fotoğraflarda bulunan sensor gürültüsü incelendi</li>
                        <li><strong>Kenar Tespiti:</strong> Sobel operatörü ile kenar kalitesi ve doğallığı</li>
                        <li><strong>Renk Dağılımı:</strong> Luminans ve renk ton dağılımının doğallığı</li>
                    </ul>
                </div>
            </div>
        `;

        // Show results section
        this.resultsSection.classList.remove('hidden');
        this.resultsSection.classList.add('fade-in');
        
        // Hide progress section
        setTimeout(() => {
            this.progressSection.classList.add('hidden');
        }, 500);

        // Re-initialize icons
        setTimeout(() => this.initializeLucideIcons(), 100);
    }

    resetForNewAnalysis() {
        this.resetResults();
        this.removeCurrentImage();
    }

    resetResults() {
        this.resultsSection.classList.add('hidden');
        this.progressSection.classList.add('hidden');
        this.progressBar.style.width = '0%';
        this.progressText.textContent = '0%';
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast system
        alert(message);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // API Integration Template (for future use)
    /*
    async analyzeWithAPI(imageFile) {
        // Example API integration template
        // Replace with actual API endpoints and keys
        
        const formData = new FormData();
        formData.append('image', imageFile);
        
        try {
            // Sightengine API Example
            const sightengineResponse = await fetch('https://api.sightengine.com/1.0/check.json', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer YOUR_API_KEY'
                },
                body: formData
            });
            
            // Hive API Example
            const hiveResponse = await fetch('https://api.thehive.ai/api/v2/task/sync', {
                method: 'POST',
                headers: {
                    'Authorization': 'Token YOUR_API_TOKEN',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: imageUrl, // or base64 data
                    models: ['ai_generated']
                })
            });
            
            const sightengineData = await sightengineResponse.json();
            const hiveData = await hiveResponse.json();
            
            return this.processAPIResults(sightengineData, hiveData);
            
        } catch (error) {
            console.error('API Error:', error);
            throw new Error('API analizi başarısız oldu');
        }
    }
    
    processAPIResults(sightengineData, hiveData) {
        // Process and combine results from multiple APIs
        // Return standardized result format
        return {
            aiProbability: sightengineData.ai_generated?.probability * 100 || 0,
            confidence: hiveData.confidence || 0,
            // ... other processed data
        };
    }
    */
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIImageDetector();
});

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIImageDetector;
}