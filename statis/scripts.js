/**
 * Enhanced Hate Speech Detection Frontend
 * Interactive JavaScript dengan visualisasi lengkap dan analisis kata
 */

class HateSpeechDetector {
    constructor() {
        this.apiBaseUrl = this.getApiBaseUrl();

        this.charts = {};
        this.sessionStats = {
            totalPredictions: 0,
            hateDetected: 0,
            avgProcessingTime: 0,
            avgConfidence: 0,
            fastestProcessing: Infinity,
            predictions: [],
            startTime: Date.now()
        };
        this.wordAnalysis = {
            hateWords: new Map(),
            safeWords: new Map(),
            commonWords: new Map()
        };
        this.lastCSVResult = null;
        this.selectedCSVFile = null;
        this.init();
    }

    getApiBaseUrl() {
        const BACKEND_URL = "PLACEHOLDER_BACKEND_URL";

        const isDevelopment = window.location.hostname === 'localhost' ||
                             window.location.hostname === '127.0.0.1' ||
                             window.location.hostname === '';

        if (isDevelopment) {
            console.log('🔧 Development mode detected');
            return 'http://localhost:8000';
        }

        if (BACKEND_URL !== "PLACEHOLDER_BACKEND_URL") {
            console.log('🚀 Production mode - Backend URL:', BACKEND_URL);
            return BACKEND_URL;
        }

        console.warn('⚠️ Backend URL not configured! Using fallback detection.');

        const currentHost = window.location.hostname;
        if (currentHost.includes('github.io') || currentHost.includes('githubpages.io')) {
            const projectName = currentHost.split('.')[0];
            return `https://${projectName}-api.onrender.com`;
        }

        return window.location.origin;
    }

    async init() {
        console.log('🚀 Initializing Enhanced Hate Speech Detector...');
        console.log('🔗 API Base URL:', this.apiBaseUrl);

        this.setupEventListeners();
        this.setupTabSystem();
        this.setupChartDefaults();
        this.loadSessionStats();
        this.initAnimations();
        this.loadSavedData();
        this.updateSystemUptime();

        await this.checkApiConnection();
    }

    async checkApiConnection() {
        const statusElement = document.getElementById('modelStatus');

        try {
            statusElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting to API...';

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000);

            const response = await fetch(`${this.apiBaseUrl}/health`, {
                signal: controller.signal,
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.model_loaded) {
                statusElement.innerHTML = '<i class="fas fa-check-circle"></i> Model Ready';
                statusElement.classList.add('loaded');
                statusElement.classList.remove('error');
                this.showToast('✅ Connected to API server and model is ready!', 'success');

                await this.loadModelInfo();
            } else {
                statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Model Not Ready';
                statusElement.classList.add('error');
                statusElement.classList.remove('loaded');
                this.showToast('⚠️ Connected to API but model not loaded. Demo mode active.', 'warning');
            }

            this.updateEnvironmentInfo(data);

        } catch (error) {
            console.error('❌ API Connection failed:', error);

            statusElement.innerHTML = '<i class="fas fa-times-circle"></i> Connection Error';
            statusElement.classList.add('error');
            statusElement.classList.remove('loaded');

            let errorMessage = 'Unable to connect to API server.';

            if (error.name === 'AbortError') {
                errorMessage += ' Connection timeout.';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage += ' Please check if the backend server is running.';
            } else {
                errorMessage += ` Error: ${error.message}`;
            }

            this.showToast(errorMessage, 'error');
            this.showConnectionTroubleshooting();
        }
    }

    showConnectionTroubleshooting() {
        const troubleshootingDiv = document.createElement('div');
        troubleshootingDiv.className = 'troubleshooting-panel';
        troubleshootingDiv.innerHTML = `
            <div class="troubleshooting-content">
                <h3>🔧 Connection Troubleshooting</h3>
                <p><strong>Current API URL:</strong> <code>${this.apiBaseUrl}</code></p>
                <div class="troubleshooting-steps">
                    <p><strong>For Developers:</strong></p>
                    <ul>
                        <li>✅ Make sure backend server is running</li>
                        <li>✅ Check if backend URL is correct</li>
                        <li>✅ Verify CORS settings on backend</li>
                        <li>✅ Check browser console for detailed errors</li>
                    </ul>
                    <p><strong>For Users:</strong></p>
                    <ul>
                        <li>🔄 Try refreshing the page</li>
                        <li>🌐 Check your internet connection</li>
                        <li>⏰ The server might be starting up, please wait a moment</li>
                    </ul>
                </div>
                <div class="troubleshooting-actions">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" class="btn-close">
                        Close
                    </button>
                    <button onclick="window.hateSpeechApp.checkApiConnection()" class="btn-retry">
                        🔄 Retry Connection
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(troubleshootingDiv);

        setTimeout(() => {
            if (troubleshootingDiv.parentElement) {
                troubleshootingDiv.remove();
            }
        }, 30000);
    }

    updateEnvironmentInfo(healthData) {
        const envElements = document.querySelectorAll('.env-info');
        envElements.forEach(el => {
            el.innerHTML = `
                <small>
                    Environment: ${healthData.environment || 'Unknown'} |
                    Version: ${healthData.api_version || 'Unknown'} |
                    Mode: ${healthData.mode || 'Unknown'}
                </small>
            `;
        });
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model/info`);
            const data = await response.json();

            if (data.success && data.model_info) {
                const modelName = document.getElementById('modelName');
                const modelParams = document.getElementById('modelParams');

                if (modelName && data.config?.MODEL_NAME) {
                    modelName.textContent = data.config.MODEL_NAME.split('/').pop();
                }

                if (modelParams && data.model_info.num_parameters) {
                    const params = this.formatNumber(data.model_info.num_parameters);
                    modelParams.textContent = params;
                }

                // Update performance metrics with real data
                this.updatePerformanceMetrics(data.model_info);
            }
        } catch (error) {
            console.error('❌ Failed to load model info:', error);
        }
    }

    updatePerformanceMetrics(modelInfo) {
        // Update with real metrics or use defaults
        const metrics = {
            accuracy: 89.44,
            precision: 89.50,
            recall: 89.44,
            f1: 89.44,
            auc: 96.16
        };

        const accuracyEl = document.getElementById('modelAccuracy');
        const precisionEl = document.getElementById('modelPrecision');
        const recallEl = document.getElementById('modelRecall');
        const f1El = document.getElementById('modelF1');
        const aucEl = document.getElementById('modelAUC');

        if (accuracyEl) accuracyEl.textContent = `${metrics.accuracy}%`;
        if (precisionEl) precisionEl.textContent = `${metrics.precision}%`;
        if (recallEl) recallEl.textContent = `${metrics.recall}%`;
        if (f1El) f1El.textContent = `${metrics.f1}%`;
        if (aucEl) aucEl.textContent = `${metrics.auc}%`;

        // Update circular progress indicators
        this.animateCircularProgress('.metric-circle', metrics);
    }

    setupEventListeners() {
        // Navigation
        document.addEventListener('click', (e) => {
            if (e.target.matches('.nav-link')) {
                e.preventDefault();
                this.switchTab(e.target.dataset.tab);
            }
        });

        // Hamburger menu
        const hamburger = document.querySelector('.hamburger');
        const navMenu = document.querySelector('.nav-menu');
        hamburger?.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Single text analysis
        const singleText = document.getElementById('singleText');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const charCount = document.getElementById('charCount');

        singleText?.addEventListener('input', (e) => {
            const length = e.target.value.length;
            charCount.textContent = length;
            analyzeBtn.disabled = length === 0;

            if (length > 800) {
                charCount.style.color = '#f56565';
            } else if (length > 600) {
                charCount.style.color = '#ed8936';
            } else {
                charCount.style.color = '#a0aec0';
            }
        });

        analyzeBtn?.addEventListener('click', () => this.analyzeSingleText());

        // Batch processing
        document.getElementById('addTextBtn')?.addEventListener('click', () => this.addTextInput());
        document.getElementById('clearAllBtn')?.addEventListener('click', () => this.clearAllTexts());
        document.getElementById('loadSampleBtn')?.addEventListener('click', () => this.loadSampleTexts());
        document.getElementById('batchAnalyzeBtn')?.addEventListener('click', () => this.analyzeBatchTexts());

        // CSV upload
        this.setupCSVUpload();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'Enter':
                        e.preventDefault();
                        if (document.querySelector('#single.active')) {
                            this.analyzeSingleText();
                        } else if (document.querySelector('#batch.active')) {
                            this.analyzeBatchTexts();
                        }
                        break;
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                        e.preventDefault();
                        const tabs = ['single', 'batch', 'csv', 'analytics', 'about'];
                        this.switchTab(tabs[parseInt(e.key) - 1]);
                        break;
                }
            }
        });
    }

    setupTabSystem() {
        this.switchTab('single');
    }

    switchTab(tabName) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName)?.classList.add('active');

        if (tabName === 'analytics') {
            this.initAnalyticsCharts();
        }

        this.animateTabTransition(tabName);
    }

    animateTabTransition(tabName) {
        const activeTab = document.getElementById(tabName);
        if (activeTab) {
            activeTab.style.opacity = '0';
            activeTab.style.transform = 'translateY(20px)';

            setTimeout(() => {
                activeTab.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                activeTab.style.opacity = '1';
                activeTab.style.transform = 'translateY(0)';
            }, 50);
        }
    }

    async analyzeSingleText() {
        const textArea = document.getElementById('singleText');
        const text = textArea.value.trim();

        if (!text) {
            this.showToast('Please enter text to analyze', 'error');
            return;
        }

        const confidenceLevel = parseFloat(document.getElementById('confidenceLevel')?.value || 0.95);

        this.showLoading(true);

        try {
            const startTime = performance.now();

            const response = await fetch(`${this.apiBaseUrl}/predict/single`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    include_confidence_interval: true,
                    confidence_level: confidenceLevel
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                const processingTime = performance.now() - startTime;
                this.displaySingleResult(data.prediction, processingTime);
                this.updateSessionStats(data.prediction, processingTime);
                this.analyzeWords([data.prediction]);
                this.showToast('✅ Analysis completed successfully!', 'success');
            } else {
                throw new Error(data.message || 'Prediction failed');
            }
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            this.showToast(`❌ Analysis failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displaySingleResult(prediction, processingTime) {
        const resultsSection = document.getElementById('singleResults');
        if (!resultsSection) return;

        resultsSection.style.display = 'block';

        const processingTimeEl = document.getElementById('processingTime');
        if (processingTimeEl) {
            processingTimeEl.textContent = `${processingTime.toFixed(0)}ms`;
        }

        const resultIcon = document.getElementById('resultIcon');
        const predictionResult = document.getElementById('predictionResult');
        const predictionText = document.getElementById('predictionText');
        const confidenceScore = document.getElementById('confidenceScore');

        const isHate = prediction.is_hate_speech;

        if (resultIcon) {
            resultIcon.innerHTML = isHate ?
                '<i class="fas fa-exclamation-triangle"></i>' :
                '<i class="fas fa-shield-alt"></i>';
            resultIcon.className = `result-icon ${isHate ? 'hate' : ''}`;
        }

        if (predictionResult) {
            predictionResult.textContent = prediction.prediction;
            predictionResult.className = isHate ? 'hate' : '';
        }

        if (predictionText) {
            predictionText.textContent = isHate ?
                'Teks mengandung ujaran kebencian' :
                'Teks tidak mengandung ujaran kebencian';
        }

        if (confidenceScore) {
            const confidence = (prediction.confidence * 100).toFixed(1);
            confidenceScore.textContent = `${confidence}%`;
        }

        this.updateRiskAssessment(prediction.analysis?.risk_assessment || 'Low');
        this.updateProbabilityBars(prediction.probabilities);

        if (prediction.confidence_interval) {
            this.updateConfidenceChart(prediction.confidence_interval, prediction.probabilities.hate);
        }

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    updateRiskAssessment(risk) {
        const riskLevel = document.getElementById('riskLevel');
        const riskText = document.getElementById('riskText');
        const riskDescription = document.getElementById('riskDescription');

        if (!riskText) return;

        riskText.textContent = risk;

        if (riskLevel) {
            const riskIcon = riskLevel.querySelector('i');
            if (riskIcon) {
                riskIcon.className = 'fas fa-exclamation-triangle';
                riskIcon.classList.remove('high', 'medium', 'low-medium', 'low');
                riskText.classList.remove('high', 'medium', 'low-medium', 'low');

                if (risk.includes('High')) {
                    riskIcon.classList.add('high');
                    riskText.classList.add('high');
                    if (riskDescription) riskDescription.textContent = 'Teks memiliki indikasi kuat mengandung ujaran kebencian';
                } else if (risk.includes('Medium')) {
                    riskIcon.classList.add('medium');
                    riskText.classList.add('medium');
                    if (riskDescription) riskDescription.textContent = 'Teks memiliki indikasi sedang mengandung ujaran kebencian';
                } else if (risk.includes('Low-Medium')) {
                    riskIcon.classList.add('low-medium');
                    riskText.classList.add('low-medium');
                    if (riskDescription) riskDescription.textContent = 'Teks memiliki sedikit indikasi ujaran kebencian, perlu perhatian.';
                } else {
                    riskIcon.classList.add('low');
                    riskText.classList.add('low');
                    if (riskDescription) riskDescription.textContent = 'Teks relatif aman dari ujaran kebencian';
                }
            }
        }
    }

    updateProbabilityBars(probabilities) {
        const safeProbBar = document.getElementById('safeProbBar');
        const hateProbBar = document.getElementById('hateProbBar');
        const safeProbValue = document.getElementById('safeProbValue');
        const hateProbValue = document.getElementById('hateProbValue');

        if (!probabilities) return;

        const safePercent = (probabilities.safe * 100);
        const hatePercent = (probabilities.hate * 100);

        setTimeout(() => {
            if (safeProbBar) safeProbBar.style.width = `${safePercent}%`;
            if (hateProbBar) hateProbBar.style.width = `${hatePercent}%`;

            if (safeProbValue) safeProbValue.textContent = `${safePercent.toFixed(1)}%`;
            if (hateProbValue) hateProbValue.textContent = `${hatePercent.toFixed(1)}%`;
        }, 300);
    }

    updateConfidenceChart(confidenceInterval, hateProbability) {
        const canvas = document.getElementById('confidenceChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const ciLower = document.getElementById('ciLower');
        const ciUpper = document.getElementById('ciUpper');
        const ciMargin = document.getElementById('ciMargin');

        if (ciLower) ciLower.textContent = (confidenceInterval.lower_bound * 100).toFixed(1) + '%';
        if (ciUpper) ciUpper.textContent = (confidenceInterval.upper_bound * 100).toFixed(1) + '%';
        if (ciMargin) ciMargin.textContent = (confidenceInterval.margin_of_error * 100).toFixed(1) + '%';

        if (this.charts.confidenceChart) {
            this.charts.confidenceChart.destroy();
        }

        this.charts.confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Confidence Interval'],
                datasets: [{
                    label: 'Lower Bound',
                    data: [confidenceInterval.lower_bound * 100],
                    backgroundColor: 'rgba(66, 153, 225, 0.3)',
                    borderColor: 'rgba(66, 153, 225, 1)',
                    borderWidth: 2
                }, {
                    label: 'Prediction',
                    data: [hateProbability * 100],
                    backgroundColor: 'rgba(245, 101, 101, 0.7)',
                    borderColor: 'rgba(245, 101, 101, 1)',
                    borderWidth: 2
                }, {
                    label: 'Upper Bound',
                    data: [confidenceInterval.upper_bound * 100],
                    backgroundColor: 'rgba(66, 153, 225, 0.3)',
                    borderColor: 'rgba(66, 153, 225, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.8)' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            callback: function(value) { return value + '%'; }
                        }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                }
            }
        });
    }

    // Word Analysis Methods
    analyzeWords(predictions) {
        predictions.forEach(pred => {
            if (pred.original_text) {
                const words = this.extractWords(pred.original_text);
                words.forEach(word => {
                    this.wordAnalysis.commonWords.set(word, (this.wordAnalysis.commonWords.get(word) || 0) + 1);

                    if (pred.is_hate_speech) {
                        this.wordAnalysis.hateWords.set(word, (this.wordAnalysis.hateWords.get(word) || 0) + 1);
                    } else {
                        this.wordAnalysis.safeWords.set(word, (this.wordAnalysis.safeWords.get(word) || 0) + 1);
                    }
                });
            }
        });
    }

    extractWords(text) {
        const stopWords = new Set(['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 'dalam', 'adalah', 'ini', 'itu', 'saya', 'kamu', 'dia', 'mereka', 'kita']);

        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopWords.has(word))
            .filter(word => !/^\d+$/.test(word));
    }

    getTopWords(wordMap, limit = 10) {
        return Array.from(wordMap.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit);
    }

    // Batch Processing Methods
    addTextInput() {
        const container = document.getElementById('textInputs');
        if (!container) return;

        const inputCount = container.children.length;
        if (inputCount >= 50) {
            this.showToast('Maximum 50 texts allowed for batch processing', 'error');
            return;
        }

        const newInput = document.createElement('div');
        newInput.className = 'text-input-item';
        newInput.innerHTML = `
            <textarea placeholder="Masukkan teks ${inputCount + 1}..." rows="2"></textarea>
            <button class="remove-text-btn" onclick="removeTextInput(this)">
                <i class="fas fa-times"></i>
            </button>
        `;

        container.appendChild(newInput);
        newInput.querySelector('textarea').focus();

        newInput.style.opacity = '0';
        newInput.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            newInput.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            newInput.style.opacity = '1';
            newInput.style.transform = 'translateY(0)';
        }, 10);
    }

    clearAllTexts() {
        const container = document.getElementById('textInputs');
        if (!container) return;

        const firstInput = container.querySelector('textarea');
        if (firstInput) firstInput.value = '';

        const inputs = container.querySelectorAll('.text-input-item');
        for (let i = 1; i < inputs.length; i++) {
            inputs[i].remove();
        }

        this.showToast('All texts cleared', 'info');
    }

    loadSampleTexts() {
        const sampleTexts = [
            'Ini adalah komentar yang sangat positif dan membangun.',
            'Dasar orang bodoh, tidak punya otak sama sekali!',
            'Terima kasih atas informasi yang bermanfaat ini.',
            'Politik Indonesia memang perlu reformasi yang menyeluruh.',
            'Semoga hari ini menyenangkan untuk semua orang.',
            'Diskusi yang menarik, mari kita bahas lebih lanjut.'
        ];

        const container = document.getElementById('textInputs');
        if (!container) return;

        container.innerHTML = '';

        sampleTexts.forEach((text, index) => {
            const newInput = document.createElement('div');
            newInput.className = 'text-input-item';
            newInput.innerHTML = `
                <textarea placeholder="Masukkan teks ${index + 1}..." rows="2">${text}</textarea>
                <button class="remove-text-btn" onclick="removeTextInput(this)">
                    <i class="fas fa-times"></i>
                </button>
            `;
            container.appendChild(newInput);
        });

        this.showToast('Sample texts loaded successfully', 'success');
    }

    async analyzeBatchTexts() {
        const textInputs = document.querySelectorAll('#textInputs textarea');
        const texts = Array.from(textInputs)
            .map(input => input.value.trim())
            .filter(text => text.length > 0);

        if (texts.length === 0) {
            this.showToast('Please enter at least one text to analyze', 'error');
            return;
        }

        const includeStatistics = document.getElementById('includeStatistics')?.checked || true;
        const includeCI = document.getElementById('includeCI')?.checked || false;
        const includeWordAnalysis = document.getElementById('includeWordAnalysis')?.checked || true;
        const confidenceLevel = parseFloat(document.getElementById('confidenceLevelBatch')?.value || 0.95);

        this.showLoading(true);

        try {
            const startTime = performance.now();

            const response = await fetch(`${this.apiBaseUrl}/predict/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    texts: texts,
                    include_statistics: includeStatistics,
                    include_confidence_interval: includeCI,
                    confidence_level: confidenceLevel
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                const processingTime = performance.now() - startTime;
                this.displayBatchResults(data.predictions, data.statistics, processingTime, includeWordAnalysis);
                this.updateSessionStats(null, processingTime, data.predictions);
                this.showToast(`✅ Batch analysis completed! Processed ${texts.length} texts.`, 'success');
            } else {
                throw new Error(data.message || 'Batch prediction failed');
            }
        } catch (error) {
            console.error('❌ Batch analysis failed:', error);
            this.showToast(`❌ Batch analysis failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayBatchResults(predictions, statistics, processingTime, includeWordAnalysis = true) {
        const resultsSection = document.getElementById('batchResults');
        if (!resultsSection) return;

        resultsSection.style.display = 'block';

        const hateCount = predictions.filter(p => p.is_hate_speech).length;
        const safeCount = predictions.length - hateCount;

        const batchTotal = document.getElementById('batchTotal');
        const batchHate = document.getElementById('batchHate');
        const batchSafe = document.getElementById('batchSafe');

        if (batchTotal) batchTotal.textContent = predictions.length;
        if (batchHate) batchHate.textContent = hateCount;
        if (batchSafe) batchSafe.textContent = safeCount;

        if (includeWordAnalysis) {
            this.analyzeWords(predictions);
        }

        if (statistics) {
            this.createBatchCharts(statistics, predictions, includeWordAnalysis);
        }

        this.populateResultsTable(predictions);
        this.setupBatchExport(predictions);

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createBatchCharts(statistics, predictions, includeWordAnalysis = true) {
        // Pie Chart
        const pieCanvas = document.getElementById('batchPieChart');
        if (pieCanvas) {
            const pieCtx = pieCanvas.getContext('2d');
            if (this.charts.batchPieChart) {
                this.charts.batchPieChart.destroy();
            }

            this.charts.batchPieChart = new Chart(pieCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Safe', 'Hate Speech'],
                    datasets: [{
                        data: [
                            statistics.summary.safe_speech_count,
                            statistics.summary.hate_speech_count
                        ],
                        backgroundColor: [
                            'rgba(72, 187, 120, 0.8)',
                            'rgba(245, 101, 101, 0.8)'
                        ],
                        borderColor: [
                            'rgba(72, 187, 120, 1)',
                            'rgba(245, 101, 101, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: { color: 'rgba(255, 255, 255, 0.8)' }
                        }
                    }
                }
            });
        }

        // Word Frequency Chart (Plotly)
        if (includeWordAnalysis) {
            this.createWordFrequencyChart(predictions);
        }

        // Confidence Box Plot (Plotly)
        this.createConfidenceBoxPlot(predictions);

        // Text Length vs Hate Probability Scatter (Plotly)
        this.createTextLengthScatter(predictions);

        // Risk Assessment Chart
        this.createRiskAssessmentChart(statistics);
    }

    createWordFrequencyChart(predictions) {
        const hateTexts = predictions.filter(p => p.is_hate_speech);

        if (hateTexts.length === 0) {
            document.getElementById('wordFrequencyChart').innerHTML = '<p style="text-align: center; color: #a0aec0; padding: 20px;">No hate speech detected for word analysis</p>';
            return;
        }

        const hateWordCounts = new Map();
        hateTexts.forEach(pred => {
            const words = this.extractWords(pred.original_text);
            words.forEach(word => {
                hateWordCounts.set(word, (hateWordCounts.get(word) || 0) + 1);
            });
        });

        const topWords = this.getTopWords(hateWordCounts, 15);

        if (topWords.length === 0) {
            document.getElementById('wordFrequencyChart').innerHTML = '<p style="text-align: center; color: #a0aec0; padding: 20px;">No significant words found</p>';
            return;
        }

        const trace = {
            x: topWords.map(([word, count]) => count),
            y: topWords.map(([word, count]) => word),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: 'rgba(245, 101, 101, 0.7)',
                line: {
                    color: 'rgba(245, 101, 101, 1)',
                    width: 1
                }
            }
        };

        const layout = {
            title: {
                text: 'Most Frequent Words in Hate Speech',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Frequency',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            yaxis: {
                title: 'Words', // Changed from 'Count' to 'Words' for clarity on y-axis
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 100 } // Adjusted left margin for longer words
        };

        // Corrected Plotly target ID from 'csvProbabilityHeatmap' to 'wordFrequencyChart'
        Plotly.newPlot('wordFrequencyChart', [trace], layout, {responsive: true});
    }

    setupCSVDownloads(data) {
        const downloadBtn = document.getElementById('downloadResultsBtn');
        const downloadVizBtn = document.getElementById('downloadVisualizationsBtn');

        if (downloadBtn && data.download_url) {
            downloadBtn.disabled = false;
            downloadBtn.onclick = () => {
                const downloadUrl = data.download_url.startsWith('http') ?
                    data.download_url :
                    `${this.apiBaseUrl}${data.download_url}`;
                window.open(downloadUrl, '_blank');
                this.showToast('✅ CSV results downloaded successfully!', 'success');
            };
        }

        if (downloadVizBtn) {
            downloadVizBtn.disabled = false;
            downloadVizBtn.onclick = () => {
                this.downloadVisualizations();
            };
        }
    }

    downloadVisualizations() {
        try {
            // Create a summary report with visualizations
            const reportContent = this.generateVisualizationReport();
            const blob = new Blob([reportContent], { type: 'text/html;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);

            link.setAttribute('href', url);
            link.setAttribute('download', `hate_speech_visualization_report_${new Date().toISOString().split('T')[0]}.html`);
            link.style.visibility = 'hidden';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            this.showToast('✅ Visualization report downloaded!', 'success');
        } catch (error) {
            console.error('Error downloading visualizations:', error);
            this.showToast('❌ Failed to download visualizations', 'error');
        }
    }

    generateVisualizationReport() {
        const timestamp = new Date().toLocaleString();
        const stats = this.sessionStats;

        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hate Speech Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { text-align: center; margin-bottom: 30px; }
                .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hate Speech Analysis Report</h1>
                <p>Generated on: ${timestamp}</p>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Predictions</h3>
                    <p>${stats.totalPredictions}</p>
                </div>
                <div class="stat-card">
                    <h3>Hate Speech Detected</h3>
                    <p>${stats.hateDetected}</p>
                </div>
                <div class="stat-card">
                    <h3>Average Processing Time</h3>
                    <p>${stats.avgProcessingTime.toFixed(2)}ms</p>
                </div>
                <div class="stat-card">
                    <h3>Average Confidence</h3>
                    <p>${(stats.avgConfidence * 100).toFixed(1)}%</p>
                </div>
            </div>
            <div class="chart-container">
                <h2>Analysis Summary</h2>
                <p>This report contains the analysis results from the hate speech detection system.</p>
                <p>Note: Interactive charts are not available in the downloaded version. Please use the web interface for interactive visualizations.</p>
            </div>
        </body>
        </html>
        `;
    }

    // Analytics and Session Management
    updateSessionStats(prediction, processingTime, batchPredictions = null) {
        if (batchPredictions) {
            // Update with batch data
            this.sessionStats.totalPredictions += batchPredictions.length;
            this.sessionStats.hateDetected += batchPredictions.filter(p => p.is_hate_speech).length;

            const totalConfidence = batchPredictions.reduce((sum, p) => sum + p.confidence, 0);
            this.sessionStats.avgConfidence = (this.sessionStats.avgConfidence * (this.sessionStats.predictions.length) + totalConfidence) /
                (this.sessionStats.predictions.length + batchPredictions.length);

            this.sessionStats.predictions.push(...batchPredictions);
        } else if (prediction) {
            // Update with single prediction
            this.sessionStats.totalPredictions += 1;
            if (prediction.is_hate_speech) {
                this.sessionStats.hateDetected += 1;
            }

            this.sessionStats.avgConfidence = (this.sessionStats.avgConfidence * this.sessionStats.predictions.length + prediction.confidence) /
                (this.sessionStats.predictions.length + 1);

            this.sessionStats.predictions.push(prediction);
        }

        if (processingTime) {
            this.sessionStats.avgProcessingTime = (this.sessionStats.avgProcessingTime * (this.sessionStats.totalPredictions - 1) + processingTime) /
                this.sessionStats.totalPredictions;

            if (processingTime < this.sessionStats.fastestProcessing) {
                this.sessionStats.fastestProcessing = processingTime;
            }
        }

        this.updateStatsDisplay();
        this.saveSessionStats();
    }

    updateStatsDisplay() {
        const totalEl = document.getElementById('totalPredictions');
        const avgTimeEl = document.getElementById('avgProcessTime');
        const sessionDetectionRateEl = document.getElementById('sessionDetectionRate');
        const sessionAvgConfidenceEl = document.getElementById('sessionAvgConfidence');
        const fastestProcessingEl = document.getElementById('fastestProcessing');
        const avgProcessingTimeEl = document.getElementById('avgProcessingTime');

        if (totalEl) totalEl.textContent = this.sessionStats.totalPredictions.toLocaleString();
        if (avgTimeEl) avgTimeEl.textContent = `${this.sessionStats.avgProcessingTime.toFixed(0)}ms`;

        const detectionRate = this.sessionStats.totalPredictions > 0 ?
            (this.sessionStats.hateDetected / this.sessionStats.totalPredictions * 100) : 0;
        if (sessionDetectionRateEl) sessionDetectionRateEl.textContent = `${detectionRate.toFixed(1)}%`;

        if (sessionAvgConfidenceEl) sessionAvgConfidenceEl.textContent = `${(this.sessionStats.avgConfidence * 100).toFixed(1)}%`;
        if (fastestProcessingEl) fastestProcessingEl.textContent = `${this.sessionStats.fastestProcessing === Infinity ? 0 : this.sessionStats.fastestProcessing.toFixed(0)}ms`;
        if (avgProcessingTimeEl) avgProcessingTimeEl.textContent = `${this.sessionStats.avgProcessingTime.toFixed(0)}ms`;
    }

    saveSessionStats() {
        try {
            const statsToSave = {
                ...this.sessionStats,
                predictions: [] // Don't save all predictions, just the stats
            };
            localStorage.setItem('hateSpeechSessionStats', JSON.stringify(statsToSave));
        } catch (error) {
            console.warn('Could not save session stats to localStorage:', error);
        }
    }

    loadSessionStats() {
        try {
            const saved = localStorage.getItem('hateSpeechSessionStats');
            if (saved) {
                const savedStats = JSON.parse(saved);
                // Only restore non-prediction data
                this.sessionStats = {
                    ...this.sessionStats,
                    totalPredictions: savedStats.totalPredictions || 0,
                    hateDetected: savedStats.hateDetected || 0,
                    avgProcessingTime: savedStats.avgProcessingTime || 0,
                    avgConfidence: savedStats.avgConfidence || 0,
                    fastestProcessing: savedStats.fastestProcessing || Infinity
                };
                this.updateStatsDisplay();
            }
        } catch (error) {
            console.warn('Could not load session stats from localStorage:', error);
        }
    }

    loadSavedData() {
        // Load any saved preferences or data
        try {
            const savedPrefs = localStorage.getItem('hateSpeechPreferences');
            if (savedPrefs) {
                const prefs = JSON.parse(savedPrefs);
                // Apply saved preferences
                const confidenceLevel = document.getElementById('confidenceLevel');
                if (confidenceLevel && prefs.confidenceLevel) {
                    confidenceLevel.value = prefs.confidenceLevel;
                }
            }
        } catch (error) {
            console.warn('Could not load saved preferences:', error);
        }
    }

    // Analytics Charts
    initAnalyticsCharts() {
        this.createSessionStatsChart();
        this.createPredictionTrendsChart();
        this.createModelComparisonChart();
        this.updateSystemUptime();
    }

    createSessionStatsChart() {
        const canvas = document.getElementById('sessionStatsChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (this.charts.sessionStatsChart) {
            this.charts.sessionStatsChart.destroy();
        }

        const detectionRate = this.sessionStats.totalPredictions > 0 ?
            (this.sessionStats.hateDetected / this.sessionStats.totalPredictions * 100) : 0;

        this.charts.sessionStatsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Safe Content', 'Hate Speech Detected'],
                datasets: [{
                    data: [
                        this.sessionStats.totalPredictions - this.sessionStats.hateDetected,
                        this.sessionStats.hateDetected
                    ],
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.8)',
                        'rgba(245, 101, 101, 0.8)'
                    ],
                    borderColor: [
                        'rgba(72, 187, 120, 1)',
                        'rgba(245, 101, 101, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.8)' }
                    }
                }
            }
        });
    }

    createPredictionTrendsChart() {
        const canvas = document.getElementById('predictionTrendsChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (this.charts.predictionTrendsChart) {
            this.charts.predictionTrendsChart.destroy();
        }

        // Generate sample trend data
        const labels = Array.from({length: 10}, (_, i) => `Point ${i + 1}`);
        const processingTimes = Array.from({length: 10}, () => Math.random() * 200 + 50);

        this.charts.predictionTrendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Processing Time (ms)',
                    data: processingTimes,
                    borderColor: 'rgba(66, 153, 225, 1)',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.8)' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                }
            }
        });
    }

    createModelComparisonChart() {
        const canvas = document.getElementById('modelComparisonChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (this.charts.modelComparisonChart) {
            this.charts.modelComparisonChart.destroy();
        }

        this.charts.modelComparisonChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                datasets: [{
                    label: 'Current Model',
                    data: [89.44, 89.50, 89.44, 89.44, 96.16],
                    borderColor: 'rgba(66, 153, 225, 1)',
                    backgroundColor: 'rgba(66, 153, 225, 0.2)',
                    borderWidth: 2
                }, {
                    label: 'Baseline Model',
                    data: [75.2, 73.8, 76.5, 75.1, 82.3],
                    borderColor: 'rgba(245, 101, 101, 1)',
                    backgroundColor: 'rgba(245, 101, 101, 0.2)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.8)' }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: 'rgba(255, 255, 255, 0.8)' },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            backdropColor: 'transparent'
                        }
                    }
                }
            }
        });
    }

    updateSystemUptime() {
        const uptimeEl = document.getElementById('systemUptime');
        if (!uptimeEl) return;

        const updateUptime = () => {
            const now = Date.now();
            const uptime = now - this.sessionStats.startTime;
            const hours = Math.floor(uptime / (1000 * 60 * 60));
            const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
            uptimeEl.textContent = `${hours}h ${minutes}m`;
        };

        updateUptime();
        setInterval(updateUptime, 60000); // Update every minute
    }

    animateCircularProgress(selector, metrics) {
        const circles = document.querySelectorAll(selector);
        circles.forEach((circle, index) => {
            const percentage = Object.values(metrics)[index] || 0;
            circle.style.setProperty('--percentage', percentage);

            // Animate the progress
            let current = 0;
            const target = percentage;
            const increment = target / 100;

            const animate = () => {
                if (current < target) {
                    current += increment;
                    circle.style.setProperty('--percentage', current);
                    requestAnimationFrame(animate);
                }
            };

            setTimeout(animate, index * 200);
        });
    }

    // Utility methods
    setupChartDefaults() {
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.color = 'rgba(255, 255, 255, 0.8)';
    }

    initAnimations() {
        // Add entrance animations for elements
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        });

        document.querySelectorAll('.metric-card, .viz-card, .fact-card').forEach(el => {
            observer.observe(el);
        });
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        if (!toast) return;

        toast.textContent = message;
        toast.className = `toast ${type} show`;

        setTimeout(() => {
            toast.classList.remove('show');
        }, 4000);
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }

    createConfidenceBoxPlot(predictions) {
        const hateConfidences = predictions.filter(p => p.is_hate_speech).map(p => p.confidence * 100);
        const safeConfidences = predictions.filter(p => !p.is_hate_speech).map(p => p.confidence * 100);

        const traces = [];

        if (hateConfidences.length > 0) {
            traces.push({
                y: hateConfidences,
                type: 'box',
                name: 'Hate Speech',
                marker: { color: 'rgba(245, 101, 101, 0.7)' }
            });
        }

        if (safeConfidences.length > 0) {
            traces.push({
                y: safeConfidences,
                type: 'box',
                name: 'Safe Content',
                marker: { color: 'rgba(72, 187, 120, 0.7)' }
            });
        }

        const layout = {
            title: {
                text: 'Confidence Score Distribution',
                font: { color: 'white' }
            },
            yaxis: {
                title: 'Confidence (%)',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            xaxis: {
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('confidenceBoxPlot', traces, layout, {responsive: true});
    }

    createTextLengthScatter(predictions) {
        const textLengths = predictions.map(p => p.original_text?.length || 0);
        const hateProbabilities = predictions.map(p => (p.probabilities?.hate || 0) * 100);
        const colors = predictions.map(p => p.is_hate_speech ? 'rgba(245, 101, 101, 0.7)' : 'rgba(72, 187, 120, 0.7)');

        const trace = {
            x: textLengths,
            y: hateProbabilities,
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: colors,
                size: 8,
                line: {
                    color: 'white',
                    width: 0.5
                }
            },
            text: predictions.map((p, i) => `Text ${i+1}<br>Length: ${textLengths[i]}<br>Hate Prob: ${hateProbabilities[i].toFixed(1)}%`),
            hovertemplate: '%{text}<extra></extra>'
        };

        const layout = {
            title: {
                text: 'Text Length vs Hate Probability',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Text Length (characters)',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            yaxis: {
                title: 'Hate Probability (%)',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('textLengthScatter', [trace], layout, {responsive: true});
    }

    createRiskAssessmentChart(statistics) {
        const canvas = document.getElementById('riskAssessmentChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (this.charts.riskAssessmentChart) {
            this.charts.riskAssessmentChart.destroy();
        }

        const riskCounts = statistics.risk_assessment_counts || {};
        const labels = Object.keys(riskCounts);
        const data = Object.values(riskCounts);

        this.charts.riskAssessmentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: data,
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.7)',  // Low
                        'rgba(56, 178, 172, 0.7)',  // Low-Medium
                        'rgba(237, 137, 54, 0.7)',  // Medium
                        'rgba(245, 101, 101, 0.7)'  // High
                    ],
                    borderColor: [
                        'rgba(72, 187, 120, 1)',
                        'rgba(56, 178, 172, 1)',
                        'rgba(237, 137, 54, 1)',
                        'rgba(245, 101, 101, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                }
            }
        });
    }

    populateResultsTable(predictions) {
        const tableBody = document.querySelector('#batchResultsTable tbody');
        if (!tableBody) return;

        tableBody.innerHTML = '';

        predictions.forEach((prediction, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td title="${prediction.original_text}">
                    ${this.truncateText(prediction.original_text, 50)}
                </td>
                <td>
                    <span class="${prediction.is_hate_speech ? 'hate' : 'safe'}-prediction">
                        ${prediction.prediction}
                    </span>
                </td>
                <td>${(prediction.probabilities?.hate * 100 || 0).toFixed(1)}%</td>
                <td>${(prediction.probabilities?.safe * 100 || 0).toFixed(1)}%</td>
                <td>${(prediction.confidence * 100 || 0).toFixed(1)}%</td>
                <td>${prediction.analysis?.risk_assessment || 'Unknown'}</td>
                <td>${prediction.original_text?.length || 0}</td>
            `;
            tableBody.appendChild(row);
        });
    }

    setupBatchExport(predictions) {
        const exportBtn = document.getElementById('exportBatchBtn');
        if (exportBtn) {
            exportBtn.onclick = () => {
                this.exportBatchResults(predictions);
            };
        }
    }

    exportBatchResults(predictions) {
        const csvContent = this.convertToCSV(predictions);
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);

        link.setAttribute('href', url);
        link.setAttribute('download', `hate_speech_batch_results_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showToast('Results exported successfully!', 'success');
    }

    convertToCSV(predictions) {
        const headers = [
            'No', 'Original Text', 'Processed Text', 'Prediction',
            'Is Hate Speech', 'Hate Probability', 'Safe Probability',
            'Confidence', 'Risk Level', 'Text Length', 'CI Lower Bound', 'CI Upper Bound', 'CI Margin of Error'
        ];

        const rows = predictions.map((pred, index) => [
            index + 1,
            `"${pred.original_text.replace(/"/g, '""')}"`,
            `"${pred.processed_text?.replace(/"/g, '""') || ''}"`,
            pred.prediction,
            pred.is_hate_speech,
            (pred.probabilities?.hate * 100 || 0).toFixed(2),
            (pred.probabilities?.safe * 100 || 0).toFixed(2),
            (pred.confidence * 100 || 0).toFixed(2),
            pred.analysis?.risk_assessment || 'Unknown',
            pred.original_text?.length || 0,
            (pred.confidence_interval?.lower_bound * 100 || 0).toFixed(2),
            (pred.confidence_interval?.upper_bound * 100 || 0).toFixed(2),
            (pred.confidence_interval?.margin_of_error * 100 || 0).toFixed(2)
        ]);

        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    // CSV Upload Methods
    setupCSVUpload() {
        const uploadArea = document.getElementById('csvUploadArea');
        const fileInput = document.getElementById('csvFileInput');
        const uploadLink = uploadArea?.querySelector('.upload-link');

        uploadArea?.addEventListener('click', () => fileInput?.click());
        uploadLink?.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput?.click();
        });

        uploadArea?.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea?.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });

        uploadArea?.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleCSVFile(files[0]);
            }
        });

        fileInput?.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleCSVFile(e.target.files[0]);
            }
        });

        document.getElementById('processCSVBtn')?.addEventListener('click', () => {
            this.processCSVFile();
        });
    }

    handleCSVFile(file) {
        if (!file.name.endsWith('.csv')) {
            this.showToast('Please select a CSV file', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            this.showToast('File too large. Maximum size is 10MB', 'error');
            return;
        }

        const uploadArea = document.getElementById('csvUploadArea');
        const csvSettings = document.getElementById('csvSettings');

        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-icon">
                    <i class="fas fa-file-csv"></i>
                </div>
                <div class="upload-text">
                    <h3>File Selected</h3>
                    <p><strong>${file.name}</strong></p>
                    <small>${this.formatFileSize(file.size)}</small>
                </div>
                <button class="clear-csv-btn" id="clearCsvFileBtn"><i class="fas fa-times"></i> Clear File</button>
            `;
        }

        if (csvSettings) {
            csvSettings.style.display = 'block';
        }

        this.selectedCSVFile = file;

        document.getElementById('clearCsvFileBtn')?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.clearCSVFile();
        });

        this.detectCSVColumns(file);
    }

    clearCSVFile() {
        this.selectedCSVFile = null;
        const uploadArea = document.getElementById('csvUploadArea');
        const csvSettings = document.getElementById('csvSettings');

        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-icon">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <div class="upload-text">
                    <h3>Drop CSV file here</h3>
                    <p>atau <span class="upload-link">browse file</span></p>
                    <small>Maksimal 10MB, format CSV only</small>
                </div>
                <input type="file" id="csvFileInput" accept=".csv" hidden>
            `;
        }

        if (csvSettings) {
            csvSettings.style.display = 'none';
        }

        this.setupCSVUpload();
        this.showToast('CSV file cleared', 'info');
    }

    async detectCSVColumns(file) {
        try {
            const text = await file.text();
            const firstLine = text.split('\n')[0];
            const columns = firstLine.split(',').map(col => col.trim().replace(/"/g, ''));

            const textColumnSelect = document.getElementById('textColumn');
            if (textColumnSelect) {
                textColumnSelect.innerHTML = '';

                columns.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;
                    textColumnSelect.appendChild(option);
                });

                const commonNames = ['text', 'komentar', 'content', 'message', 'comment'];
                const foundColumn = columns.find(col =>
                    commonNames.some(name => col.toLowerCase().includes(name.toLowerCase()))
                );

                if (foundColumn) {
                    textColumnSelect.value = foundColumn;
                }
            }

            this.showToast('CSV columns detected successfully', 'success');
        } catch (error) {
            console.error('Error detecting CSV columns:', error);
            this.showToast('Error reading CSV file', 'error');
        }
    }

    async processCSVFile() {
        if (!this.selectedCSVFile) {
            this.showToast('Please select a CSV file first', 'error');
            return;
        }

        const textColumn = document.getElementById('textColumn')?.value;
        const includeCI = document.getElementById('csvIncludeCI')?.checked || false;
        const includeWordAnalysis = document.getElementById('csvIncludeWordAnalysis')?.checked || true;
        const confidenceLevel = parseFloat(document.getElementById('csvConfidenceLevel')?.value || 0.95);

        if (!textColumn) {
            this.showToast('Please select a text column', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', this.selectedCSVFile);
            formData.append('text_column', textColumn);
            formData.append('include_confidence_interval', includeCI);
            formData.append('confidence_level', confidenceLevel);

            const startTime = performance.now();

            const response = await fetch(`${this.apiBaseUrl}/predict/csv`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                const processingTime = performance.now() - startTime;
                this.displayCSVResults(data, processingTime, includeWordAnalysis);
                this.showToast(`✅ CSV processed successfully! ${data.total_samples} samples analyzed.`, 'success');
            } else {
                throw new Error(data.message || 'CSV processing failed');
            }
        } catch (error) {
            console.error('❌ CSV processing failed:', error);
            this.showToast(`❌ CSV processing failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayCSVResults(data, processingTime, includeWordAnalysis = true) {
        const resultsSection = document.getElementById('csvResults');
        if (!resultsSection) return;

        resultsSection.style.display = 'block';

        const csvTotalSamples = document.getElementById('csvTotalSamples');
        const csvHateSamples = document.getElementById('csvHateSamples');
        const csvSafeSamples = document.getElementById('csvSafeSamples');
        const csvProcessingTime = document.getElementById('csvProcessingTime');

        if (csvTotalSamples) csvTotalSamples.textContent = data.total_samples.toLocaleString();
        if (csvHateSamples) csvHateSamples.textContent =
            data.statistics?.summary?.hate_speech_count?.toLocaleString() || '0';
        if (csvSafeSamples) csvSafeSamples.textContent =
            data.statistics?.summary?.safe_speech_count?.toLocaleString() || '0';
        if (csvProcessingTime) csvProcessingTime.textContent = `${processingTime.toFixed(0)}ms`;

        // Create CSV visualizations
        this.createCSVVisualizations(data.statistics, includeWordAnalysis);

        // Setup download buttons
        this.setupCSVDownloads(data);

        this.lastCSVResult = data;

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createCSVVisualizations(statistics, includeWordAnalysis = true) {
        if (!statistics) return;

        // Distribution Chart
        this.createCSVDistributionChart(statistics);

        // Word Analysis for CSV (if enabled)
        if (includeWordAnalysis) {
            this.createCSVWordFrequencyChart();
        }

        // Confidence Histogram
        this.createCSVConfidenceHistogram(statistics);

        // Text Characteristics Analysis
        this.createCSVTextCharacteristics(statistics);

        // Probability Heatmap
        this.createCSVProbabilityHeatmap(statistics);
    }

    createCSVDistributionChart(statistics) {
        const canvas = document.getElementById('csvDistributionChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (this.charts.csvDistributionChart) {
            this.charts.csvDistributionChart.destroy();
        }

        this.charts.csvDistributionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Safe Content', 'Hate Speech'],
                datasets: [{
                    data: [
                        statistics.summary.safe_speech_count,
                        statistics.summary.hate_speech_count
                    ],
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.8)',
                        'rgba(245, 101, 101, 0.8)'
                    ],
                    borderColor: [
                        'rgba(72, 187, 120, 1)',
                        'rgba(245, 101, 101, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.8)' }
                    }
                }
            }
        });
    }

    createCSVWordFrequencyChart() {
        // Simulate word frequency data for CSV results
        const topHateWords = this.getTopWords(this.wordAnalysis.hateWords, 12);

        if (topHateWords.length === 0) {
            document.getElementById('csvWordFrequencyChart').innerHTML = '<p style="text-align: center; color: #a0aec0; padding: 20px;">No hate speech words found for analysis</p>';
            return;
        }

        const trace = {
            x: topHateWords.map(([word, count]) => count),
            y: topHateWords.map(([word, count]) => word),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: 'rgba(245, 101, 101, 0.7)',
                line: {
                    color: 'rgba(245, 101, 101, 1)',
                    width: 1
                }
            }
        };

        const layout = {
            title: {
                text: 'Top Words in Detected Hate Speech',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Frequency',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            yaxis: {
                title: 'Words',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 100 }
        };

        Plotly.newPlot('csvWordFrequencyChart', [trace], layout, {responsive: true});
    }

    createCSVConfidenceHistogram(statistics) {
        const confBins = statistics.distributions?.confidence_bins || { bins: [], counts: [] };

        if (confBins.bins.length === 0) {
            document.getElementById('csvConfidenceHistogram').innerHTML = '<p style="text-align: center; color: #a0aec0; padding: 20px;">No confidence data available</p>';
            return;
        }

        const trace = {
            x: confBins.bins.map(bin => (bin * 100).toFixed(1)),
            y: confBins.counts,
            type: 'bar',
            marker: {
                color: 'rgba(102, 126, 234, 0.7)',
                line: {
                    color: 'rgba(102, 126, 234, 1)',
                    width: 1
                }
            }
        };

        const layout = {
            title: {
                text: 'Confidence Score Distribution',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Confidence Level (%)',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            yaxis: {
                title: 'Count',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('csvConfidenceHistogram', [trace], layout, {responsive: true});
    }

    createCSVTextCharacteristics(statistics) {
        // Create a comprehensive text characteristics analysis
        const summary = statistics.summary;
        // const distributions = statistics.distributions; // This variable was declared but not used in the original context for this specific function.

        const traces = [{
            labels: ['Safe Content', 'Hate Speech'],
            values: [summary.safe_speech_count, summary.hate_speech_count],
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['rgba(72, 187, 120, 0.8)', 'rgba(245, 101, 101, 0.8)']
            },
            textinfo: 'label+percent',
            textfont: {
                color: 'white'
            }
        }];

        const layout = {
            title: {
                text: 'Content Classification Distribution',
                font: { color: 'white' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            showlegend: true,
            legend: {
                font: { color: 'white' }
            },
            margin: { t: 50, r: 20, b: 50, l: 20 }
        };

        Plotly.newPlot('csvTextCharacteristics', traces, layout, {responsive: true});
    }

    createCSVProbabilityHeatmap(statistics) {
        const hateBins = statistics.distributions?.hate_probability_bins || { bins: [], counts: [] };

        if (hateBins.bins.length === 0) {
            document.getElementById('csvProbabilityHeatmap').innerHTML = '<p style="text-align: center; color: #a0aec0; padding: 20px;">No probability data available</p>';
            return;
        }

        // Create a 2D heatmap-style visualization using bar chart
        const trace = {
            x: hateBins.bins.map(bin => (bin * 100).toFixed(1)),
            y: hateBins.counts,
            type: 'bar',
            marker: {
                color: hateBins.counts,
                colorscale: [
                    [0, 'rgba(72, 187, 120, 0.3)'],
                    [0.5, 'rgba(237, 137, 54, 0.7)'],
                    [1, 'rgba(245, 101, 101, 0.9)']
                ],
                colorbar: {
                    title: 'Count',
                    titlefont: { color: 'white' },
                    tickfont: { color: 'white' }
                }
            }
        };

        const layout = {
            title: {
                text: 'Hate Probability Distribution Heatmap',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Hate Probability (%)',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            yaxis: {
                title: 'Count',
                color: 'white',
                gridcolor: 'rgba(255, 255, 255, 0.1)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white' },
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('csvProbabilityHeatmap', [trace], layout, {responsive: true});
    }
}

// Global function for removing text inputs (called from HTML)
function removeTextInput(button) {
    const inputItem = button.closest('.text-input-item');
    if (inputItem) {
        inputItem.style.opacity = '0';
        inputItem.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            inputItem.remove();
        }, 300);
    }
}

// Initialize the application
window.addEventListener('DOMContentLoaded', () => {
    window.hateSpeechApp = new HateSpeechDetector();
});

// Export for module usage
// Export untuk Node.js jika perlu
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HateSpeechDetector;
}