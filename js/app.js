/**
 * Main Application Controller for K-Means Image Segmentation
 * Handles UI interactions and coordinates with K-Means algorithm
 */

class KMeansImageSegmentation {
    constructor() {
        this.originalImage = null;
        this.kmeansProcessor = new KMeans();
        this.isProcessing = false;

        this.initializeElements();
        this.setupEventListeners();
        this.initializeUI();
    }

    /**
     * Initialize DOM element references
     */
    initializeElements() {
        this.elements = {
            imageInput: document.getElementById('imageInput'),
            kSlider: document.getElementById('kSlider'),
            kValue: document.getElementById('kValue'),
            maxIterations: document.getElementById('maxIterations'),
            processBtn: document.getElementById('processBtn'),
            originalContainer: document.getElementById('originalContainer'),
            segmentedContainer: document.getElementById('segmentedContainer'),
            status: document.getElementById('status'),
            imageInfo: document.getElementById('imageInfo'),
            processingStats: document.getElementById('processingStats')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        this.elements.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.elements.kSlider.addEventListener('input', (e) => this.updateKValue(e));
        this.elements.processBtn.addEventListener('click', () => this.processImage());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'o':
                        e.preventDefault();
                        this.elements.imageInput.click();
                        break;
                    case 'Enter':
                        if (!this.elements.processBtn.disabled) {
                            e.preventDefault();
                            this.processImage();
                        }
                        break;
                }
            }
        });

        // Drag and drop support
        this.setupDragAndDrop();
    }

    /**
     * Setup drag and drop functionality
     */
    setupDragAndDrop() {
        const dropZone = this.elements.originalContainer;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
        });

        dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    /**
     * Prevent default drag behaviors
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Handle dropped files
     */
    handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processUploadedFile(files[0]);
        }
    }

    /**
     * Initialize UI state
     */
    initializeUI() {
        this.updateMaxIterationsDisplay();
        this.showStatus('Ready to process images! Upload an image to get started.', 'info');
    }

    /**
     * Update K value display
     */
    updateKValue(e) {
        this.elements.kValue.textContent = e.target.value;
    }

    /**
     * Update max iterations display
     */
    updateMaxIterationsDisplay() {
        this.elements.maxIterations.textContent = this.kmeansProcessor.maxIterations;
    }

    /**
     * Handle image file upload
     */
    handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            this.processUploadedFile(file);
        }
    }

    /**
     * Process uploaded image file
     */
    processUploadedFile(file) {
        if (!this.validateImageFile(file)) {
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.displayOriginalImage(img);
                this.elements.processBtn.disabled = false;
                this.showImageInfo(file, img);
                this.showStatus('Image loaded successfully! Adjust K value and click Process Image.', 'success');
                this.clearSegmentedImage();
            };
            img.onerror = () => {
                this.showStatus('Error loading image. Please try a different file.', 'error');
            };
            img.src = e.target.result;
        };
        reader.onerror = () => {
            this.showStatus('Error reading file. Please try again.', 'error');
        };
        reader.readAsDataURL(file);
    }

    /**
     * Validate uploaded image file
     */
    validateImageFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showStatus('Please select a valid image file (JPG, PNG, GIF, etc.).', 'error');
            return false;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showStatus('Image file is too large. Please select an image smaller than 10MB.', 'error');
            return false;
        }

        return true;
    }

    /**
     * Display original image in container
     */
    displayOriginalImage(img) {
        this.elements.originalContainer.innerHTML = '<h3>Original Image</h3>';
        const displayImg = img.cloneNode();
        displayImg.style.maxWidth = '100%';
        displayImg.style.maxHeight = '400px';
        displayImg.alt = 'Original Image';
        this.elements.originalContainer.appendChild(displayImg);
    }

    /**
     * Show image information
     */
    showImageInfo(file, img) {
        const fileSizeKB = Math.round(file.size / 1024);
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        const sizeText = fileSizeKB > 1024 ? `${fileSizeMB} MB` : `${fileSizeKB} KB`;

        this.elements.imageInfo.innerHTML = `
            <strong>File:</strong> ${file.name}<br>
            <strong>Size:</strong> ${sizeText}<br>
            <strong>Dimensions:</strong> ${img.width} × ${img.height} pixels<br>
            <strong>Type:</strong> ${file.type}
        `;
        this.elements.imageInfo.style.display = 'block';
    }

    /**
     * Clear segmented image display
     */
    clearSegmentedImage() {
        this.elements.segmentedContainer.innerHTML = `
            <h3>K-Means Segmented Image</h3>
            <p class="placeholder-text">Processed image will appear here</p>
        `;
        this.hideProcessingStats();
    }

    /**
     * Main image processing function
     */
    async processImage() {
        if (!this.originalImage || this.isProcessing) return;

        this.isProcessing = true;
        this.elements.processBtn.disabled = true;
        this.elements.processBtn.textContent = 'Processing...';

        try {
            const k = parseInt(this.elements.kSlider.value);

            // Show processing UI
            this.showProcessingState(k);

            // Create canvas and get image data
            const { canvas, imageData } = this.prepareImageData();

            // Process image with K-means
            const result = await this.kmeansProcessor.segment(
                imageData,
                k,
                (message, percentage) => this.updateProcessingProgress(message, percentage)
            );

            // Display results
            await this.displayResults(result, k);

        } catch (error) {
            console.error('Processing error:', error);
            this.showStatus(`Error processing image: ${error.message}`, 'error');
            this.clearSegmentedImage();
        } finally {
            this.isProcessing = false;
            this.elements.processBtn.disabled = false;
            this.elements.processBtn.textContent = 'Process Image';
        }
    }

    /**
     * Prepare image data for processing
     */
    prepareImageData() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Scale down large images for performance
        const maxDimension = 800;
        let { width, height } = this.originalImage;

        if (Math.max(width, height) > maxDimension) {
            const scale = maxDimension / Math.max(width, height);
            width = Math.round(width * scale);
            height = Math.round(height * scale);
        }

        canvas.width = width;
        canvas.height = height;

        ctx.drawImage(this.originalImage, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);

        return { canvas, imageData };
    }

    /**
     * Show processing state UI
     */
    showProcessingState(k) {
        this.showStatus(`Starting K-means segmentation with ${k} clusters...`, 'info');
        this.elements.segmentedContainer.innerHTML = `
            <h3>K-Means Segmented Image</h3>
            <div class="loading">
                <div>Processing with ${k} clusters...</div>
                <div id="progressBar" style="width: 100%; background: #ddd; border-radius: 10px; margin-top: 10px;">
                    <div id="progressFill" style="width: 0%; height: 20px; background: var(--primary-color); border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
                <div id="progressText" style="margin-top: 5px; font-size: 0.9em;">Initializing...</div>
            </div>
        `;
    }

    /**
     * Update processing progress
     */
    updateProcessingProgress(message, percentage) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        if (progressFill) {
            progressFill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
        }
        if (progressText) {
            progressText.textContent = message;
        }
    }

    /**
     * Display processing results
     */
    async displayResults(result, k) {
        const { imageData, stats } = result;

        // Create segmented image canvas and render directly
        // (Avoids toDataURL() which can fail with file:// protocol or large images)
        const segmentedCanvas = document.createElement('canvas');
        const segmentedCtx = segmentedCanvas.getContext('2d');
        segmentedCanvas.width = imageData.width;
        segmentedCanvas.height = imageData.height;
        segmentedCtx.putImageData(imageData, 0, 0);

        // Display segmented image using canvas directly
        this.elements.segmentedContainer.innerHTML = `<h3>K-Means Segmented Image (K=${k})</h3>`;

        // Style the canvas like an image
        segmentedCanvas.style.maxWidth = '100%';
        segmentedCanvas.style.maxHeight = '400px';
        segmentedCanvas.style.borderRadius = '10px';
        segmentedCanvas.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.1)';
        this.elements.segmentedContainer.appendChild(segmentedCanvas);

        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'margin-top: 10px; font-size: 0.9em; color: var(--text-light);';
        infoDiv.textContent = `${stats.iterations} iterations \u2022 ${stats.converged ? 'Converged' : 'Max iterations reached'}`;
        this.elements.segmentedContainer.appendChild(infoDiv);

        // Show processing statistics
        this.showProcessingStats(stats);

        // Show success message
        const convergenceText = stats.converged ? 'converged' : `reached maximum iterations (${stats.iterations})`;
        this.showStatus(
            `Image successfully segmented into ${k} clusters! Algorithm ${convergenceText} in ${stats.processingTime}ms.`,
            'success'
        );
    }

    /**
     * Show processing statistics
     */
    showProcessingStats(stats) {
        this.elements.processingStats.innerHTML = `
            <h4>Processing Statistics</h4>
            <div class="stat-item">
                <span>Clusters (K):</span>
                <span>${stats.clusters}</span>
            </div>
            <div class="stat-item">
                <span>Iterations:</span>
                <span>${stats.iterations}</span>
            </div>
            <div class="stat-item">
                <span>Convergence:</span>
                <span>${stats.converged ? 'Yes' : 'No (Max reached)'}</span>
            </div>
            <div class="stat-item">
                <span>Processing Time:</span>
                <span>${stats.processingTime}ms</span>
            </div>
            <div class="stat-item">
                <span>Pixels Processed:</span>
                <span>${stats.pixelCount.toLocaleString()}</span>
            </div>
        `;
        this.elements.processingStats.style.display = 'block';
    }

    /**
     * Hide processing statistics
     */
    hideProcessingStats() {
        this.elements.processingStats.style.display = 'none';
    }

    /**
     * Show status message
     */
    showStatus(message, type = 'info') {
        this.elements.status.innerHTML = `<div class="status ${type}">${message}</div>`;

        // Auto-hide after 8 seconds for success messages
        if (type === 'success') {
            setTimeout(() => {
                if (this.elements.status.innerHTML.includes(message)) {
                    this.elements.status.innerHTML = '';
                }
            }, 8000);
        }
    }

    /**
     * Download segmented image
     */
    downloadSegmentedImage() {
        const canvas = this.elements.segmentedContainer.querySelector('canvas');
        if (!canvas) {
            this.showStatus('No segmented image to download.', 'error');
            return;
        }

        const link = document.createElement('a');
        link.download = `segmented_k${this.elements.kSlider.value}_${Date.now()}.png`;
        link.href = canvas.toDataURL();
        link.click();
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.kmeansApp = new KMeansImageSegmentation();

    // Add download functionality if needed
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's' && window.kmeansApp) {
            e.preventDefault();
            window.kmeansApp.downloadSegmentedImage();
        }
    });

    console.log('K-Means Image Segmentation System initialized');
    console.log('Shortcuts: Ctrl+O (Upload), Ctrl+Enter (Process), Ctrl+S (Download)');
});