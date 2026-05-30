/**
 * K-Means Clustering Algorithm for Image Segmentation
 * Based on Lloyd's algorithm
 * Author: Adapted from Vibhav Gogate's Java implementation
 */

class KMeans {
    constructor() {
        this.maxIterations = 20;
        this.convergenceThreshold = 1.0;
        this.onProgressUpdate = null;
    }

    /**
     * Main K-means segmentation function
     * @param {ImageData} imageData - Canvas ImageData object
     * @param {number} k - Number of clusters
     * @param {Function} progressCallback - Optional progress callback
     * @returns {Promise<ImageData>} Segmented image data
     */
    async segment(imageData, k, progressCallback = null) {
        this.onProgressUpdate = progressCallback;
        
        const startTime = performance.now();
        const pixels = this.extractPixels(imageData);
        const centroids = this.initializeCentroids(pixels, k);
        
        this.updateProgress('Initializing K-means algorithm...', 0);
        
        const { finalCentroids, iterations, converged } = await this.kmeansAlgorithm(pixels, centroids, k);
        const segmentedImageData = this.createSegmentedImage(imageData, pixels, finalCentroids);
        
        const endTime = performance.now();
        const processingTime = Math.round(endTime - startTime);
        
        // Return processing statistics
        return {
            imageData: segmentedImageData,
            stats: {
                iterations,
                converged,
                processingTime,
                clusters: k,
                pixelCount: pixels.length
            }
        };
    }

    /**
     * Extract RGB pixel values from ImageData
     * @param {ImageData} imageData 
     * @returns {Array} Array of pixel objects
     */
    extractPixels(imageData) {
        const data = imageData.data;
        const pixels = [];
        
        for (let i = 0; i < data.length; i += 4) {
            pixels.push({
                r: data[i],
                g: data[i + 1],
                b: data[i + 2],
                a: data[i + 3],
                cluster: -1
            });
        }
        
        return pixels;
    }

    /**
     * Initialize centroids using random pixel selection
     * @param {Array} pixels - Array of pixels
     * @param {number} k - Number of clusters
     * @returns {Array} Initial centroids
     */
    initializeCentroids(pixels, k) {
        const centroids = [];
        const usedIndices = new Set();
        
        // Use k-means++ initialization for better results
        for (let i = 0; i < k; i++) {
            if (i === 0) {
                // Choose first centroid randomly
                const randomIndex = Math.floor(Math.random() * pixels.length);
                const randomPixel = pixels[randomIndex];
                centroids.push({
                    r: randomPixel.r,
                    g: randomPixel.g,
                    b: randomPixel.b,
                    id: i
                });
                usedIndices.add(randomIndex);
            } else {
                // Choose subsequent centroids based on distance from existing centroids
                let maxDistance = -1;
                let bestPixelIndex = 0;
                
                for (let j = 0; j < pixels.length; j++) {
                    if (usedIndices.has(j)) continue;
                    
                    // Find minimum distance to existing centroids
                    let minDistanceToExisting = Infinity;
                    for (const centroid of centroids) {
                        const distance = this.calculateDistance(pixels[j], centroid);
                        minDistanceToExisting = Math.min(minDistanceToExisting, distance);
                    }
                    
                    if (minDistanceToExisting > maxDistance) {
                        maxDistance = minDistanceToExisting;
                        bestPixelIndex = j;
                    }
                }
                
                const bestPixel = pixels[bestPixelIndex];
                centroids.push({
                    r: bestPixel.r,
                    g: bestPixel.g,
                    b: bestPixel.b,
                    id: i
                });
                usedIndices.add(bestPixelIndex);
            }
        }
        
        return centroids;
    }

    /**
     * Main K-means algorithm loop
     * @param {Array} pixels - Array of pixels
     * @param {Array} centroids - Initial centroids
     * @param {number} k - Number of clusters
     * @returns {Object} Results with final centroids, iterations, and convergence status
     */
    async kmeansAlgorithm(pixels, centroids, k) {
        let iteration = 0;
        let converged = false;
        
        while (iteration < this.maxIterations && !converged) {
            this.updateProgress(`Running iteration ${iteration + 1}/${this.maxIterations}...`, 
                             (iteration / this.maxIterations) * 100);
            
            // Assignment step: assign pixels to closest centroid
            this.assignPixelsToCentroids(pixels, centroids);
            
            // Update step: recalculate centroids
            const newCentroids = this.updateCentroids(pixels, centroids, k);
            
            // Check convergence
            converged = this.checkConvergence(centroids, newCentroids);
            
            // Update centroids for next iteration
            for (let i = 0; i < k; i++) {
                centroids[i].r = newCentroids[i].r;
                centroids[i].g = newCentroids[i].g;
                centroids[i].b = newCentroids[i].b;
            }
            
            iteration++;
            
            // Allow UI to update every few iterations
            if (iteration % 3 === 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }
        
        return {
            finalCentroids: centroids,
            iterations: iteration,
            converged: converged
        };
    }

    /**
     * Assign each pixel to the closest centroid
     * @param {Array} pixels - Array of pixels
     * @param {Array} centroids - Current centroids
     */
    assignPixelsToCentroids(pixels, centroids) {
        for (const pixel of pixels) {
            let minDistance = Infinity;
            let closestCentroid = 0;
            
            for (let i = 0; i < centroids.length; i++) {
                const distance = this.calculateDistance(pixel, centroids[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroid = i;
                }
            }
            
            pixel.cluster = closestCentroid;
        }
    }

    /**
     * Update centroid positions based on assigned pixels
     * @param {Array} pixels - Array of pixels with cluster assignments
     * @param {Array} centroids - Current centroids
     * @param {number} k - Number of clusters
     * @returns {Array} New centroid positions
     */
    updateCentroids(pixels, centroids, k) {
        const newCentroids = [];
        
        for (let i = 0; i < k; i++) {
            const clusterPixels = pixels.filter(pixel => pixel.cluster === i);
            
            if (clusterPixels.length === 0) {
                // If no pixels assigned to this centroid, keep it unchanged
                newCentroids.push({ ...centroids[i] });
                continue;
            }
            
            // Calculate mean color values
            const sumR = clusterPixels.reduce((sum, pixel) => sum + pixel.r, 0);
            const sumG = clusterPixels.reduce((sum, pixel) => sum + pixel.g, 0);
            const sumB = clusterPixels.reduce((sum, pixel) => sum + pixel.b, 0);
            
            newCentroids.push({
                r: Math.round(sumR / clusterPixels.length),
                g: Math.round(sumG / clusterPixels.length),
                b: Math.round(sumB / clusterPixels.length),
                id: i
            });
        }
        
        return newCentroids;
    }

    /**
     * Check if algorithm has converged
     * @param {Array} oldCentroids - Previous centroids
     * @param {Array} newCentroids - New centroids
     * @returns {boolean} True if converged
     */
    checkConvergence(oldCentroids, newCentroids) {
        for (let i = 0; i < oldCentroids.length; i++) {
            const distance = this.calculateDistance(oldCentroids[i], newCentroids[i]);
            if (distance > this.convergenceThreshold) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculate Euclidean distance between two color points
     * @param {Object} color1 - First color {r, g, b}
     * @param {Object} color2 - Second color {r, g, b}
     * @returns {number} Euclidean distance
     */
    calculateDistance(color1, color2) {
        const dr = color1.r - color2.r;
        const dg = color1.g - color2.g;
        const db = color1.b - color2.b;
        return Math.sqrt(dr * dr + dg * dg + db * db);
    }

    /**
     * Create segmented image by replacing pixel colors with centroid colors
     * @param {ImageData} originalImageData - Original image data
     * @param {Array} pixels - Pixels with cluster assignments
     * @param {Array} centroids - Final centroids
     * @returns {ImageData} Segmented image data
     */
    createSegmentedImage(originalImageData, pixels, centroids) {
        const width = originalImageData.width;
        const height = originalImageData.height;
        const segmentedImageData = new ImageData(width, height);
        const segmentedData = segmentedImageData.data;
        
        for (let i = 0; i < pixels.length; i++) {
            const pixel = pixels[i];
            const centroid = centroids[pixel.cluster];
            
            const dataIndex = i * 4;
            segmentedData[dataIndex] = centroid.r;         // Red
            segmentedData[dataIndex + 1] = centroid.g;     // Green
            segmentedData[dataIndex + 2] = centroid.b;     // Blue
            segmentedData[dataIndex + 3] = pixel.a;        // Alpha (preserve original)
        }
        
        return segmentedImageData;
    }

    /**
     * Update progress callback
     * @param {string} message - Progress message
     * @param {number} percentage - Progress percentage
     */
    updateProgress(message, percentage) {
        if (this.onProgressUpdate) {
            this.onProgressUpdate(message, Math.round(percentage));
        }
    }

    /**
     * Get cluster color palette
     * @param {Array} centroids - Final centroids
     * @returns {Array} Array of color strings
     */
    getColorPalette(centroids) {
        return centroids.map(centroid => 
            `rgb(${centroid.r}, ${centroid.g}, ${centroid.b})`
        );
    }

    /**
     * Calculate color histogram for the segmented image
     * @param {Array} pixels - Pixels with cluster assignments
     * @param {number} k - Number of clusters
     * @returns {Object} Histogram data
     */
    calculateHistogram(pixels, k) {
        const histogram = new Array(k).fill(0);
        
        for (const pixel of pixels) {
            if (pixel.cluster >= 0 && pixel.cluster < k) {
                histogram[pixel.cluster]++;
            }
        }
        
        const totalPixels = pixels.length;
        const percentages = histogram.map(count => 
            Math.round((count / totalPixels) * 100 * 10) / 10
        );
        
        return {
            counts: histogram,
            percentages: percentages,
            totalPixels: totalPixels
        };
    }
}