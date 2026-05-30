#!/usr/bin/env python3
"""
K-Means Image Segmentation - Python Implementation
Based on Vibhav Gogate's Java implementation
Adapted for Python with enhanced features

Usage: python kmeans_python.py <input-image> <k> <output-image>
Example: python kmeans_python.py input.jpg 5 output.jpg

Author: Converted from Java implementation
The University of Texas at Dallas approach
"""

import sys
import numpy as np
from PIL import Image
import argparse
import time
from typing import List, Tuple, Optional
import os


class KMeansImageSegmentation:
    """
    K-Means clustering implementation for image segmentation.
    
    This class provides methods to segment images using the K-means clustering
    algorithm, following Lloyd's algorithm approach.
    """
    
    def __init__(self, max_iterations: int = 20, convergence_threshold: float = 1.0):
        """
        Initialize K-Means segmentation parameters.
        
        Args:
            max_iterations: Maximum number of iterations for convergence
            convergence_threshold: Threshold for convergence detection
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Image as numpy array in RGB format
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")
    
    def save_image(self, image_array: np.ndarray, output_path: str) -> None:
        """
        Save segmented image to file.
        
        Args:
            image_array: Segmented image as numpy array
            output_path: Path for output image
        """
        try:
            image = Image.fromarray(image_array.astype(np.uint8))
            image.save(output_path)
            print(f"Segmented image saved to: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving image {output_path}: {str(e)}")
    
    def extract_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Extract RGB pixel values from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Flattened array of RGB pixels (n_pixels, 3)
        """
        height, width, channels = image.shape
        pixels = image.reshape(-1, channels)
        return pixels
    
    def initialize_centroids(self, pixels: np.ndarray, k: int) -> np.ndarray:
        """
        Initialize k centroids using K-means++ method for better convergence.
        
        Args:
            pixels: Array of pixel RGB values
            k: Number of clusters
            
        Returns:
            Initial centroids as numpy array (k, 3)
        """
        n_pixels = pixels.shape[0]
        centroids = np.zeros((k, 3))
        
        # Choose first centroid randomly
        centroids[0] = pixels[np.random.randint(0, n_pixels)]
        
        for i in range(1, k):
            # Calculate distances from each pixel to nearest existing centroid
            distances = np.array([min([np.linalg.norm(pixel - c)**2 for c in centroids[:i]]) 
                                for pixel in pixels])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, cumulative_prob in enumerate(cumulative_probabilities):
                if r < cumulative_prob:
                    centroids[i] = pixels[j]
                    break
                    
        return centroids
    
    def assign_pixels_to_centroids(self, pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each pixel to the closest centroid.
        
        Args:
            pixels: Array of pixel RGB values
            centroids: Current centroids
            
        Returns:
            Array of cluster assignments for each pixel
        """
        distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def update_centroids(self, pixels: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """
        Update centroid positions based on assigned pixels.
        
        Args:
            pixels: Array of pixel RGB values
            assignments: Current cluster assignments
            k: Number of clusters
            
        Returns:
            Updated centroids
        """
        new_centroids = np.zeros((k, 3))
        
        for i in range(k):
            cluster_pixels = pixels[assignments == i]
            if len(cluster_pixels) > 0:
                new_centroids[i] = cluster_pixels.mean(axis=0)
            else:
                # If no pixels assigned, keep the old centroid
                new_centroids[i] = pixels[np.random.randint(0, len(pixels))]
                
        return new_centroids
    
    def check_convergence(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """
        Check if centroids have converged.
        
        Args:
            old_centroids: Previous centroids
            new_centroids: Current centroids
            
        Returns:
            True if converged, False otherwise
        """
        distances = np.linalg.norm(old_centroids - new_centroids, axis=1)
        return np.all(distances < self.convergence_threshold)
    
    def kmeans_algorithm(self, pixels: np.ndarray, k: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Main K-means clustering algorithm.
        
        Args:
            pixels: Array of pixel RGB values
            k: Number of clusters
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (final_centroids, assignments, statistics)
        """
        start_time = time.time()
        
        # Initialize centroids
        centroids = self.initialize_centroids(pixels, k)
        if verbose:
            print(f"Initialized {k} centroids using K-means++")
        
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            old_centroids = centroids.copy()
            
            # Assignment step
            assignments = self.assign_pixels_to_centroids(pixels, centroids)
            
            # Update step
            centroids = self.update_centroids(pixels, assignments, k)
            
            # Check convergence
            converged = self.check_convergence(old_centroids, centroids)
            
            iteration += 1
            
            if verbose:
                print(f"Iteration {iteration}/{self.max_iterations} - "
                      f"Converged: {converged}")
        
        end_time = time.time()
        
        statistics = {
            'iterations': iteration,
            'converged': converged,
            'processing_time': end_time - start_time,
            'total_pixels': len(pixels),
            'clusters': k
        }
        
        return centroids, assignments, statistics
    
    def create_segmented_image(self, original_shape: Tuple[int, int, int], 
                             centroids: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """
        Create segmented image by replacing pixel colors with centroid colors.
        
        Args:
            original_shape: Shape of the original image
            centroids: Final centroids
            assignments: Pixel assignments to clusters
            
        Returns:
            Segmented image as numpy array
        """
        segmented_pixels = centroids[assignments]
        segmented_image = segmented_pixels.reshape(original_shape)
        return segmented_image
    
    def segment_image(self, image_path: str, k: int, output_path: str, verbose: bool = True) -> dict:
        """
        Complete image segmentation pipeline.
        
        Args:
            image_path: Input image path
            k: Number of clusters
            output_path: Output image path
            verbose: Whether to print progress
            
        Returns:
            Dictionary with processing statistics
        """
        if verbose:
            print(f"Starting K-means image segmentation...")
            print(f"Input: {image_path}")
            print(f"Clusters (K): {k}")
            print(f"Output: {output_path}")
            print("-" * 50)
        
        # Load and prepare image
        original_image = self.load_image(image_path)
        if verbose:
            print(f"Loaded image: {original_image.shape[1]}x{original_image.shape[0]} pixels")
        
        pixels = self.extract_pixels(original_image)
        
        # Run K-means algorithm
        centroids, assignments, stats = self.kmeans_algorithm(pixels, k, verbose)
        
        # Create segmented image
        segmented_image = self.create_segmented_image(original_image.shape, centroids, assignments)
        
        # Save result
        self.save_image(segmented_image, output_path)
        
        if verbose:
            print("-" * 50)
            print("Processing completed!")
            print(f"Iterations: {stats['iterations']}")
            print(f"Converged: {stats['converged']}")
            print(f"Processing time: {stats['processing_time']:.2f} seconds")
            print(f"Pixels processed: {stats['total_pixels']:,}")
        
        return stats
    
    def get_color_palette(self, centroids: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Get the color palette from final centroids.
        
        Args:
            centroids: Final centroids
            
        Returns:
            List of RGB tuples representing the color palette
        """
        return [(int(c[0]), int(c[1]), int(c[2])) for c in centroids]
    
    def calculate_histogram(self, assignments: np.ndarray, k: int) -> dict:
        """
        Calculate color histogram for the segmented image.
        
        Args:
            assignments: Cluster assignments
            k: Number of clusters
            
        Returns:
            Dictionary with histogram data
        """
        unique, counts = np.unique(assignments, return_counts=True)
        histogram = np.zeros(k, dtype=int)
        histogram[unique] = counts
        
        total_pixels = len(assignments)
        percentages = (histogram / total_pixels) * 100
        
        return {
            'counts': histogram.tolist(),
            'percentages': percentages.tolist(),
            'total_pixels': total_pixels
        }


def main():
    """Main function to handle command line arguments and run segmentation."""
    parser = argparse.ArgumentParser(
        description='K-Means Image Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kmeans_python.py input.jpg 5 output.jpg
  python kmeans_python.py photo.png 3 segmented.png --max-iter 30
  python kmeans_python.py image.jpg 4 result.jpg --quiet
        """
    )
    
    parser.add_argument('input_image', help='Input image path')
    parser.add_argument('k', type=int, help='Number of clusters (2-20 recommended)')
    parser.add_argument('output_image', help='Output image path')
    parser.add_argument('--max-iter', type=int, default=20, 
                       help='Maximum iterations (default: 20)')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Convergence threshold (default: 1.0)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.")
        sys.exit(1)
    
    if args.k < 2 or args.k > 50:
        print("Error: Number of clusters (k) should be between 2 and 50.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize and run K-means segmentation
        kmeans = KMeansImageSegmentation(
            max_iterations=args.max_iter,
            convergence_threshold=args.threshold
        )
        
        stats = kmeans.segment_image(
            args.input_image, 
            args.k, 
            args.output_image,
            verbose=not args.quiet
        )
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()