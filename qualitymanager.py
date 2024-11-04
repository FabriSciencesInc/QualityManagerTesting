import cv2
import numpy as np
from PIL import Image
import pillow_heif
from pathlib import Path
from typing import Tuple, Dict, Any
import os

class QualityChecker:
    def __init__(self):
        """Initialize the quality checker."""
        # Quality thresholds
        self.thresholds = {
            'brightness': {'lower': -2.0, 'upper': 2.0},
            'sharpness': 50.0,
            'grain': 30.0,
            'contrast': {'lower': 20.0, 'upper': 80.0}
        }

    def read_heic_image(self, image_path: str) -> np.ndarray:
        """Read HEIC image and convert to RGB numpy array."""
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
        )
        return np.array(image)

    def measure_graininess_and_sharpness(self, gray_image: np.ndarray) -> Tuple[float, float]:
        """Measure image graininess and sharpness using Fourier analysis."""
        # Convert to float32
        float_image = gray_image.astype(np.float32)
        
        # Apply DFT
        dft = cv2.dft(float_image, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        # Get image dimensions
        rows, cols = gray_image.shape
        center_row, center_col = rows//2, cols//2
        
        # Create masks for different frequency ranges
        y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
        dist_from_center = np.sqrt(x*x + y*y)
        
        # High frequency mask (graininess)
        high_freq_mask = dist_from_center > min(center_row, center_col)/2
        
        # Mid frequency mask (sharpness)
        mid_freq_mask = (dist_from_center > min(center_row, center_col)*0.1) & \
                       (dist_from_center < min(center_row, center_col)*0.75)
        
        # Calculate metrics
        graininess = np.mean(magnitude_spectrum[high_freq_mask])
        sharpness = np.mean(magnitude_spectrum[mid_freq_mask])
        
        return graininess, sharpness

    def measure_contrast(self, gray_image: np.ndarray) -> float:
        """Calculate image contrast using percentile method."""
        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
        # Calculate cumulative histogram
        cum_hist = np.cumsum(hist)
        total_pixels = cum_hist[-1]
        
        # Find 5th and 95th percentiles
        p5 = np.searchsorted(cum_hist, total_pixels * 0.05)
        p95 = np.searchsorted(cum_hist, total_pixels * 0.95)
        
        # Calculate and normalize contrast
        contrast = (p95 - p5) / 255.0 * 100.0
        return contrast

    def check_image_quality(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform quality checks on the image using provided mask or creating one.
        
        Args:
            image: Input image as numpy array
            mask: Optional binary mask where 0 is background, 255 is foreground
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        # Create or process mask
        if mask is None:
            mask = self.create_foreground_mask(image)
        
        # Ensure mask is the right size
        if mask.shape != gray_image.shape:
            mask = cv2.resize(mask, (gray_image.shape[1], gray_image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply mask to grayscale image
        masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        # Measure quality metrics
        graininess, sharpness = self.measure_graininess_and_sharpness(masked_gray)
        contrast = self.measure_contrast(masked_gray)
        
        # Prepare results
        results = {
            'metrics': {
                'graininess': float(graininess),
                'sharpness': float(sharpness),
                'contrast': float(contrast)
            },
            'passes_threshold': {
                'graininess': graininess >= self.thresholds['grain'],
                'sharpness': sharpness >= self.thresholds['sharpness'],
                'contrast': self.thresholds['contrast']['lower'] <= contrast <= self.thresholds['contrast']['upper']
            }
        }
        
        return results

    def process_folder(self, folder_path: str, output_folder: str = None):
        """Process all HEIC images in a folder."""
        if output_folder is None:
            output_folder = os.path.join(folder_path, 'results')
        
        os.makedirs(output_folder, exist_ok=True)
        
        folder = Path(folder_path)
        heic_files = list(folder.glob('*.heic')) + list(folder.glob('*.HEIC'))
        
        results = {}
        for file_path in heic_files:
            print(f"Processing {file_path.name}...")
            
            # Read image
            image = self.read_heic_image(str(file_path))
            
            # Create mask
            mask = self.create_foreground_mask(image)
            
            # Check quality using mask
            quality_results = self.check_image_quality(image, mask)
            
            # Store results
            results[file_path.name] = quality_results
            
            # Save mask if output folder is specified
            if output_folder:
                mask_path = os.path.join(output_folder, f"{file_path.stem}_mask.png")
                cv2.imwrite(mask_path, mask)
        
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize quality checker
    checker = QualityChecker()
    
    # Process a folder of HEIC images
    folder_path = "test"
    results = checker.process_folder(folder_path)
    
    # Print results
    for image_name, quality_results in results.items():
        print(f"\nResults for {image_name}:")
        print(f"Metrics: {quality_results['metrics']}")
        print(f"Passes thresholds: {quality_results['passes_threshold']}")