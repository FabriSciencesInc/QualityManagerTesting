# import cv2
# import numpy as np
# from PIL import Image
# import pillow_heif
# from pathlib import Path
# from typing import Tuple, Dict, Any
# import os

# class QualityChecker:
#     def __init__(self):
#         """Initialize the quality checker."""
#         # Quality thresholds
#         self.thresholds = {
#             'brightness': {'lower': -2.0, 'upper': 2.0},
#             'sharpness': 50.0,
#             'grain': 30.0,
#             'contrast': {'lower': 20.0, 'upper': 80.0}
#         }

#     def read_heic_image(self, image_path: str) -> np.ndarray:
#         """Read HEIC image and convert to RGB numpy array."""
#         heif_file = pillow_heif.read_heif(image_path)
#         image = Image.frombytes(
#             heif_file.mode, 
#             heif_file.size, 
#             heif_file.data,
#             "raw",
#         )
#         return np.array(image)

#     def measure_graininess_and_sharpness(self, gray_image: np.ndarray) -> Tuple[float, float]:
#         """Measure image graininess and sharpness using Fourier analysis."""
#         # Convert to float32
#         float_image = gray_image.astype(np.float32)
        
#         # Apply DFT
#         dft = cv2.dft(float_image, flags=cv2.DFT_COMPLEX_OUTPUT)
#         dft_shift = np.fft.fftshift(dft)
        
#         # Calculate magnitude spectrum
#         magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
#         # Get image dimensions
#         rows, cols = gray_image.shape
#         center_row, center_col = rows//2, cols//2
        
#         # Create masks for different frequency ranges
#         y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
#         dist_from_center = np.sqrt(x*x + y*y)
        
#         # High frequency mask (graininess)
#         high_freq_mask = dist_from_center > min(center_row, center_col)/2
        
#         # Mid frequency mask (sharpness)
#         mid_freq_mask = (dist_from_center > min(center_row, center_col)*0.1) & \
#                        (dist_from_center < min(center_row, center_col)*0.75)
        
#         # Calculate metrics
#         graininess = np.mean(magnitude_spectrum[high_freq_mask])
#         sharpness = np.mean(magnitude_spectrum[mid_freq_mask])
        
#         return graininess, sharpness

#     def measure_contrast(self, gray_image: np.ndarray) -> float:
#         """Calculate image contrast using percentile method."""
#         # Calculate histogram
#         hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
#         # Calculate cumulative histogram
#         cum_hist = np.cumsum(hist)
#         total_pixels = cum_hist[-1]
        
#         # Find 5th and 95th percentiles
#         p5 = np.searchsorted(cum_hist, total_pixels * 0.05)
#         p95 = np.searchsorted(cum_hist, total_pixels * 0.95)
        
#         # Calculate and normalize contrast
#         contrast = (p95 - p5) / 255.0 * 100.0
#         return contrast

#     def check_image_quality(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
#         """
#         Perform quality checks on the image using provided mask.
        
#         Args:
#             image: Input image as numpy array
#             mask: Binary mask where 0 is background, 255 is foreground
#         """
#         # Convert to grayscale if needed
#         if len(image.shape) == 3:
#             gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         else:
#             gray_image = image
            
#         # Ensure mask is binary
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
#         # Apply mask to grayscale image
#         masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
#         # Measure quality metrics
#         graininess, sharpness = self.measure_graininess_and_sharpness(masked_gray)
#         contrast = self.measure_contrast(masked_gray)
        
#         # Prepare results
#         results = {
#             'metrics': {
#                 'graininess': float(graininess),
#                 'sharpness': float(sharpness),
#                 'contrast': float(contrast)
#             },
#             'passes_threshold': {
#                 'graininess': graininess >= self.thresholds['grain'],
#                 'sharpness': sharpness >= self.thresholds['sharpness'],
#                 'contrast': self.thresholds['contrast']['lower'] <= contrast <= self.thresholds['contrast']['upper']
#             }
#         }
        
#         return results

#     def process_folder(self, folder_path: str, masks_folder: str = None):
#         """
#         Process all HEIC images in a folder using existing masks.
        
#         Args:
#             folder_path: Path to folder containing HEIC images
#             masks_folder: Path to folder containing corresponding masks
#         """
#         if masks_folder is None:
#             masks_folder = os.path.join(folder_path, 'results')
        
#         folder = Path(folder_path)
#         masks_folder = Path(masks_folder)
#         heic_files = list(folder.glob('*.heic')) + list(folder.glob('*.HEIC'))
        
#         results = {}
#         for file_path in heic_files:
#             print(f"Processing {file_path.name}...")
            
#             # Check for existing mask
#             mask_path = masks_folder / f"{file_path.stem}_mask.png"
#             if not mask_path.exists():
#                 print(f"Skipping {file_path.name} - no mask found at {mask_path}")
#                 continue
                
#             # Read image and mask
#             image = self.read_heic_image(str(file_path))
#             mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
#             # Check quality using mask
#             quality_results = self.check_image_quality(image, mask)
            
#             # Store results
#             results[file_path.name] = quality_results
        
#         return results

import cv2
import numpy as np
from PIL import Image
import pillow_heif
from pathlib import Path
import torch
import depth_pro  # Assuming depth_pro is installed
from typing import Tuple, Dict, Any
import os

class QualityChecker:
    def __init__(self, model_device: str = None):
        """Initialize the quality checker with optional device specification."""
        self.device = self._get_torch_device() if model_device is None else torch.device(model_device)
        self.depth_model = None
        self.transform = None
        self._initialize_depth_model()
        
        # Quality thresholds
        self.thresholds = {
            'brightness': {'lower': -2.0, 'upper': 2.0},
            'sharpness': 50.0,
            'grain': 30.0,
            'contrast': {'lower': 20.0, 'upper': 80.0}
        }

    def _get_torch_device(self) -> torch.device:
        """Determine the optimal torch device."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _initialize_depth_model(self):
        """Initialize the depth prediction model."""
        print("Loading depth model...")
        self.depth_model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device,
            precision=torch.half
        )
        self.depth_model.eval()
        print("Depth model loaded successfully!")

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

    def create_subject_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a foreground mask using depth prediction."""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Process image through depth model
        transformed_image = self.transform(pil_image)
        
        with torch.no_grad():
            prediction = self.depth_model.infer(transformed_image)
            depth_map = prediction["depth"].cpu().numpy().squeeze()
        
        # Normalize depth map to 0-255 range
        depth_map = ((depth_map - depth_map.min()) * (255 / (depth_map.max() - depth_map.min()))).astype(np.uint8)
        
        # Create binary mask using Otsu's thresholding
        _, mask = cv2.threshold(depth_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask

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

    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform quality checks on the image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        # Create foreground mask
        mask = self.create_subject_mask(image)
        
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
            
            # Check quality
            quality_results = self.check_image_quality(image)
            
            # Store results
            results[file_path.name] = quality_results
            
            # Save mask if output folder is specified
            if output_folder:
                mask = self.create_subject_mask(image)
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