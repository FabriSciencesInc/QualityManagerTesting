import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Set
from qualitymanager import QualityChecker
from scipy import stats
import cv2
import hashlib

class ThresholdAnalyzer:
    def __init__(self, folder_paths: Union[str, List[str]], brightness_range: Tuple[float, float]):
        """
        Initialize ThresholdAnalyzer with multiple folder paths.
        
        Args:
            folder_paths: Single folder path or list of folder paths
            brightness_range: Tuple of (min_brightness, max_brightness)
        """
        # Convert single path to list for consistent handling
        self.folder_paths = [Path(folder_paths)] if isinstance(folder_paths, str) else [Path(p) for p in folder_paths]
        self.brightness_range = brightness_range
        self.quality_checker = QualityChecker()
        
        # Track processed images to handle duplicates
        self.processed_images: Set[str] = set()
        
        # Weights for different metrics (must sum to 1.0)
        self.metric_weights = {
            'contrast': 0.4,
            'graininess': 0.3,
            'sharpness': 0.3
        }
        
        self.good_metrics = {
            'contrast': [],
            'graininess': [],
            'sharpness': []
        }
        self.bad_metrics = {
            'contrast': [],
            'graininess': [],
            'sharpness': []
        }
        
        # Mapping of image hashes to their metadata
        self.image_metadata: Dict[str, Dict] = {}

    def generate_image_hash(self, image_path: Path) -> str:
        """
        Generate a unique hash for an image based on its content and path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Unique hash string for the image
        """
        try:
            # Read first 8KB of file for quick hashing
            with open(image_path, 'rb') as f:
                content = f.read(8192)
            
            # Combine file content hash with filename for uniqueness
            hasher = hashlib.md5()
            hasher.update(content)
            hasher.update(str(image_path).encode())
            return hasher.hexdigest()
            
        except Exception as e:
            print(f"Error generating hash for {image_path}: {e}")
            return None

    def find_mask_path(self, image_path: Path) -> Path:
        """
        Find the corresponding mask file for an image, checking both local and parent results folders.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the mask file if found, None otherwise
        """
        # Check immediate results folder
        local_mask = image_path.parent / 'results' / f"{image_path.stem}_mask.png"
        if local_mask.exists():
            return local_mask
            
        # Check parent results folder
        parent_mask = image_path.parent.parent / 'results' / f"{image_path.stem}_mask.png"
        if parent_mask.exists():
            return parent_mask
            
        return None

    def load_classifications(self) -> Dict:
        """
        Load and merge classifications from all folder paths.
        
        Returns:
            Dictionary of merged classifications
        """
        merged_classifications = {}
        found_valid_file = False
        
        for folder_path in self.folder_paths:
            classification_file = folder_path / 'classifications.json'
            if not classification_file.exists():
                print(f"Warning: No classifications file found at {classification_file}")
                continue
                
            try:
                with open(classification_file, 'r') as f:
                    classifications = json.load(f)
                    found_valid_file = True
                    
                # Store source folder with classification for handling duplicates
                for image_name, classification in classifications.items():
                    if image_name in merged_classifications:
                        print(f"Warning: Duplicate image name found: {image_name}")
                        # Keep the first occurrence
                        continue
                    merged_classifications[image_name] = {
                        'classification': classification,
                        'source_folder': str(folder_path)
                    }
                    
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {classification_file}")
                continue
                
        if not found_valid_file:
            raise FileNotFoundError("No valid classifications found in any provided folders")
            
        return merged_classifications

    def collect_metrics(self):
        """Collect metrics for all classified images across all folders."""
        classifications = self.load_classifications()
        
        for image_name, info in classifications.items():
            if info['classification'] not in ['good', 'bad']:
                continue
                
            # Construct full path using source folder
            image_path = Path(info['source_folder']) / image_name
            if not image_path.exists():
                print(f"Warning: Image {image_name} not found at {image_path}")
                continue
                
            # Generate unique identifier for image
            image_hash = self.generate_image_hash(image_path)
            if not image_hash:
                continue
                
            if image_hash in self.processed_images:
                print(f"Warning: Skipping duplicate image content: {image_name}")
                continue
            
            self.processed_images.add(image_hash)
            
            # Find mask file
            mask_path = self.find_mask_path(image_path)
            
            try:
                # Read and process image
                image = self.quality_checker.read_heic_image(str(image_path))
                if mask_path:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_not(mask)
                else:
                    print(f"Warning: No mask found for {image_name}")
                    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
                
                # Get metrics
                results = self.quality_checker.check_image_quality(image, mask)
                
                # Store metrics based on classification
                metrics_dict = self.good_metrics if info['classification'] == 'good' else self.bad_metrics
                for metric in ['contrast', 'graininess', 'sharpness']:
                    metrics_dict[metric].append(results['metrics'][metric])
                
                # Store metadata for reference
                self.image_metadata[image_hash] = {
                    'path': str(image_path),
                    'classification': info['classification'],
                    'metrics': results['metrics']
                }
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

    def remove_outliers(self, data: np.ndarray, zscore_threshold: float = 2.5) -> np.ndarray:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Input array of values
            zscore_threshold: Z-score threshold for outlier detection (default 2.5)
        
        Returns:
            Cleaned array with outliers removed
        """
        if len(data) < 4:  # Need enough data for meaningful statistics
            return data
            
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(data))
        
        # Keep only values within threshold
        return data[z_scores < zscore_threshold]

    def calculate_robust_statistics(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate robust mean and standard deviation using trimmed statistics.
        
        Args:
            data: Input array of values
        
        Returns:
            Tuple of (robust_mean, robust_std)
        """
        if len(data) < 4:
            return np.mean(data), np.std(data)
            
        # Use trimmed mean (removing top and bottom 10%)
        trimmed_mean = stats.trim_mean(data, 0.1)
        
        # Calculate MAD (Median Absolute Deviation) for robust std
        mad = stats.median_abs_deviation(data)
        robust_std = mad * 1.4826  # Scale factor to make MAD comparable to std
        
        return trimmed_mean, robust_std

    def calculate_thresholds(self, percentile: float = 10, zscore_threshold: float = 2.5) -> Dict:
        """
        Calculate thresholds with outlier handling.
        
        Args:
            percentile: Percentile to use for threshold calculation
            zscore_threshold: Z-score threshold for outlier detection
        """
        self.collect_metrics()
        thresholds = {}
        
        for metric in ['contrast', 'graininess', 'sharpness']:
            # Convert to numpy arrays
            good_values = np.array(self.good_metrics[metric])
            bad_values = np.array(self.bad_metrics[metric])
            
            if len(good_values) == 0 or len(bad_values) == 0:
                print(f"Warning: No data for {metric}")
                continue
            
            # Remove outliers
            good_values_clean = self.remove_outliers(good_values, zscore_threshold)
            bad_values_clean = self.remove_outliers(bad_values, zscore_threshold)
            
            # Calculate robust statistics
            good_mean, good_std = self.calculate_robust_statistics(good_values_clean)
            bad_mean, bad_std = self.calculate_robust_statistics(bad_values_clean)
            
            # Calculate initial thresholds using cleaned data
            lower_threshold = np.percentile(good_values_clean, percentile)
            upper_threshold = np.percentile(good_values_clean, 100 - percentile)
            
            # Adjust thresholds based on bad image patterns
            if bad_mean < good_mean:
                # Bad images tend to have lower values
                separation_point = (good_mean - good_std + bad_mean + bad_std) / 2
                lower_threshold = max(lower_threshold, separation_point)
            
            if bad_mean > good_mean:
                # Bad images tend to have higher values
                separation_point = (good_mean + good_std + bad_mean - bad_std) / 2
                upper_threshold = min(upper_threshold, separation_point)
            
            thresholds[metric] = {
                'lower': float(lower_threshold),
                'upper': float(upper_threshold)
            }
        
        # Add static brightness thresholds
        thresholds['brightness'] = {
            'lower': self.brightness_range[0],
            'upper': self.brightness_range[1]
        }
        
        return thresholds

    def save_thresholds(self, thresholds: Dict, output_file: str = "calculated_thresholds"):
        """
        Save calculated thresholds and analysis metadata to a JSON file.
        
        Args:
            thresholds: Dictionary of calculated thresholds
            output_file: Optional output file path
        """
        if output_file is None:
            output_file = self.folder_paths[0] / 'quality_thresholds.json'
                
        with open(output_file, 'w') as f:
            json.dump({
                'thresholds': thresholds,
                'weights': self.metric_weights,
                'metadata': {
                    'source_folders': [str(p) for p in self.folder_paths],
                    'total_images_processed': len(self.processed_images),
                    'good_samples': {k: len(v) for k, v in self.good_metrics.items()},
                    'bad_samples': {k: len(v) for k, v in self.bad_metrics.items()}
                }
            }, f, indent=4)

    def print_threshold_analysis(self, thresholds: Dict):
        """Print detailed analysis including outlier information."""
        print("\nThreshold Analysis:")
        print("-" * 50)
        
        for metric in thresholds:
            print(f"\n{metric.capitalize()}:")
            print(f"  Range: {thresholds[metric]['lower']:.2f} - {thresholds[metric]['upper']:.2f}")
            
            if metric != 'brightness':
                good_values = np.array(self.good_metrics[metric])
                bad_values = np.array(self.bad_metrics[metric])
                
                # Clean values
                good_clean = self.remove_outliers(good_values)
                bad_clean = self.remove_outliers(bad_values)
                
                # Calculate statistics
                good_mean, good_std = self.calculate_robust_statistics(good_clean)
                bad_mean, bad_std = self.calculate_robust_statistics(bad_clean)
                
                print(f"  Good images (after outlier removal):")
                print(f"    Samples: {len(good_clean)} (removed {len(good_values) - len(good_clean)} outliers)")
                print(f"    Robust Mean: {good_mean:.2f}")
                print(f"    Robust Std:  {good_std:.2f}")
                print(f"    Range: {np.min(good_clean):.2f} - {np.max(good_clean):.2f}")
                
                print(f"  Bad images (after outlier removal):")
                print(f"    Samples: {len(bad_clean)} (removed {len(bad_values) - len(bad_clean)} outliers)")
                print(f"    Robust Mean: {bad_mean:.2f}")
                print(f"    Robust Std:  {bad_std:.2f}")
                print(f"    Range: {np.min(bad_clean):.2f} - {np.max(bad_clean):.2f}")

def main():
    # Example usage with multiple folders
    folder_paths = [
        "/Users/amoghpanhale/Documents/GitHub/QualityManagerTesting/iPhone 12 Pro - Batch 2/batch_2",
        "/Users/amoghpanhale/Documents/GitHub/QualityManagerTesting/iPhone 12 Pro - Batch 2/batch_3",
        "/Users/amoghpanhale/Documents/GitHub/QualityManagerTesting/iPhone 12 Pro - Batch 2/batch_4"
    ]
    brightness_range = (-2.0, 10.0)
    
    analyzer = ThresholdAnalyzer(folder_paths, brightness_range)
    thresholds = analyzer.calculate_thresholds(percentile=5)
    
    # Save thresholds
    analyzer.save_thresholds(thresholds)
    
    # Print analysis
    analyzer.print_threshold_analysis(thresholds)

if __name__ == "__main__":
    main()