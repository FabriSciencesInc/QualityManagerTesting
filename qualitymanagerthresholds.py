import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from qualitymanager import QualityChecker
from scipy import stats
import cv2

class ThresholdAnalyzer:
    def __init__(self, folder_path: str, brightness_range: Tuple[float, float]):
        self.folder_path = Path(folder_path)
        self.brightness_range = brightness_range
        self.quality_checker = QualityChecker()
        
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

    def collect_metrics(self):
        """Collect metrics for all classified images."""
        try:
            classifications = self.load_classifications()
            
            for image_name, classification in classifications.items():
                if classification not in ['good', 'bad']:
                    continue
                    
                # Load image and its mask
                image_path = self.folder_path / image_name
                if not image_path.exists():
                    print(f"Warning: Image {image_name} not found")
                    continue
                    
                mask_path = self.folder_path / 'results' / f"{image_path.stem}_mask.png"
                
                try:
                    # Read and process image
                    image = self.quality_checker.read_heic_image(str(image_path))
                    if mask_path.exists():
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
                    metrics_dict = self.good_metrics if classification == 'good' else self.bad_metrics
                    for metric in ['contrast', 'graininess', 'sharpness']:
                        metrics_dict[metric].append(results['metrics'][metric])
                        
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
                    
        except FileNotFoundError:
            print("No classifications.json file found. Please classify some images first.")
            raise

    def load_classifications(self) -> Dict:
        """Load the classifications from JSON file."""
        classification_file = self.folder_path / 'classifications.json'
        if not classification_file.exists():
            raise FileNotFoundError(f"No classifications file found at {classification_file}")
                
        with open(classification_file, 'r') as f:
            return json.load(f)

    def save_thresholds(self, thresholds: Dict, output_file: str = None):
        """Save calculated thresholds to a JSON file."""
        if output_file is None:
            output_file = self.folder_path / 'quality_thresholds.json'
                
        with open(output_file, 'w') as f:
            json.dump({
                'thresholds': thresholds,
                'weights': self.metric_weights,
                'metadata': {
                    'good_samples': {k: len(v) for k, v in self.good_metrics.items()},
                    'bad_samples': {k: len(v) for k, v in self.bad_metrics.items()}
                }
            }, f, indent=4)

def main():
    # Example usage
    folder_path = "/Users/amoghpanhale/Documents/GitHub/QualityManagerTesting/iPhone 12 Pro - Batch 2/batch_1"
    brightness_range = (-2.0, 10.0)  # Adjust these values as needed
    
    analyzer = ThresholdAnalyzer(folder_path, brightness_range)
    thresholds = analyzer.calculate_thresholds(percentile=5)
    
    # Save thresholds
    analyzer.save_thresholds(thresholds)
    
    # Print analysis
    analyzer.print_threshold_analysis(thresholds)

if __name__ == "__main__":
    main()