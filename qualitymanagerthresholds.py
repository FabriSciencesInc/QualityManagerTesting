import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from qualitymanager import QualityChecker
from PIL import Image
from PIL.ExifTags import TAGS
import cv2

# Initialize checker
checker = QualityChecker()
folder_path = "iPhone_14_Pro_Max"
folder = Path(folder_path)
masks_folder = folder / "results"

def get_brightness_from_exif(image_path):
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif is None:
                return None
                
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'BrightnessValue':
                    return float(value)
            return None
    except:
        return None

def load_mask(image_name, masks_folder):
    mask_path = masks_folder / f"{image_name.rsplit('.', 1)[0]}_mask.png"
    if not mask_path.exists():
        print(f"Warning: No mask found for {image_name}")
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# Read labels
labels = {}
with open(folder / "classifications.txt", 'r') as f:
    for line in f:
        if ',' in line:
            image_name, label = line.strip().split(',')
            labels[image_name] = label.strip().lower()

# Use a single case-insensitive glob pattern for HEIC files
print("Collecting metrics from images...")
image_metrics = []
heic_files = list(set(folder.glob('*.[Hh][Ee][Ii][Cc]')))  # Use set to ensure uniqueness
print(f"Found {len(heic_files)} HEIC files")

for file_path in heic_files:
    if file_path.name not in labels:
        print(f"Warning: No label found for {file_path.name}")
        continue
        
    print(f"Processing {file_path.name}...")
    
    # Load mask first - if no mask, skip this image
    mask = load_mask(file_path.name, masks_folder)
    if mask is None:
        continue
    
    # Load image and check quality
    image = checker.read_heic_image(str(file_path))
    results = checker.check_image_quality(image, mask)
    
    # Add brightness from EXIF
    brightness = get_brightness_from_exif(file_path)
    
    # Store all metrics and label for this image
    image_metrics.append({
        'graininess': float(results['metrics']['graininess']),
        'sharpness': float(results['metrics']['sharpness']),
        'contrast': float(results['metrics']['contrast']),
        'brightness': float(brightness if brightness is not None else 0),
        'label': labels[file_path.name]
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(image_metrics)
good_imgs = df[df['label'] == 'good']

# Calculate percentile ranges for good images
grain_lower_min = good_imgs['graininess'].quantile(0.1)
sharp_lower_min = good_imgs['sharpness'].quantile(0.1)
contrast_lower_min = good_imgs['contrast'].quantile(0.1)
brightness_lower_min = good_imgs['brightness'].quantile(0.1)

grain_upper_max = good_imgs['graininess'].quantile(0.9)
sharp_upper_max = good_imgs['sharpness'].quantile(0.9)
contrast_upper_max = good_imgs['contrast'].quantile(0.9)
brightness_upper_max = good_imgs['brightness'].quantile(0.9)

# Create search ranges
grain_lower = np.linspace(grain_lower_min, grain_upper_max/2, 10)
grain_upper = np.linspace(grain_lower_min*2, grain_upper_max, 10)
sharp_lower = np.linspace(sharp_lower_min, sharp_upper_max/2, 10)
sharp_upper = np.linspace(sharp_lower_min*2, sharp_upper_max, 10)
contrast_lower = np.linspace(contrast_lower_min, contrast_upper_max/2, 10)
contrast_upper = np.linspace(contrast_lower_min*2, contrast_upper_max, 10)
brightness_lower = np.linspace(brightness_lower_min, brightness_upper_max/2, 10)
brightness_upper = np.linspace(brightness_lower_min*2, brightness_upper_max, 10)

# Grid search for optimal thresholds
print("Starting grid search for optimal thresholds...")
best_score = 0
best_thresholds = None
total_iterations = (len(grain_lower) * len(grain_upper) * 
                   len(sharp_lower) * len(sharp_upper) * 
                   len(contrast_lower) * len(contrast_upper) *
                   len(brightness_lower) * len(brightness_upper))
current_iteration = 0

for g_lower in grain_lower:
    for g_upper in grain_upper:
        if g_lower >= g_upper:
            continue
            
        for s_lower in sharp_lower:
            for s_upper in sharp_upper:
                if s_lower >= s_upper:
                    continue
                    
                for c_lower in contrast_lower:
                    for c_upper in contrast_upper:
                        if c_lower >= c_upper:
                            continue
                            
                        for b_lower in brightness_lower:
                            for b_upper in brightness_upper:
                                current_iteration += 1
                                if current_iteration % 1000 == 0:
                                    print(f"Progress: {current_iteration}/{total_iterations} combinations tested")
                                    
                                if b_lower >= b_upper:
                                    continue
                                    
                                # Test current thresholds
                                predictions = []
                                true_labels = []
                                
                                for metric in image_metrics:
                                    passes = (
                                        g_lower <= metric['graininess'] <= g_upper and
                                        s_lower <= metric['sharpness'] <= s_upper and
                                        c_lower <= metric['contrast'] <= c_upper and
                                        b_lower <= metric['brightness'] <= b_upper
                                    )
                                    
                                    predictions.append('good' if passes else 'bad')
                                    true_labels.append(metric['label'])
                                
                                score = f1_score(true_labels, predictions, pos_label='good')
                                
                                if score > best_score:
                                    best_score = score
                                    best_thresholds = {
                                        'grain': {'lower': g_lower, 'upper': g_upper},
                                        'sharpness': {'lower': s_lower, 'upper': s_upper},
                                        'contrast': {'lower': c_lower, 'upper': c_upper},
                                        'brightness': {'lower': b_lower, 'upper': b_upper}
                                    }

# Print and save results
print(f"\nOptimization complete!")
print(f"Best F1 score: {best_score:.3f}")
print("\nOptimal thresholds:")
print(f"Graininess range: {best_thresholds['grain']['lower']:.2f} - {best_thresholds['grain']['upper']:.2f}")
print(f"Sharpness range: {best_thresholds['sharpness']['lower']:.2f} - {best_thresholds['sharpness']['upper']:.2f}")
print(f"Contrast range: {best_thresholds['contrast']['lower']:.2f} - {best_thresholds['contrast']['upper']:.2f}")
print(f"Brightness range: {best_thresholds['brightness']['lower']:.2f} - {best_thresholds['brightness']['upper']:.2f}")

# Save thresholds to file
with open('optimal_thresholds.txt', 'w') as f:
    f.write(f"Graininess lower: {best_thresholds['grain']['lower']:.2f}\n")
    f.write(f"Graininess upper: {best_thresholds['grain']['upper']:.2f}\n")
    f.write(f"Sharpness lower: {best_thresholds['sharpness']['lower']:.2f}\n")
    f.write(f"Sharpness upper: {best_thresholds['sharpness']['upper']:.2f}\n")
    f.write(f"Contrast lower: {best_thresholds['contrast']['lower']:.2f}\n")
    f.write(f"Contrast upper: {best_thresholds['contrast']['upper']:.2f}\n")
    f.write(f"Brightness lower: {best_thresholds['brightness']['lower']:.2f}\n")
    f.write(f"Brightness upper: {best_thresholds['brightness']['upper']:.2f}\n")