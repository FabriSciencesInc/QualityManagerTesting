import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import pillow_heif
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any
import json
import exiftool
from qualitymanager import QualityChecker

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HEIC Image Quality Classifier")
        
        # Initialize ExifTool
        self.et = exiftool.ExifToolHelper(executable='/opt/homebrew/bin/exiftool')
        
        # Rest of initialization...
        self.quality_checker = QualityChecker()
        self.current_folder = None
        self.image_files = []
        self.current_index = 0
        self.classifications = {}
        
        self.setup_ui()
        self.bind_keys()
    
    def get_image_brightness(self, heic_file):
        try:
            # Get metadata using ExifTool
            metadata = self.et.get_metadata(str(heic_file))[0]
            
            # Check different possible brightness tags
            brightness_keys = [
                'BrightnessValue',
                'LightValue',
                'MeasuredEV',
                'MeasuredLV',
                'Brightness'
            ]
            
            # Look for brightness in various metadata keys
            for key in metadata:
                for brightness_key in brightness_keys:
                    if brightness_key.lower() in key.lower():
                        value = metadata[key]
                        if isinstance(value, (int, float)):
                            return f"{float(value):.2f}"
            
            # If no specific brightness value found, try calculating from exposure settings
            if 'ExposureTime' in metadata and 'FNumber' in metadata:
                exposure_time = float(metadata['ExposureTime'])
                f_number = float(metadata['FNumber'])
                ev = -np.log2(exposure_time * (f_number ** 2))
                return f"{ev:.2f}"
            
            return 'N/A'
            
        except Exception as e:
            print(f"Error reading brightness: {e}")
            return 'N/A'

    def __del__(self):
        # Clean up ExifTool instance
        if hasattr(self, 'et'):
            self.et.terminate()
        
    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Top frame for folder selection and counter
        top_frame = ttk.Frame(self.main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Folder selection
        ttk.Button(top_frame, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        
        # Image counter
        self.counter_label = ttk.Label(top_frame, text="Image: 0/0")
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        # Image display
        self.image_frame = ttk.Frame(self.main_frame, borderwidth=2, relief='solid')
        self.image_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(self.main_frame, text="Image Metrics", padding="10")
        metrics_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.metrics_labels = {}
        metrics = ['Brightness', 'Contrast', 'Graininess', 'Sharpness', 'Classification']
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5)
            self.metrics_labels[metric.lower()] = ttk.Label(metrics_frame, text="")
            self.metrics_labels[metric.lower()].grid(row=i, column=1, sticky=tk.W, padx=5)
        
        # Navigation
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Instructions
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        ttk.Label(controls_frame, text="Controls:", font=('Helvetica', 10, 'bold')).pack()
        ttk.Label(controls_frame, text="← Left Arrow: Mark as BAD").pack()
        ttk.Label(controls_frame, text="→ Right Arrow: Mark as GOOD").pack()

        
    def bind_keys(self):
        self.root.bind('<Left>', lambda e: self.classify_image('bad'))
        self.root.bind('<Right>', lambda e: self.classify_image('good'))
        
    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.current_folder = Path(folder)
            self.load_images()
            self.load_classifications()
            self.show_current_image()
    
    def load_images(self):
        self.image_files = []
        if self.current_folder:
            self.image_files = sorted(
                list(self.current_folder.glob('*.heic')) + 
                list(self.current_folder.glob('*.HEIC'))
            )
            self.current_index = 0
    
    def load_classifications(self):
        classification_file = self.current_folder / 'classifications.json'
        if classification_file.exists():
            with open(classification_file, 'r') as f:
                self.classifications = json.load(f)
        
    def save_classifications(self):
        if self.current_folder:
            classification_file = self.current_folder / 'classifications.json'
            with open(classification_file, 'w') as f:
                json.dump(self.classifications, f, indent=4)

    def show_current_image(self):
        if not self.image_files:
            return
        
        # Update counter
        self.counter_label.config(text=f"Image: {self.current_index + 1}/{len(self.image_files)}")
        
        current_file = self.image_files[self.current_index]
        
        # Load and display image
        heif_file = pillow_heif.read_heif(str(current_file))
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
        )
        
        # Convert to numpy array for processing
        np_image = np.array(image)
        
        # Load mask
        mask_path = self.current_folder / 'results' / f"{current_file.stem}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (np_image.shape[1], np_image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.ones((np_image.shape[0], np_image.shape[1]), dtype=np.uint8) * 255
        
        mask = cv2.bitwise_not(mask)

        # Calculate and display metrics
        quality_results = self.quality_checker.check_image_quality(np_image, mask)
        
        # Apply mask to the image
        masked_image = np_image.copy()
        if len(masked_image.shape) == 3:  # Color image
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            masked_image = cv2.bitwise_and(masked_image, mask_3channel)
            
            # Optional: make background gray instead of black
            background = np.ones_like(masked_image) * 128
            mask_inv = cv2.bitwise_not(mask_3channel)
            background = cv2.bitwise_and(background, mask_inv)
            masked_image = cv2.add(masked_image, background)
        else:  # Grayscale image
            masked_image = cv2.bitwise_and(masked_image, mask)
        
        # Convert back to PIL Image for display
        masked_image_pil = Image.fromarray(masked_image)
        
        # Resize image for display after processing
        display_size = (800, 600)
        display_image = masked_image_pil.copy()
        display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        brightness = self.get_image_brightness(current_file)
        self.metrics_labels['brightness'].configure(text=brightness)
        self.metrics_labels['contrast'].configure(
            text=f"{quality_results['metrics']['contrast']:.2f}"
        )
        self.metrics_labels['graininess'].configure(
            text=f"{quality_results['metrics']['graininess']:.2f}"
        )
        self.metrics_labels['sharpness'].configure(
            text=f"{quality_results['metrics']['sharpness']:.2f}"
        )
        
        # Show classification if exists
        classification = self.classifications.get(current_file.name, 'Unclassified')
        self.metrics_labels['classification'].configure(
            text=classification,
            foreground='green' if classification == 'good' else 
                    'red' if classification == 'bad' else 'black'
        )
        
    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.show_current_image()
            
    def prev_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.show_current_image()
    
    def classify_image(self, classification):
        if not self.image_files:
            return
        
        current_file = self.image_files[self.current_index]
        self.classifications[current_file.name] = classification
        self.save_classifications()
        self.show_current_image()
        self.next_image()

def main():
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()