import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from pathlib import Path
import keyboard
import json
import exiftool
from pillow_heif import register_heif_opener

# Register HEIF/HEIC support
register_heif_opener()

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        
        # Initialize variables
        self.current_folder = None
        self.image_files = []
        self.current_index = 0
        self.classifications = {}
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard events
        self.root.bind('<Left>', lambda e: self.classify_image('bad'))
        self.root.bind('<Right>', lambda e: self.classify_image('good'))
        
    def extract_full_metadata(self, image_path):
        """
        Extract comprehensive metadata from an image file including EXIF, XMP, 
        ICC profiles, and other metadata types.
        """
        try:
            with exiftool.ExifToolHelper(executable='/opt/homebrew/bin/exiftool') as et:
                # Extract metadata
                metadata = et.get_metadata(image_path)[0]
                
                # Organize metadata
                organized_metadata = {
                    'Basic': {},
                    'Color & Brightness': {},
                    'Camera': {},
                    'Location': {},
                    'Technical': {},
                    'Other': {}
                }
                
                # Sort metadata into categories
                for key, value in metadata.items():
                    if isinstance(value, (bytes, dict, list)):
                        continue
                        
                    key_lower = key.lower()
                    
                    if any(term in key_lower for term in ['brightness', 'color', 'icc', 'contrast', 'saturation', 'light']):
                        organized_metadata['Color & Brightness'][key] = value
                    elif any(term in key_lower for term in ['iso', 'exposure', 'aperture', 'focal', 'flash', 'lens']):
                        organized_metadata['Camera'][key] = value
                    elif any(term in key_lower for term in ['gps', 'location', 'coordinate']):
                        organized_metadata['Location'][key] = value
                    elif any(term in key_lower for term in ['create', 'date', 'time', 'model', 'make', 'software']):
                        organized_metadata['Basic'][key] = value
                    elif any(term in key_lower for term in ['resolution', 'depth', 'size', 'dimension', 'format']):
                        organized_metadata['Technical'][key] = value
                    else:
                        organized_metadata['Other'][key] = value
                
                return organized_metadata
                
        except Exception as e:
            return {"error": f"Failed to extract metadata: {str(e)}"}
    
    def setup_ui(self):
        # Create main frame
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(expand=True, fill='both')
        
        # Load folder button
        self.load_btn = tk.Button(self.main_frame, text="Load Folder", command=self.load_folder)
        self.load_btn.pack(pady=5)
        
        # Image display
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self.main_frame, text="No folder loaded")
        self.status_label.pack(pady=5)
        
        # Instructions
        instructions = "Left Arrow = Bad    Right Arrow = Good"
        self.instruction_label = tk.Label(self.main_frame, text=instructions)
        self.instruction_label.pack(pady=5)
        
    def load_folder(self):
        self.current_folder = filedialog.askdirectory()
        if self.current_folder:
            # Get all image files including HEIC
            self.image_files = [f for f in os.listdir(self.current_folder) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic'))]
            
            if self.image_files:
                self.current_index = 0
                self.load_existing_classifications()
                self.show_current_image()
                
                # Create metadata folder if it doesn't exist
                metadata_folder = os.path.join(self.current_folder, 'metadata')
                os.makedirs(metadata_folder, exist_ok=True)
                
                # Extract metadata for all images
                for image_file in self.image_files:
                    image_path = os.path.join(self.current_folder, image_file)
                    metadata = self.extract_full_metadata(image_path)
                    
                    # Save metadata to JSON file
                    metadata_path = os.path.join(metadata_folder, f"{os.path.splitext(image_file)[0]}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
            else:
                self.status_label.config(text="No images found in folder")
                
    def load_existing_classifications(self):
        classification_file = os.path.join(self.current_folder, 'classifications.txt')
        if os.path.exists(classification_file):
            with open(classification_file, 'r') as f:
                for line in f:
                    if line.strip():
                        filename, classification = line.strip().split(',')
                        self.classifications[filename] = classification
                        
    def show_current_image(self):
        if 0 <= self.current_index < len(self.image_files):
            # Load and display image
            image_path = os.path.join(self.current_folder, self.image_files[self.current_index])
            
            # Handle HEIC images
            if image_path.lower().endswith('.heic'):
                image = Image.open(image_path)
            else:
                image = Image.open(image_path)
            
            # Resize image to fit window (max 800x600)
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update status
            status = f"Image {self.current_index + 1} of {len(self.image_files)}"
            if self.image_files[self.current_index] in self.classifications:
                status += f" (Classified as: {self.classifications[self.image_files[self.current_index]]})"
            self.status_label.config(text=status)
            
    def classify_image(self, classification):
        if self.current_folder and self.image_files:
            current_image = self.image_files[self.current_index]
            self.classifications[current_image] = classification
            
            # Write to file immediately
            with open(os.path.join(self.current_folder, 'classifications.txt'), 'w') as f:
                for filename, cls in self.classifications.items():
                    f.write(f"{filename},{cls}\n")
            
            # Move to next image
            self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
            self.show_current_image()

def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()