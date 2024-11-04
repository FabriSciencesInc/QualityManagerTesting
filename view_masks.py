import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
from pillow_heif import register_heif_opener
import traceback
import numpy as np

# Register HEIF opener with PIL
register_heif_opener()

class ImageMaskViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Mask Viewer")
        
        # Store the current image data
        self.current_index = 0
        self.image_data = []
        self.flagged_pairs = set()
        
        # Create main layout
        self.create_gui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.show_previous())
        self.root.bind('<Right>', lambda e: self.show_next())
        self.root.bind('<Up>', lambda e: self.flag_current_pair())
        self.root.bind('<Escape>', lambda e: self.exit_program())
        
    def create_gui(self):
        # Top controls
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Load Folder", command=self.load_folder).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Previous (←)", command=self.show_previous).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Next (→)", command=self.show_next).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Flag Pair (↑)", command=self.flag_current_pair).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Save Flagged", command=self.save_flagged).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Exit (Esc)", command=self.exit_program).pack(side='right', padx=5)
        
        # Image display area
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create labels for images
        self.original_label = ttk.Label(self.display_frame)
        self.original_label.grid(row=0, column=0, padx=5)
        
        self.mask_label = ttk.Label(self.display_frame)
        self.mask_label.grid(row=0, column=1, padx=5)
        
        self.combined_label = ttk.Label(self.display_frame)
        self.combined_label.grid(row=0, column=2, padx=5)
        
        # Add labels above images
        ttk.Label(self.display_frame, text="Original Image").grid(row=1, column=0)
        ttk.Label(self.display_frame, text="Mask").grid(row=1, column=1)
        ttk.Label(self.display_frame, text="Combined").grid(row=1, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("No folder loaded | Use ← → to navigate, ↑ to flag, Esc to exit")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)

    def apply_mask(self, image, mask):
        """Apply the mask to make black regions transparent"""
        # Convert images to numpy arrays for manipulation
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create output array starting with the original image
        result = img_array.copy()
        
        # Make pixels transparent where mask is black (all channels near 0)
        black_regions = np.all(mask_array[:, :, :3] > 230, axis=2)  # Allow some tolerance for "black"
        result[black_regions, 3] = 0  # Set alpha to 0 (fully transparent) for black regions
        
        return Image.fromarray(result)

    def show_current_pair(self):
        if not self.image_data:
            return
            
        try:
            current = self.image_data[self.current_index]
            
            # Load images
            original = Image.open(current['image_path'])
            mask = Image.open(current['mask_path'])
            
            # Ensure mask is in RGBA mode
            mask = mask.convert('RGBA')
            
            # Resize images to fit display (maintain aspect ratio)
            display_size = (400, 400)
            original_resized = original.copy()
            original_resized.thumbnail(display_size)
            original_resized = original_resized.convert('RGBA')  # Ensure original is RGBA
            mask = mask.resize(original_resized.size)
            
            # Create combined image using the mask
            combined = self.apply_mask(original_resized, mask)
            
            # Convert to PhotoImage for display
            self.original_photo = ImageTk.PhotoImage(original_resized)
            self.mask_photo = ImageTk.PhotoImage(mask)
            self.combined_photo = ImageTk.PhotoImage(combined)
            
            # Update labels
            self.original_label.configure(image=self.original_photo)
            self.mask_label.configure(image=self.mask_photo)
            self.combined_label.configure(image=self.combined_photo)
            
            # Update status and window title
            flag_status = " (Flagged)" if current['name'] in self.flagged_pairs else ""
            status_text = (f"Image {self.current_index + 1}/{len(self.image_data)}: "
                          f"{current['name']} - {current['classification']}{flag_status} | "
                          "Use ← → to navigate, ↑ to flag, Esc to exit")
            self.status_var.set(status_text)
            self.root.title(f"Image Mask Viewer - {current['name']}{flag_status}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image:\n{str(e)}")
            traceback.print_exc()
        
    def exit_program(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()
        
    def load_folder(self):
        try:
            folder_path = filedialog.askdirectory()
            if not folder_path:
                return
                
            self.image_data = []
            folder_path = Path(folder_path)
            
            # Load classifications
            classifications = {}
            class_path = folder_path / 'classifications.txt'
            if class_path.exists():
                with open(class_path, 'r') as f:
                    for line in f:
                        name, classification = line.strip().split(',')
                        classifications[name] = classification
            
            # Find all HEIC images and their corresponding masks
            for heic_file in folder_path.glob('*.HEIC'):
                base_name = heic_file.stem
                mask_path = folder_path / 'results' / f"{base_name}_mask.png"
                
                if mask_path.exists():
                    classification = classifications.get(base_name, "unclassified")
                    self.image_data.append({
                        'image_path': str(heic_file),
                        'mask_path': str(mask_path),
                        'name': base_name,
                        'classification': classification
                    })
            
            if self.image_data:
                self.current_index = 0
                self.show_current_pair()
                self.status_var.set(f"Loaded {len(self.image_data)} image pairs | Use ← → to navigate, ↑ to flag, Esc to exit")
            else:
                self.status_var.set("No valid image pairs found in folder")
                messagebox.showwarning("No Images", "No valid image pairs found in the selected folder")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading folder:\n{str(e)}")
            traceback.print_exc()
    
    def show_previous(self):
        if self.image_data:
            self.current_index = (self.current_index - 1) % len(self.image_data)
            self.show_current_pair()
    
    def show_next(self):
        if self.image_data:
            self.current_index = (self.current_index + 1) % len(self.image_data)
            self.show_current_pair()
    
    def flag_current_pair(self):
        if not self.image_data:
            return
            
        current_name = self.image_data[self.current_index]['name']
        if current_name in self.flagged_pairs:
            self.flagged_pairs.remove(current_name)
        else:
            self.flagged_pairs.add(current_name)
        self.show_current_pair()  # Refresh display to show updated flag status
    
    def save_flagged(self):
        if not self.flagged_pairs:
            self.status_var.set("No flagged pairs to save")
            messagebox.showinfo("Info", "No flagged pairs to save")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="flagged_pairs.txt"
        )
        
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    for name in sorted(self.flagged_pairs):
                        f.write(f"{name}\n")
                self.status_var.set(f"Saved {len(self.flagged_pairs)} flagged pairs to {save_path}")
                messagebox.showinfo("Success", f"Saved {len(self.flagged_pairs)} flagged pairs to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving flagged pairs:\n{str(e)}")

def main():
    root = tk.Tk()
    app = ImageMaskViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()