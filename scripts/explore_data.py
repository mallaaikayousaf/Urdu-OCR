
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class DataExplorer:
    
    def __init__(self, data_path="data/raw"):
       
        self.data_path = Path(data_path)
        self.class_names = []
        self.class_counts = {}
        self.image_shapes = []
        
    def get_all_classes(self):
       
        if not self.data_path.exists():
            raise FileNotFoundError(f"Path {self.data_path} not found!")
        
        # Get all subdirectories (each subdirectory is a character class)
        self.class_names = [f.name for f in self.data_path.iterdir() if f.is_dir()]
        self.class_names.sort()
        
        print(f"Found {len(self.class_names)} character classes")
        return self.class_names
    
    def count_samples_per_class(self):
        
        for class_name in self.class_names:
            class_path = self.data_path / class_name
            # Count all image files (png, jpg, jpeg)
            image_files = list(class_path.glob("*.png")) + \
                         list(class_path.glob("*.jpg")) + \
                         list(class_path.glob("*.jpeg"))
            self.class_counts[class_name] = len(image_files)
        
        return self.class_counts
    
    def analyze_image_properties(self, num_samples=5):
       
        print("\nAnalyzing image properties...")
        
        for class_name in self.class_names[:3]:  # Analyze first 3 classes only
            class_path = self.data_path / class_name
            image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            
            for img_file in image_files[:num_samples]:
                img = cv2.imread(str(img_file))
                if img is not None:
                    height, width = img.shape[:2]
                    self.image_shapes.append((height, width))
        
        if self.image_shapes:
            shapes_array = np.array(self.image_shapes)
            print(f"   Height range: {shapes_array[:,0].min()} - {shapes_array[:,0].max()} pixels")
            print(f"   Width range: {shapes_array[:,1].min()} - {shapes_array[:,1].max()} pixels")
            print(f"   Average height: {shapes_array[:,0].mean():.0f} pixels")
            print(f"   Average width: {shapes_array[:,1].mean():.0f} pixels")
    
    def visualize_sample_images(self, samples_per_class=3):
       
        num_classes = min(len(self.class_names), 5)  # Show max 5 classes
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                 figsize=(12, 3*num_classes))
        
        # If only one class, axes is 1D, handle that case
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(self.class_names[:num_classes]):
            class_path = self.data_path / class_name
            image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            
            for j in range(min(samples_per_class, len(image_files))):
                img = cv2.imread(str(image_files[j]))
                if img is not None:
                    # Convert BGR to RGB for correct color display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(img_rgb)
                    axes[i, j].axis('off')
                    
                    if j == 0:
                        axes[i, j].set_ylabel(class_name, fontsize=10, rotation=0, 
                                             labelpad=30)
        
        plt.suptitle("Sample Images from Each Character Class", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self):
        
        if not self.class_counts:
            self.count_samples_per_class()
        
        # Create DataFrame for easy plotting
        df = pd.DataFrame(list(self.class_counts.items()), 
                         columns=['Character', 'Number of Images'])
        df = df.sort_values('Number of Images', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(15, 6))
        plt.bar(range(len(df)), df['Number of Images'].values, color='skyblue', alpha=0.7)
        plt.xlabel('Character Classes', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Distribution of Images Across Urdu Character Classes', fontsize=14)
        plt.xticks(range(len(df)), df['Character'].values, rotation=90, fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"   Total images: {df['Number of Images'].sum()}")
        print(f"   Average per class: {df['Number of Images'].mean():.1f}")
        print(f"   Max images in a class: {df['Number of Images'].max()}")
        print(f"   Min images in a class: {df['Number of Images'].min()}")
        print(f"   Classes with < 100 images: {(df['Number of Images'] < 100).sum()}")
    
    def check_for_corrupted_images(self):
       
        print("\nChecking for corrupted images...")
        corrupted = []
        
        for class_name in self.class_names:
            class_path = self.data_path / class_name
            image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        corrupted.append(str(img_file))
                except Exception as e:
                    corrupted.append(str(img_file))
        
        if corrupted:
            print(f"Found {len(corrupted)} corrupted images:")
            for img in corrupted[:10]:  # Show first 10
                print(f"   - {img}")
        else:
            print("No corrupted images found!")
        
        return corrupted
    
    def run_full_analysis(self):
       
        print("="*60)
        print("URDU HANDWRITING DATASET ANALYSIS")
        print("="*60)
        
        # Step 1: Get all classes
        self.get_all_classes()
        
        # Step 2: Count samples
        self.count_samples_per_class()
        
        # Step 3: Analyze image properties
        self.analyze_image_properties()
        
        # Step 4: Check for corrupted images
        self.check_for_corrupted_images()
        
        # Step 5: Visualizations
        self.visualize_sample_images()
        self.plot_class_distribution()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

# Run the exploration
if __name__ == "__main__":
    # Create explorer object
    explorer = DataExplorer(data_path="data/raw")
    
    # Run full analysis
    explorer.run_full_analysis()