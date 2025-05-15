#!/usr/bin/env python3
# Script to create proper test data for Mean Teacher

import numpy as np
import os
from pathlib import Path
import random
import shutil
import sys

# Define the path for test data
data_dir = Path('/scratch/lustre/home/mdah0000/images/preprocessed_v2')

# Clean the directory if it exists
if data_dir.exists():
    print(f"Cleaning directory: {data_dir}")
    shutil.rmtree(data_dir)

# Create the directory
data_dir.mkdir(parents=True, exist_ok=True)

# Generate sample data
num_patients = 20  # Total number of patients
img_size = 256  # Image size (HxW)

print(f"Creating {num_patients} patient directories with sample data in {data_dir}")

# Create patient directories
for i in range(1, num_patients + 1):
    patient_dir = data_dir / f'pancreas_{i:03d}'
    patient_dir.mkdir(exist_ok=True)
    
    # Create sample image (with more defined structures)
    img = np.zeros((img_size, img_size, 1), dtype=np.float32)
    
    # Add some basic structures - a central "pancreas-like" blob
    center_x, center_y = img_size // 2, img_size // 2
    radius = random.randint(20, 50)
    
    # Create a circular region
    y, x = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist_from_center <= radius
    
    # Add the circle to the image
    img[mask] = random.uniform(0.7, 1.0)
    
    # Add noise
    img += np.random.normal(0, 0.1, (img_size, img_size, 1)).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    # Save the image
    np.save(patient_dir / 'image.npy', img)
    
    # Create corresponding mask
    mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
    mask[dist_from_center <= radius * 0.8] = 1.0  # Slightly smaller than the image blob
    
    # Save the mask
    np.save(patient_dir / 'mask.npy', mask)
    
    if i % 5 == 0:
        print(f"Created {i}/{num_patients} patient directories")

print(f"Successfully created {num_patients} patient directories with sample data")
print(f"Sample data location: {data_dir}")

# Verify some data was created properly
sample_img_path = data_dir / 'pancreas_001' / 'image.npy'
sample_mask_path = data_dir / 'pancreas_001' / 'mask.npy'

if sample_img_path.exists() and sample_mask_path.exists():
    img = np.load(sample_img_path)
    mask = np.load(sample_mask_path)
    print(f"Verified sample image shape: {img.shape}, min: {img.min():.2f}, max: {img.max():.2f}")
    print(f"Verified sample mask shape: {mask.shape}, min: {mask.min():.2f}, max: {mask.max():.2f}")
    print("Data looks good!")
else:
    print("ERROR: Failed to verify sample data!")
    sys.exit(1)

sys.exit(0)
