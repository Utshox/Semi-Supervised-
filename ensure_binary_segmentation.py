"""
Script to modify the necessary files to ensure binary segmentation is properly handled
with shape [H, W, 1] for labels instead of [H, W, 2].
"""

import sys

# Paths to modify
CONFIG_PATH = '/scratch/lustre/home/mdah0000/smm/v14/config.py'
DATALOADER_PATH = '/scratch/lustre/home/mdah0000/smm/v14/data_loader_tf2.py'

# Read the files
with open(CONFIG_PATH, 'r') as f:
    config_content = f.read()

with open(DATALOADER_PATH, 'r') as f:
    dataloader_content = f.read()

# Update config.py to ensure num_classes is 1
if 'num_classes: int = 2' in config_content:
    config_content = config_content.replace('num_classes: int = 2', 'num_classes: int = 1  # Changed from 2 to 1 for binary segmentation')
    print(f"Updated {CONFIG_PATH} to set num_classes = 1")
elif 'num_classes: int = 1' in config_content:
    print(f"No change needed for {CONFIG_PATH}, num_classes is already 1")
else:
    print(f"Warning: Could not find num_classes in {CONFIG_PATH}")

# Check if we need to force num_classes=1 in the data loader
if 'num_classes_val = self.config.num_classes' in dataloader_content:
    dataloader_content = dataloader_content.replace(
        'num_classes_val = self.config.num_classes', 
        '# Force binary segmentation\nnum_classes_val = 1  # Force to 1 regardless of config.num_classes'
    )
    print(f"Updated {DATALOADER_PATH} to force num_classes_val = 1")
else:
    print(f"Warning: Could not find expected code pattern in {DATALOADER_PATH}")

# Write the updated files
with open(CONFIG_PATH, 'w') as f:
    f.write(config_content)

with open(DATALOADER_PATH, 'w') as f:
    f.write(dataloader_content)

print("Files updated successfully. Re-run your script to use binary segmentation.")
