import numpy as np
import argparse
from pathlib import Path
import sys

def inspect_mask(mask_path: Path):
    """
    Loads a .npy mask file and prints statistics about its label values.
    """
    if not mask_path.exists():
        print(f"ERROR: Mask file not found at {mask_path}", file=sys.stderr)
        return

    try:
        mask_data = np.load(mask_path)
        print(f"Successfully loaded mask: {mask_path.name}")
        print(f"Shape of the 3D mask: {mask_data.shape}")

        unique_all, counts_all = np.unique(mask_data, return_counts=True)
        print("\n--- Overall 3D Mask Statistics ---")
        for val, count in zip(unique_all, counts_all):
            print(f"Value {int(val)}: appears {count} times")

        num_slices = mask_data.shape[2]
        slice_indices_to_inspect = sorted(list(set([0, num_slices // 2, num_slices - 1]))) # First, middle, last

        if not slice_indices_to_inspect:
            print("\nNo slices to inspect (perhaps a 2D mask or empty?).")
            return

        print("\n--- Sample 2D Slice Statistics ---")
        for i in slice_indices_to_inspect:
            if i >= num_slices: # Should not happen with sorted set but good check
                continue
            
            slice_2d = mask_data[..., i]
            unique_slice, counts_slice = np.unique(slice_2d, return_counts=True)
            print(f"\nSlice {i}:")
            print(f"  Shape of this 2D slice: {slice_2d.shape}")
            if unique_slice.size == 0:
                print("  This slice is empty or has an unexpected format.")
                continue
            for val, count in zip(unique_slice, counts_slice):
                print(f"  Value {int(val)}: appears {count} times")
        
        print("\n------------------------------------\n")

    except Exception as e:
        print(f"ERROR: Could not load or process mask {mask_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect label values in a .npy mask file.")
    parser.add_argument("mask_file_path", type=str, help="Path to the .npy mask file.")
    
    args = parser.parse_args()
    
    mask_file = Path(args.mask_file_path)
    inspect_mask(mask_file)

    print("Instructions:")
    print("1. Run this script with the path to one of your 'mask_cropped.npy' files.")
    print(f"   Example: python inspect_mask_labels.py /path/to/your/data/pancreas_XXX/mask_cropped.npy")
    print("2. Repeat for a few different mask files (e.g., one where you saw 'Raw label_data sum' being high, and one where it might be lower).")
    print("3. Copy the output from your terminal and paste it back to me.")
    print("This will help understand what values 0, 1, and 2 represent.")
