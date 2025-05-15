import os
import shutil
import argparse
from pathlib import Path

def clean_preprocessed_folders(base_dir_str):
    """
    Deletes patient subfolders in the preprocessed directory if they
    do not contain both 'image.npy' and 'mask.npy'.
    """
    base_dir = Path(base_dir_str)
    if not base_dir.is_dir():
        print(f"Error: Directory not found: {base_dir_str}")
        return

    print(f"Scanning preprocessed directory: {base_dir}")
    folders_to_delete = []
    patient_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("pancreas_")]

    if not patient_folders:
        print("No patient folders found to clean.")
        return

    for patient_folder in patient_folders:
        image_file = patient_folder / "image.npy"
        mask_file = patient_folder / "mask.npy"

        if not (image_file.exists() and mask_file.exists()):
            folders_to_delete.append(patient_folder)
            print(f"Marked for deletion: {patient_folder} (missing image.npy or mask.npy)")
        else:
            print(f"Kept: {patient_folder} (contains both image.npy and mask.npy)")


    if not folders_to_delete:
        print("No folders to delete. Preprocessed directory seems clean based on image.npy and mask.npy presence.")
        return

    print(f"\nFound {len(folders_to_delete)} folders to delete.")
    confirm = input("Proceed with deletion? (yes/no): ")

    if confirm.lower() == 'yes':
        for folder_path in folders_to_delete:
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
            except OSError as e:
                print(f"Error deleting {folder_path}: {e}")
        print("Cleanup complete.")
    else:
        print("Deletion cancelled by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean preprocessed data folders.")
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="Path to the preprocessed data directory (e.g., /scratch/lustre/home/mdah0000/images/preprocessed_v2/)."
    )
    args = parser.parse_args()
    clean_preprocessed_folders(args.preprocessed_dir)
