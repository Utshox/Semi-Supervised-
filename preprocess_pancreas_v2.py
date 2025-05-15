import os
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import json
from tqdm import tqdm
import tensorflow as tf # For tf.image.resize_with_pad

# --- Configuration ---
RAW_DATA_BASE_DIR = Path('/scratch/lustre/home/mdah0000/images/raw')
PREPROCESSED_OUTPUT_DIR = Path('/scratch/lustre/home/mdah0000/images/preprocessed_v2')
DATASET_JSON_FILE = RAW_DATA_BASE_DIR / 'dataset.json'

TARGET_VOXEL_SPACING = [1.0, 1.0, 2.5]  # Target spacing in mm (x, y, z)
TARGET_PIXEL_DIM_XY = (256, 256)      # Target slice dimensions (height, width)

# --- Helper Functions ---

def load_dataset_json(json_path):
    """Load and parse the dataset.json file"""
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

def normalize_intensity_percentile(img_data):
    """Normalize image intensity to range [0,1] using 1st and 99th percentiles."""
    p1, p99 = np.percentile(img_data, (1, 99))
    img_data_clipped = np.clip(img_data, p1, p99)
    img_data_normalized = (img_data_clipped - p1) / (p99 - p1 + 1e-8) # Add epsilon for stability
    return img_data_normalized.astype(np.float32)

def resample_sitk_image(sitk_image, target_spacing, interpolator=sitk.sitkBSpline, new_size=None):
    """Resample a SimpleITK image to a target voxel spacing."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    if new_size is None:
        new_size = [
            int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
        ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(map(float, target_spacing)))
    resample.SetSize(list(map(int, new_size)))  # Ensure integer size
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # Set default pixel value to 0
    resample.SetInterpolator(interpolator)

    return resample.Execute(sitk_image)

def resize_volume_slices_tf(volume_np, target_hw_dim, method='bilinear'):
    """
    Resizes each slice of a 3D volume to target_hw_dim (height, width) using TensorFlow.
    Assumes volume_np is (D, H, W) or (H, W, D). TensorFlow expects (num_slices, H, W, C) or (H,W, num_slices*C)
    This function assumes input (D, H, W) and outputs (D, target_H, target_W)
    """
    num_slices = volume_np.shape[0]
    original_h, original_w = volume_np.shape[1], volume_np.shape[2]
    target_h, target_w = target_hw_dim

    # Reshape to (num_slices, H, W, 1) for tf.image.resize
    volume_reshaped = volume_np.reshape((num_slices, original_h, original_w, 1))

    # TF resize expects float32
    if volume_reshaped.dtype != tf.float32:
        volume_reshaped = tf.cast(volume_reshaped, tf.float32)

    # Use resize_with_pad to maintain aspect ratio and pad if necessary
    # For labels (nearest), padding with 0 is fine. For images (bilinear), padding with 0 is also common.
    resized_slices_tf = tf.image.resize_with_pad(
        volume_reshaped,
        target_h,
        target_w,
        method=method
    )
    # Squeeze the channel dimension and convert back to numpy
    return tf.squeeze(resized_slices_tf, axis=-1).numpy()


def process_single_volume(nifti_img_path, nifti_label_path=None, 
                          target_spacing_physical_xyz=TARGET_VOXEL_SPACING, 
                          target_pixel_dim_hw=TARGET_PIXEL_DIM_XY):
    """
    Processes a single NIfTI volume:
    1. Loads NIfTI image (and label if provided) and orients to RAS canonical.
    2. Transposes NumPy array to (Z_slices, Y, X) for internal processing.
    3. Converts to SimpleITK format, setting correct spacing.
    4. Resamples to target_spacing_physical_xyz.
    5. Converts resampled SITK image back to NumPy array (Z_slices, Y, X).
    6. Normalizes intensity for the image.
    7. Resizes/pads each slice of the (image and label) NumPy arrays to target_pixel_dim_hw.
    8. Ensures label is binary.
    """
    print(f"Processing image: {nifti_img_path}")
    if nifti_label_path:
        print(f"Processing label: {nifti_label_path}")

    # --- Load Image and Reorient to RAS (X, Y, Z_slices) ---
    img_nib_original = nib.load(nifti_img_path)
    img_nib_canonical = nib.as_closest_canonical(img_nib_original)
    img_data_xyz = img_nib_canonical.get_fdata().astype(np.float32) # Shape (X, Y, Z_slices)
    
    # Original spacing from canonical header (sp_X, sp_Y, sp_Z_slices)
    original_spacing_xyz = list(map(float, img_nib_canonical.header.get_zooms()))

    # --- Transpose to (Z_slices, Y, X) for internal processing ---
    img_data_zyx = img_data_xyz.transpose(2, 1, 0) # Now (Z_slices, Y, X)

    # --- Convert to SimpleITK ---
    # GetImageFromArray expects (depth, height, width) which is our (Z_slices, Y, X)
    img_sitk = sitk.GetImageFromArray(img_data_zyx) 
    # SetSpacing expects (spacing_X, spacing_Y, spacing_Z_slices)
    img_sitk.SetSpacing(original_spacing_xyz) 
    # Set other geometric info from canonical nibabel object if needed (origin, direction)
    # For simplicity, we often rely on ITK defaults if not critical or use nibabel affine.
    # img_sitk.SetDirection(img_nib_canonical.affine[:3,:3].flatten()) # Example, needs testing
    # img_sitk.SetOrigin(img_nib_canonical.affine[:3,3])

    # --- Resample Image to Target Voxel Spacing ---
    # target_spacing_physical_xyz is (target_sp_X, target_sp_Y, target_sp_Z_slices)
    img_resampled_sitk = resample_sitk_image(img_sitk, target_spacing_physical_xyz, interpolator=sitk.sitkBSpline)
    # GetArrayFromImage returns (sitk_Z, sitk_Y, sitk_X) -> (Z_slices_res, Y_res, X_res)
    img_resampled_np_zyx = sitk.GetArrayFromImage(img_resampled_sitk) 

    # --- Normalize Image Intensity ---
    img_normalized_np_zyx = normalize_intensity_percentile(img_resampled_np_zyx)

    # --- Resize/Pad Image Slices to Target Pixel Dimensions ---
    # resize_volume_slices_tf expects (D, H, W) and target_pixel_dim_hw (target_H, target_W)
    img_final_np_zyx = resize_volume_slices_tf(img_normalized_np_zyx, target_pixel_dim_hw, method='bilinear')
    # img_final_np_zyx is now (Z_slices_res, target_H, target_W)

    processed_label_final_np_zyx = None
    if nifti_label_path and os.path.exists(nifti_label_path):
        label_nib_original = nib.load(nifti_label_path)
        label_nib_canonical = nib.as_closest_canonical(label_nib_original)
        label_data_xyz = label_nib_canonical.get_fdata().astype(np.uint8) # Labels should be integer

        # Transpose to (Z_slices, Y, X)
        label_data_zyx = label_data_xyz.transpose(2, 1, 0)

        label_sitk = sitk.GetImageFromArray(label_data_zyx)
        # Use same original spacing as image (derived from canonical image NIfTI)
        label_sitk.SetSpacing(original_spacing_xyz) 
        # label_sitk.SetDirection(img_nib_canonical.affine[:3,:3].flatten())
        # label_sitk.SetOrigin(img_nib_canonical.affine[:3,3])


        # Resample label to target voxel spacing, ensuring alignment with image
        label_resampled_sitk = resample_sitk_image(label_sitk, target_spacing_physical_xyz, 
                                                   interpolator=sitk.sitkNearestNeighbor,
                                                   new_size=img_resampled_sitk.GetSize()) # Match resampled image size
        label_resampled_np_zyx = sitk.GetArrayFromImage(label_resampled_sitk) # (Z_slices_res, Y_res, W_res)

        # --- Resize/Pad Label Slices to Target Pixel Dimensions ---
        label_resized_np_zyx = resize_volume_slices_tf(label_resampled_np_zyx, target_pixel_dim_hw, method='nearest')
        
        # Ensure label is binary (0 or 1)
        processed_label_final_np_zyx = (label_resized_np_zyx > 0.5).astype(np.uint8)
        # processed_label_final_np_zyx is now (Z_slices_res, target_H, target_W)
        
        # Sanity check shapes
        if img_final_np_zyx.shape != processed_label_final_np_zyx.shape:
            print(f"WARNING: Final image shape {img_final_np_zyx.shape} and label shape {processed_label_final_np_zyx.shape} mismatch for {nifti_img_path}")
            min_depth = min(img_final_np_zyx.shape[0], processed_label_final_np_zyx.shape[0])
            img_final_np_zyx = img_final_np_zyx[:min_depth, :, :]
            processed_label_final_np_zyx = processed_label_final_np_zyx[:min_depth, :, :]
            print(f"Adjusted to common depth: {img_final_np_zyx.shape}")

    return img_final_np_zyx, processed_label_final_np_zyx


def run_preprocessing():
    """Main function to run the preprocessing pipeline."""
    print("Starting preprocessing v2...")
    PREPROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_JSON_FILE.exists():
        print(f"ERROR: dataset.json not found at {DATASET_JSON_FILE}")
        return

    dataset_info = load_dataset_json(DATASET_JSON_FILE)

    # --- Process Training Data ---
    if 'training' in dataset_info:
        print(f"\nProcessing {len(dataset_info['training'])} training cases...")
        for case_info in tqdm(dataset_info['training'], desc="Training Cases"):
            img_rel_path = case_info['image'].replace('./', '') # imagesTr/pancreas_XXX.nii.gz
            label_rel_path = case_info['label'].replace('./', '') # labelsTr/pancreas_XXX.nii.gz

            nifti_img_path = RAW_DATA_BASE_DIR / img_rel_path
            nifti_label_path = RAW_DATA_BASE_DIR / label_rel_path
            
            case_id = Path(img_rel_path).name.replace('.nii.gz', '') # e.g., pancreas_XXX

            # ---- MODIFICATION: Check for raw file existence before creating output directory ----
            if not nifti_img_path.exists():
                print(f"Raw image file not found: {nifti_img_path}, skipping case {case_id} from dataset.json")
                continue
            if not nifti_label_path.exists(): # For training data, label is essential
                print(f"Raw label file not found: {nifti_label_path}, skipping case {case_id} from dataset.json")
                continue
            # ---- END MODIFICATION ----

            case_output_dir = PREPROCESSED_OUTPUT_DIR / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)

            output_img_path = case_output_dir / 'image.npy'
            output_label_path = case_output_dir / 'mask.npy'

            if output_img_path.exists() and output_label_path.exists():
                print(f"Skipping {case_id}, already processed.")
                continue
            
            try:
                img_processed, label_processed = process_single_volume(
                    nifti_img_path, nifti_label_path
                )
                if img_processed is not None and label_processed is not None:
                    np.save(output_img_path, img_processed)
                    np.save(output_label_path, label_processed)
                    print(f"Saved {case_id} to {case_output_dir}")
                else:
                    print(f"Processing failed for {case_id}, img_processed or label_processed is None.")
            except Exception as e:
                print(f"ERROR processing training case {case_id}: {e}")
                import traceback
                traceback.print_exc()

    # --- Process Test Data (if listed in dataset.json) ---
    # Assuming 'test' or 'testing' key might exist
    test_key = None
    if 'test' in dataset_info:
        test_key = 'test'
    elif 'testing' in dataset_info: # Common alternative key
        test_key = 'testing'
        
    if test_key and dataset_info[test_key]:
        print(f"\nProcessing {len(dataset_info[test_key])} test cases...")
        for test_img_rel_path_info in tqdm(dataset_info[test_key], desc="Test Cases"):
            # Test data might be a list of strings or dicts
            if isinstance(test_img_rel_path_info, dict) and 'image' in test_img_rel_path_info:
                 img_rel_path = test_img_rel_path_info['image'].replace('./', '')
            elif isinstance(test_img_rel_path_info, str):
                 img_rel_path = test_img_rel_path_info.replace('./', '')
            else:
                print(f"Skipping unrecognized test case format: {test_img_rel_path_info}")
                continue

            nifti_img_path = RAW_DATA_BASE_DIR / img_rel_path
            case_id = Path(img_rel_path).name.replace('.nii.gz', '')

            # ---- MODIFICATION: Check for raw file existence before creating output directory ----
            if not nifti_img_path.exists():
                print(f"Raw image file not found for test case: {nifti_img_path}, skipping case {case_id} from dataset.json")
                continue
            # ---- END MODIFICATION ----

            case_output_dir = PREPROCESSED_OUTPUT_DIR / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            output_img_path = case_output_dir / 'image.npy'

            if output_img_path.exists():
                print(f"Skipping {case_id}, already processed.")
                continue

            try:
                img_processed, _ = process_single_volume(nifti_img_path, nifti_label_path=None)
                if img_processed is not None:
                    np.save(output_img_path, img_processed)
                    print(f"Saved {case_id} to {case_output_dir}")
                else:
                    print(f"Processing failed for test case {case_id}, img_processed is None.")
            except Exception as e:
                print(f"ERROR processing test case {case_id}: {e}")
                import traceback
                traceback.print_exc()

    print("\nPreprocessing v2 complete!")

if __name__ == '__main__':
    # Ensure GPU memory for TensorFlow is not grabbed if TF is only used for tf.image
    tf.config.set_visible_devices([], 'GPU') # Make GPUs invisible to this TF session
    
    # You might want to check available memory and set a limit if needed
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]) # 1GB
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         print(e)
            
    run_preprocessing()
