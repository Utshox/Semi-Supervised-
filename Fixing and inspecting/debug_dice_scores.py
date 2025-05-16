#!/usr/bin/env python3
"""
Dice Score Debugging Script
--------------------------
This script diagnoses issues with the segmentation model's low Dice scores,
examining data preprocessing, model outputs, and evaluation metrics.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Add the directory containing modules to Python's path
sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

# Import necessary modules
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import SupervisedTrainer
from data_loader_tf2 import DataPipeline
from main import prepare_data_paths

# Define a class to handle the debugging process
class DiceScoreDebugger:
    def __init__(self, args):
        self.args = args
        self.config = StableSSLConfig()
        self.config.img_size_x = 512
        self.config.img_size_y = 512
        self.config.num_channels = 1
        self.config.batch_size = args.batch_size
        
        # Set up mixed precision for consistent behavior
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Prepare data paths
        print("Preparing data paths...")
        self.data_paths = prepare_data_paths(args.data_dir, num_labeled=225, num_validation=56)
        
        # Create directories
        self.debug_dir = Path("dice_debug_results")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Set up data pipeline
        self.data_pipeline = DataPipeline(self.config)
        
        # Load model if specified
        self.model_path = args.model_path
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model weights from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("No model loaded. Creating a new supervised model.")
            self.trainer = SupervisedTrainer(self.config)
            self.model = self.trainer.model
    
    def inspect_input_data(self):
        """Inspect and visualize input data for quality issues"""
        print("\nStep 1: Inspecting Input Data...")
        
        # Create dataset
        train_ds = self.data_pipeline.create_dataset(
            self.data_paths['labeled']['images'],
            self.data_paths['labeled']['labels'],
            batch_size=4,
            augment=False
        )
        
        # Examine a few samples
        sample_count = 0
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        for images, masks in train_ds.take(3):
            # Print shape and value ranges
            print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
            print(f"Image value range: [{tf.reduce_min(images)}, {tf.reduce_max(images)}]")
            print(f"Mask value range: [{tf.reduce_min(masks)}, {tf.reduce_max(masks)}]")
            print(f"Unique mask values: {np.unique(masks.numpy())}")
            
            # Check if masks are binary
            binary_check = np.logical_or(np.isclose(masks.numpy(), 0), np.isclose(masks.numpy(), 1))
            if not np.all(binary_check):
                print("WARNING: Masks are not binary! Found values between 0 and 1.")
            
            # Visualize first image and mask in batch
            for i in range(min(3, images.shape[0])):
                # Plot image
                axes[sample_count, 0].imshow(images[i, ..., 0], cmap='gray')
                axes[sample_count, 0].set_title(f"Image {sample_count+1}")
                axes[sample_count, 0].axis('off')
                
                # Plot ground truth mask
                axes[sample_count, 1].imshow(masks[i, ..., 0], cmap='viridis')
                axes[sample_count, 1].set_title(f"Ground Truth Mask {sample_count+1}")
                axes[sample_count, 1].axis('off')
                
                # Calculate mask coverage percentage
                mask_coverage = np.sum(masks[i, ..., 0]) / np.prod(masks[i, ..., 0].shape) * 100
                axes[sample_count, 2].text(0.5, 0.5, f"Mask Coverage: {mask_coverage:.2f}%", 
                                         ha='center', va='center', fontsize=12,
                                         bbox=dict(facecolor='white', alpha=0.7))
                axes[sample_count, 2].axis('off')
                
                sample_count += 1
                if sample_count >= 3:
                    break
        
        plt.tight_layout()
        plt.savefig(self.debug_dir / "input_data_samples.png")
        plt.close()
        
        print(f"✓ Input data inspection complete. Results saved to {self.debug_dir/'input_data_samples.png'}")
        
    def fix_data_normalization(self):
        """Fix data normalization in the data loader"""
        print("\nStep 2: Fixing Data Normalization...")
        
        # First, let's create a specific test case
        sample_data_path = self.data_paths['labeled']['images'][0]
        sample_label_path = self.data_paths['labeled']['labels'][0]
        
        print(f"Sample image path: {sample_data_path}")
        print(f"Sample mask path: {sample_label_path}")
        
        # Load the raw data
        if isinstance(sample_data_path, tf.Tensor):
            sample_data_path = sample_data_path.numpy().decode('utf-8')
        if isinstance(sample_label_path, tf.Tensor):
            sample_label_path = sample_label_path.numpy().decode('utf-8')
        
        # Load the raw numpy data
        try:
            raw_img = np.load(sample_data_path)
            raw_mask = np.load(sample_label_path)
            
            print(f"Raw image shape: {raw_img.shape}, dtype: {raw_img.dtype}")
            print(f"Raw mask shape: {raw_mask.shape}, dtype: {raw_mask.dtype}")
            print(f"Raw image range: [{np.min(raw_img)}, {np.max(raw_img)}]")
            print(f"Raw mask range: [{np.min(raw_mask)}, {np.max(raw_mask)}]")
            
            # Extract middle slice if 3D
            if len(raw_img.shape) == 3:
                middle_idx = raw_img.shape[1] // 2
                raw_img_2d = raw_img[:, middle_idx, :]
            else:
                raw_img_2d = raw_img
                
            if len(raw_mask.shape) == 3:
                middle_idx = raw_mask.shape[1] // 2
                raw_mask_2d = raw_mask[:, middle_idx, :]
            else:
                raw_mask_2d = raw_mask
                
            # Visualize raw, normalized and binarized versions
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Raw data
            axes[0, 0].imshow(raw_img_2d, cmap='gray')
            axes[0, 0].set_title("Raw Image")
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(raw_mask_2d, cmap='viridis')
            axes[1, 0].set_title("Raw Mask")
            axes[1, 0].axis('off')
            
            # Normalized data (min-max normalization)
            img_min = np.min(raw_img_2d)
            img_max = np.max(raw_img_2d)
            normalized_img = (raw_img_2d - img_min) / max(img_max - img_min, 1e-7)
            
            axes[0, 1].imshow(normalized_img, cmap='gray')
            axes[0, 1].set_title("Normalized Image")
            axes[0, 1].axis('off')
            
            # Binarized mask
            binary_mask = (raw_mask_2d > 0.5).astype(np.float32)
            
            axes[1, 1].imshow(binary_mask, cmap='viridis')
            axes[1, 1].set_title("Binarized Mask")
            axes[1, 1].axis('off')
            
            # Histogram of image values
            axes[0, 2].hist(raw_img_2d.flatten(), bins=50)
            axes[0, 2].set_title("Image Value Histogram")
            
            # Histogram of mask values
            axes[1, 2].hist(raw_mask_2d.flatten(), bins=50)
            axes[1, 2].set_title("Mask Value Histogram")
            
            plt.tight_layout()
            plt.savefig(self.debug_dir / "data_normalization_check.png")
            plt.close()
            
            print(f"✓ Data normalization check complete. Results saved to {self.debug_dir/'data_normalization_check.png'}")
        
        except Exception as e:
            print(f"Error in data normalization check: {str(e)}")
    
    def verify_model_outputs(self):
        """Check the model's outputs to ensure they're reasonable"""
        print("\nStep 3: Verifying Model Outputs...")
        
        # Create validation dataset
        val_ds = self.data_pipeline.create_dataset(
            self.data_paths['validation']['images'],
            self.data_paths['validation']['labels'],
            batch_size=4,
            augment=False
        )
        
        # Get a sample batch
        for images, masks in val_ds.take(1):
            break
        
        # Get model predictions
        predictions = self.model(images, training=False)
        
        # Visualize predictions
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        for i in range(min(4, images.shape[0])):
            # Original image
            axes[i, 0].imshow(images[i, ..., 0], cmap='gray')
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i, ..., 0], cmap='viridis')
            axes[i, 1].set_title(f"True Mask {i+1}")
            axes[i, 1].axis('off')
            
            # Model prediction
            pred_mask = tf.nn.sigmoid(predictions[i, ..., 0])
            axes[i, 2].imshow(pred_mask, cmap='viridis')
            axes[i, 2].set_title(f"Prediction {i+1} (Raw Sigmoid)")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.debug_dir / "model_output_check.png")
        plt.close()
        
        # Check sigmoid output ranges
        sigmoid_preds = tf.nn.sigmoid(predictions)
        print(f"Sigmoid predictions range: [{tf.reduce_min(sigmoid_preds)}, {tf.reduce_max(sigmoid_preds)}]")
        
        # Check thresholded predictions
        threshold_preds = tf.cast(sigmoid_preds > 0.5, tf.float32)
        print(f"Thresholded predictions unique values: {np.unique(threshold_preds.numpy())}")
        
        # Calculate and print Dice with both methods
        dice_scores = []
        dice_scores_corrected = []
        
        for i in range(images.shape[0]):
            # Method 1: Original dice calculation
            true_mask = masks[i, ..., 0]
            pred_mask = tf.cast(sigmoid_preds[i, ..., 0] > 0.5, tf.float32)
            
            intersection = tf.reduce_sum(true_mask * pred_mask)
            union = tf.reduce_sum(true_mask) + tf.reduce_sum(pred_mask)
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(float(dice))
            
            # Method 2: Alternative calculation (output may differ if implementation is incorrect)
            pred_sigmoid = sigmoid_preds[i, ..., 0]  # Keep as probability
            intersection_alt = 2. * tf.reduce_sum(true_mask * pred_sigmoid)
            union_alt = tf.reduce_sum(true_mask) + tf.reduce_sum(pred_sigmoid)
            dice_alt = (intersection_alt + 1e-6) / (union_alt + 1e-6)
            dice_scores_corrected.append(float(dice_alt))
        
        print(f"Original Dice scores: {dice_scores}, Mean: {np.mean(dice_scores)}")
        print(f"Alternative Dice scores: {dice_scores_corrected}, Mean: {np.mean(dice_scores_corrected)}")
        
        print(f"✓ Model output verification complete. Results saved to {self.debug_dir/'model_output_check.png'}")
    
    def check_dice_implementation(self):
        """Check and correct the Dice score implementation if needed"""
        print("\nStep 4: Checking Dice Score Implementation...")
        
        # Create synthetic data for dice testing
        print("Testing with synthetic data:")
        
        # Case 1: Perfect prediction
        y_true = np.zeros((1, 10, 10, 1))
        y_true[0, 3:7, 3:7, 0] = 1.0  # 4x4 square at center
        
        y_pred_perfect = np.zeros((1, 10, 10, 1))
        y_pred_perfect[0, 3:7, 3:7, 0] = 1.0  # Identical to ground truth
        
        # Case 2: No overlap
        y_pred_no_overlap = np.zeros((1, 10, 10, 1))
        y_pred_no_overlap[0, 0:4, 0:4, 0] = 1.0  # 4x4 square at top-left
        
        # Case 3: Partial overlap
        y_pred_partial = np.zeros((1, 10, 10, 1))
        y_pred_partial[0, 2:6, 2:6, 0] = 1.0  # 4x4 square shifted
        
        # Case 4: Prediction larger than truth
        y_pred_larger = np.zeros((1, 10, 10, 1))
        y_pred_larger[0, 2:8, 2:8, 0] = 1.0  # 6x6 square around truth
        
        # Convert to tensors
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred_perfect_tf = tf.convert_to_tensor(y_pred_perfect, dtype=tf.float32)
        y_pred_no_overlap_tf = tf.convert_to_tensor(y_pred_no_overlap, dtype=tf.float32)
        y_pred_partial_tf = tf.convert_to_tensor(y_pred_partial, dtype=tf.float32)
        y_pred_larger_tf = tf.convert_to_tensor(y_pred_larger, dtype=tf.float32)
        
        # Calculate dice scores with original method
        def compute_dice_original(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            if len(y_true.shape) > 3:
                y_true = y_true[..., 0]
                y_pred = y_pred[..., 0]
            
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
            union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
            
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            return tf.reduce_mean(dice)
        
        # Alternative implementation (more numerically stable with probabilities)
        def compute_dice_corrected(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            if len(y_true.shape) > 3:
                y_true = y_true[..., 0]
                y_pred = y_pred[..., 0]
            
            intersection = 2. * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
            denominator = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
            
            dice = (intersection + 1e-6) / (denominator + 1e-6)
            return tf.reduce_mean(dice)
            
        # Print results for different cases
        print("Case 1: Perfect prediction")
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_perfect_tf)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_perfect_tf)}")
        
        print("\nCase 2: No overlap")
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_no_overlap_tf)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_no_overlap_tf)}")
        
        print("\nCase 3: Partial overlap")
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_partial_tf)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_partial_tf)}")
        
        print("\nCase 4: Prediction larger than truth")
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_larger_tf)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_larger_tf)}")
        
        # Now test with sigmoid outputs (not thresholded)
        print("\nTesting with sigmoid (probability) outputs:")
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        
        # Create sigmoid-like outputs (values between 0-1)
        y_pred_sigmoid = np.zeros((1, 10, 10, 1))
        y_pred_sigmoid[0, 3:7, 3:7, 0] = 0.7  # High probability in correct area
        y_pred_sigmoid = tf.convert_to_tensor(y_pred_sigmoid, dtype=tf.float32)
        
        # Test with and without thresholding
        print("With thresholding:")
        y_pred_thresholded = tf.cast(y_pred_sigmoid > 0.5, tf.float32)
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_thresholded)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_thresholded)}")
        
        print("Without thresholding (using probabilities directly):")
        print(f"Original method: {compute_dice_original(y_true_tf, y_pred_sigmoid)}")
        print(f"Corrected method: {compute_dice_corrected(y_true_tf, y_pred_sigmoid)}")
        
        print("✓ Dice implementation check complete.")
        
    def fix_supervised_trainer(self):
        """Fix the supervised trainer with corrected implementations"""
        print("\nStep 5: Creating a Corrected implementation...")
        
        # Create a file with corrected implementations
        dice_fix_code = f"""
#Corrected Dice Implementation for SupervisedTrainer class

def compute_dice_corrected(y_true, y_pred):
    """Compute Dice score with proper implementation"""
    # Apply sigmoid activation to get probabilities
    y_pred_sigmoid = tf.nn.sigmoid(y_pred) 
    
    # Cast to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred_sigmoid = tf.cast(y_pred_sigmoid, tf.float32)
    
    # Get the last channel if multi-channel
    if len(y_true.shape) > 3:
        y_true = y_true[..., -1]
    if len(y_pred_sigmoid.shape) > 3:
        y_pred_sigmoid = y_pred_sigmoid[..., -1]
    
    # Threshold predictions at 0.5 to get binary mask
    y_pred_binary = tf.cast(y_pred_sigmoid > 0.5, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred_binary, axis=[1, 2])
    sum_true = tf.reduce_sum(y_true, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred_binary, axis=[1, 2])
    
    # Calculate Dice coefficient with smoothing term
    dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
    
    # Handle edge cases
    dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
    
    return tf.reduce_mean(dice)

def validate_corrected(self, val_dataset):
    """Run validation with corrected dice implementation"""
    dice_scores = []
    
    for batch in val_dataset:
        images, labels = batch
        
        # Get model predictions
        predictions = self.model(images, training=False)
        
        # Calculate dice score
        dice = compute_dice_corrected(labels, predictions)
        
        if not tf.math.is_nan(dice):
            dice_scores.append(float(dice))
    
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        print(f"Validation Dice Score: {mean_dice:.4f}")
        return mean_dice
    
    return 0.0
"""
        
        with open(self.debug_dir / "dice_fix_code.py", "w") as f:
            f.write(dice_fix_code)
            
        print(f"✓ Fixed implementation written to {self.debug_dir/'dice_fix_code.py'}")
    
    def validate_with_fixed_implementation(self):
        """Validate the model using the fixed implementation"""
        print("\nStep 6: Validating model with fixed implementation...")
        
        # Create validation dataset
        val_ds = self.data_pipeline.create_dataset(
            self.data_paths['validation']['images'],
            self.data_paths['validation']['labels'],
            batch_size=4,
            augment=False
        )
        
        # Fixed dice implementation
        def compute_dice_fixed(y_true, y_pred):
            # Apply sigmoid activation to get probabilities
            y_pred_sigmoid = tf.nn.sigmoid(y_pred) 
            
            # Cast to float32 for numerical stability
            y_true = tf.cast(y_true, tf.float32)
            y_pred_sigmoid = tf.cast(y_pred_sigmoid, tf.float32)
            
            # Get the last channel if multi-channel
            if len(y_true.shape) > 3:
                y_true = y_true[..., -1]
            if len(y_pred_sigmoid.shape) > 3:
                y_pred_sigmoid = y_pred_sigmoid[..., -1]
            
            # Threshold predictions at 0.5 to get binary mask
            y_pred_binary = tf.cast(y_pred_sigmoid > 0.5, tf.float32)
            
            # Calculate intersection and union
            intersection = tf.reduce_sum(y_true * y_pred_binary, axis=[1, 2])
            sum_true = tf.reduce_sum(y_true, axis=[1, 2])
            sum_pred = tf.reduce_sum(y_pred_binary, axis=[1, 2])
            
            # Calculate Dice coefficient with smoothing term
            dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
            
            # Handle edge cases
            dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
            
            return tf.reduce_mean(dice)
        
        # Run validation
        dice_scores = []
        
        print("Running validation with fixed implementation...")
        for batch in tqdm(val_ds):
            images, labels = batch
            
            # Get model predictions
            predictions = self.model(images, training=False)
            
            # Calculate dice score
            dice = compute_dice_fixed(labels, predictions)
            
            if not tf.math.is_nan(dice):
                dice_scores.append(float(dice))
        
        if dice_scores:
            mean_dice = np.mean(dice_scores)
            print(f"\nValidation Results:")
            print(f"Mean Dice Score using fixed implementation: {mean_dice:.4f}")
        else:
            print("No valid Dice scores calculated.")
            
        # Save visualization of predictions with dice scores
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        sample_count = 0
        for batch in val_ds.take(2):
            images, labels = batch
            predictions = self.model(images, training=False)
            
            for i in range(min(2, images.shape[0])):
                img = images[i, ..., 0].numpy()
                true_mask = labels[i, ..., 0].numpy() 
                pred_sigmoid = tf.nn.sigmoid(predictions[i, ..., 0]).numpy()
                pred_binary = (pred_sigmoid > 0.5).numpy().astype(np.float32)
                
                # Calculate dice for this sample
                intersection = np.sum(true_mask * pred_binary)
                sum_true = np.sum(true_mask)
                sum_pred = np.sum(pred_binary)
                dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
                
                row = sample_count // 2
                
                # Plot image
                axes[row, 0].imshow(img, cmap='gray')
                axes[row, 0].set_title(f"Image {sample_count+1}")
                axes[row, 0].axis('off')
                
                # Plot true mask
                axes[row, 1].imshow(true_mask, cmap='viridis')
                axes[row, 1].set_title(f"True Mask {sample_count+1}")
                axes[row, 1].axis('off')
                
                # Plot predicted mask with dice
                axes[row, 2].imshow(pred_binary, cmap='viridis')
                axes[row, 2].set_title(f"Pred Mask (Dice={dice:.4f})")
                axes[row, 2].axis('off')
                
                sample_count += 1
                if sample_count >= 4:
                    break
                    
            if sample_count >= 4:
                break
        
        plt.tight_layout()
        plt.savefig(self.debug_dir / "fixed_validation_results.png")
        plt.close()
        
        print(f"✓ Validation visualization saved to {self.debug_dir/'fixed_validation_results.png'}")
    
    def run_all_debug_steps(self):
        """Run all debugging steps in sequence"""
        print("Starting Dice score debugging process...")
        self.inspect_input_data()
        self.fix_data_normalization()
        self.verify_model_outputs()
        self.check_dice_implementation()
        self.fix_supervised_trainer()
        self.validate_with_fixed_implementation()
        print("\nDebugging complete. Please check the results and apply the suggested fixes.")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Dice scores for segmentation model")
    parser.add_argument('--data_dir', type=str, default='/scratch/lustre/home/mdah0000/images/cropped',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    debugger = DiceScoreDebugger(args)
    debugger.run_all_debug_steps()