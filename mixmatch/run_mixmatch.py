#!/usr.bin/env python3
import argparse
import sys
from pathlib import Path
import random
import time
import numpy as np
import tensorflow as tf
import pandas as pd # For saving history

# Assuming these are in the same directory or accessible via PYTHONPATH
from config import ExperimentConfig, StableSSLConfig
from data_loader_tf2 import DataPipeline
# PancreasSeg model is in models_tf2.py, MixMatchTrainer is in train_ssl_tf2n.py
from train_ssl_tf2n import MixMatchTrainer # Ensure PancreasSeg is imported within train_ssl_tf2n or models_tf2

# --- Helper to prepare data paths (simplified from run_mean_teacher_v2.py) ---
def prepare_data_paths_for_ssl(data_dir_str: str, num_labeled_patients: int, num_validation_patients: int, seed: int = 42):
    data_dir = Path(data_dir_str)
    all_patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pancreas_")])
    
    if not all_patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {data_dir_str} matching 'pancreas_*'. Check --data_dir and preprocessed data.")

    random.seed(seed)
    random.shuffle(all_patient_dirs)

    if len(all_patient_dirs) < num_labeled_patients + num_validation_patients:
        raise ValueError(
            f"Not enough patient data ({len(all_patient_dirs)}) for the specified splits: "
            f"{num_labeled_patients} labeled, {num_validation_patients} validation. Need at least {num_labeled_patients + num_validation_patients} total."
        )

    validation_patient_dirs = all_patient_dirs[:num_validation_patients]
    remaining_patient_dirs = all_patient_dirs[num_validation_patients:]
    
    labeled_patient_dirs = remaining_patient_dirs[:num_labeled_patients]
    # All remaining data after labeled and validation becomes unlabeled
    unlabeled_patient_dirs = remaining_patient_dirs[num_labeled_patients:] 

    print(f"Data Split: Total patient dirs: {len(all_patient_dirs)}, "
          f"Labeled: {len(labeled_patient_dirs)}, "
          f"Unlabeled: {len(unlabeled_patient_dirs)}, "
          f"Validation: {len(validation_patient_dirs)}")

    def get_file_paths_from_patient_dirs(patient_dirs_list, is_labeled_data=True):
        image_paths = []
        label_paths = [] if is_labeled_data else None # Unlabeled data doesn't have label_paths
        
        for p_dir in patient_dirs_list:
            img_file = p_dir / "image.npy"
            mask_file = p_dir / "mask.npy" # Only relevant if is_labeled_data is True
            
            if img_file.exists():
                if is_labeled_data: # For labeled and validation sets
                    if mask_file.exists():
                        image_paths.append(str(img_file))
                        label_paths.append(str(mask_file))
                    else:
                        print(f"Warning: Missing mask.npy in {p_dir} for a supposedly labeled/validation patient. Skipping.", file=sys.stderr)
                else: # For unlabeled set, only image is needed
                    image_paths.append(str(img_file))
            else:
                print(f"Warning: Missing image.npy in {p_dir}. Skipping.", file=sys.stderr)
                
        return (image_paths, label_paths) if is_labeled_data else image_paths # Return only image_paths for unlabeled

    labeled_images, labeled_labels = get_file_paths_from_patient_dirs(labeled_patient_dirs, is_labeled_data=True)
    # For unlabeled data, we only need image paths.
    unlabeled_images = get_file_paths_from_patient_dirs(unlabeled_patient_dirs, is_labeled_data=False)
    validation_images, validation_labels = get_file_paths_from_patient_dirs(validation_patient_dirs, is_labeled_data=True)
    
    if not labeled_images: print("CRITICAL WARNING: No Labeled image paths found after filtering.", file=sys.stderr)
    if not unlabeled_images and len(unlabeled_patient_dirs) > 0 : print("WARNING: Unlabeled patient dirs were selected, but no valid image.npy files found in them.", file=sys.stderr)
    elif not unlabeled_images: print("INFO: No Unlabeled image paths found (this might be expected if num_labeled + num_validation covers all data).", file=sys.stderr)
    if not validation_images: print("CRITICAL WARNING: No Validation image paths found after filtering.", file=sys.stderr)

    return {
        'labeled': {'images': labeled_images, 'labels': labeled_labels},
        'unlabeled': {'images': unlabeled_images}, # Only 'images' key for unlabeled
        'validation': {'images': validation_images, 'labels': validation_labels}
    }

def main_mixmatch(args):
    # --- Setup Experiment Config ---
    # ExperimentConfig is for overall experiment directory management, not trainer's internal output_dir
    # The MixMatchTrainer itself will create subdirectories within args.output_dir / args.experiment_name

    # --- Setup Model and Data Config (StableSSLConfig) ---
    model_data_config = StableSSLConfig(
        img_size_x=args.img_size, 
        img_size_y=args.img_size,
        num_channels=1, 
        num_classes=1, # Binary segmentation (pancreas vs background)
        batch_size=args.batch_size,
        # Pass MixMatch specific HPs to StableSSLConfig so MixMatchTrainer can access them
        mixmatch_T = args.mixmatch_T,
        mixmatch_K = args.mixmatch_K,
        mixmatch_alpha = args.mixmatch_alpha,
        mixmatch_consistency_max = args.consistency_max,
        mixmatch_rampup_steps = args.consistency_rampup_steps, # Rampup in steps
        learning_rate = args.learning_rate, # For optimizer
        ema_decay = args.ema_decay, # For teacher EMA final decay
        initial_ema_decay = args.initial_teacher_ema_decay, # For teacher EMA initial decay
        ema_warmup_steps = args.teacher_ema_warmup_steps, # Steps for EMA decay rampup
        early_stopping_patience = args.early_stopping_patience,
        output_dir = args.output_dir, # Base output dir for experiments
        experiment_name = args.experiment_name # Specific experiment name
    )
    model_data_config.checkpoint_dir = Path(args.output_dir) / args.experiment_name / "trainer_checkpoints" # Override
    model_data_config.log_dir = Path(args.output_dir) / args.experiment_name / "trainer_logs" # Override

    # --- GPU Setup ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s). TensorFlow version: {tf.__version__}")
        except RuntimeError as e:
            print(f"GPU Error: {e}", file=sys.stderr)
    else:
        print("No GPU found. Using CPU.", file=sys.stderr)

    # --- Prepare Data Paths ---
    print("Preparing data paths...")
    data_paths = prepare_data_paths_for_ssl(args.data_dir, args.num_labeled, args.num_validation, args.seed)
    
    if not data_paths['labeled']['images'] or not data_paths['validation']['images']:
        sys.exit("CRITICAL ERROR: Labeled or Validation data paths are empty. Check data preparation and paths.")
    if not data_paths['unlabeled']['images']:
        print("WARNING: No unlabeled data found. MixMatch might not perform as expected without unlabeled samples.", file=sys.stderr)
        # Potentially exit or allow to continue if this is an intended test
        # sys.exit("CRITICAL ERROR: Unlabeled data paths are empty. MixMatch requires unlabeled data.")


    # --- Initialize Trainer ---
    # The MixMatchTrainer's __init__ will use model_data_config to set up its internal paths
    trainer = MixMatchTrainer(config=model_data_config)

    # --- Start Training ---
    print(f"--- Starting MixMatch Training ({args.num_epochs} epochs) ---")
    start_time_train = time.time()
    
    # The train method in MixMatchTrainer will handle its own history and plotting
    # It needs num_epochs and optionally steps_per_epoch
    trainer.train(data_paths, num_epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch) 
    
    end_time_train = time.time()
    print(f"MixMatch training completed in {(end_time_train - start_time_train)/3600:.2f} hours.")
    print(f"All results, checkpoints, and logs saved in: {trainer.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MixMatch training for pancreas segmentation.")
    # Data and Output
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed data directory (containing patient_*/image.npy and mask.npy).")
    parser.add_argument("--output_dir", type=str, default="./mixmatch_experiments", help="Base directory to save experiment results.")
    parser.add_argument("--experiment_name", type=str, default=f"MixMatch_L{0}_Time{time.strftime('%Y%m%d_%H%M%S')}", help="Specific name for this experiment run.")
    
    # Data Split
    parser.add_argument("--num_labeled", type=int, default=30, help="Number of labeled patient volumes for training.")
    parser.add_argument("--num_validation", type=int, default=10, help="Number of patient volumes for validation.")
    
    # Training Generic HPs
    parser.add_argument("--img_size", type=int, default=256, help="Target image size (height and width).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs for MixMatch training.")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch. If None, estimated from labeled data.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--early_stopping_patience", type=int, default=25, help="Patience for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # MixMatch Specific HPs
    parser.add_argument("--mixmatch_T", type=float, default=0.5, help="Temperature for sharpening pseudo-labels.")
    parser.add_argument("--mixmatch_K", type=int, default=2, help="Number of augmentations for generating pseudo-labels.")
    parser.add_argument("--mixmatch_alpha", type=float, default=0.75, help="Alpha parameter for Beta distribution in MixUp.")
    parser.add_argument("--consistency_max", type=float, default=10.0, help="Maximum weight for the consistency loss.")
    parser.add_argument("--consistency_rampup_steps", type=int, default=5000, help="Number of training steps for consistency weight to ramp up to max_consistency_weight.") # Changed from epochs to steps

    # Teacher EMA HPs
    parser.add_argument("--initial_teacher_ema_decay", type=float, default=0.95, help="Initial EMA decay for teacher model during its warmup phase.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Final (base) EMA decay for updating teacher model.")
    parser.add_argument("--teacher_ema_warmup_steps", type=int, default=2000, help="Number of training steps for teacher EMA decay to ramp up to base_ema_decay.") # Changed from epochs to steps
    
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2], help="Verbosity level for training.") # Keras verbose

    args = parser.parse_args()

    # Update experiment name if L0 was default
    if f"L{0}" in args.experiment_name:
         args.experiment_name = f"MixMatch_L{args.num_labeled}_Time{time.strftime('%Y%m%d_%H%M%S')}"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("Arguments received:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
    main_mixmatch(args)