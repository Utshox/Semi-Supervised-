#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # In case you want to plot at the end from main
import pandas as pd

# Assuming these modules are in the same directory or Python path
from config import ExperimentConfig, StableSSLConfig # Make sure StableSSLConfig has MixMatch params
from data_loader_tf2 import DataPipeline
# PancreasSeg is used by MixMatchTrainer
from train_ssl_tf2n import MixMatchTrainer # Import your MixMatchTrainer
from run_mean_teacher_v2 import prepare_data_paths 


AUTOTUNE = tf.data.experimental.AUTOTUNE

def main(args):
    # --- GPU SETUP (Same as run_mean_teacher_v2.py) ---
    gpus_for_setup = tf.config.experimental.list_physical_devices('GPU')
    if gpus_for_setup:
        try:
            for gpu_setup in gpus_for_setup:
                tf.config.experimental.set_memory_growth(gpu_setup, True)
            print(f"SUCCESS: Set memory growth for {len(gpus_for_setup)} GPU(s).")
        except RuntimeError as e_setup:
            print(f"ERROR setting memory growth: {e_setup}", file=sys.stderr)
    else:
        print("No GPUs found by TensorFlow for memory growth setup.")


    # 1. Setup Configuration
    # output_dir_root is passed from shell, experiment_name is also from shell
    exp_config_obj = ExperimentConfig(
        experiment_name=args.experiment_name, # This will be the full name with params and date
        experiment_type='semi-supervised-mixmatch',
        results_dir=Path(args.output_dir) # This is OUTPUT_DIR_ROOT from shell
    )
    # The MixMatchTrainer will create its own experiment_name subdir within this Path(args.output_dir)
    # So, exp_config_obj.results_dir is actually the base for where trainer saves.
    # Let's adjust how MixMatchTrainer forms its output_dir or how we pass it.
    # For now, let MixMatchTrainer handle creating its own subfolder.

    model_data_config = StableSSLConfig(
        img_size_x=args.img_size, img_size_y=args.img_size,
        num_channels=1, num_classes=1, # Binary segmentation
        batch_size=args.batch_size, 
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate, # For MixMatchTrainer's optimizer
        # MixMatch specific params from args, to be put into config
        mixmatch_T = args.mixmatch_T,
        mixmatch_K = args.mixmatch_K,
        mixmatch_alpha = args.mixmatch_alpha,
        mixmatch_consistency_max = args.mixmatch_consistency_max,
        mixmatch_rampup_steps = args.mixmatch_rampup_steps,
        # EMA params for MixMatch if its trainer uses them (current one doesn't use these EMA specific from config)
        # initial_ema_decay = args.initial_ema_decay
        # ema_decay = args.ema_decay 
        # ema_warmup_steps = args.ema_warmup_steps
        output_dir = args.output_dir, # Pass the root output dir
        experiment_name = args.experiment_name # Pass the unique experiment name
    )
    if not hasattr(model_data_config, 'n_filters'): model_data_config.n_filters = 32


    tf.print("Preparing data paths for MixMatch...")
    # MixMatch also needs labeled, unlabeled, validation
    # num_labeled from args will define the split for training
    data_paths = prepare_data_paths(args.data_dir, args.num_labeled, args.num_validation, args.seed)

    # Instantiate MixMatch Trainer
    # The trainer will create its own output subdirectories based on config.output_dir and config.experiment_name
    mixmatch_trainer = MixMatchTrainer(config=model_data_config)

    # Train
    # The train method in MixMatchTrainer needs num_epochs and optionally steps_per_epoch
    tf.print(f"--- Starting MixMatch Training ({args.num_epochs} epochs) ---")
    
    # Calculate steps_per_epoch for MixMatch
    # This is typically based on the number of labeled examples
    num_labeled_files = len(data_paths['labeled']['images'])
    # Assuming each .npy file yields 1 slice for this calculation, as per your data structure
    num_labeled_samples_total = num_labeled_files 
    if num_labeled_samples_total == 0 and args.num_labeled > 0:
        tf.print("CRITICAL ERROR: No labeled files found for MixMatch training despite num_labeled > 0. Check data_paths.", output_stream=sys.stderr)
        sys.exit(1)
    
    calculated_steps_per_epoch = 100 # Default if no labeled data (should not happen)
    if num_labeled_samples_total > 0 :
        calculated_steps_per_epoch = (num_labeled_samples_total + args.batch_size - 1) // args.batch_size
    
    tf.print(f"MixMatch using steps_per_epoch: {calculated_steps_per_epoch} (based on {num_labeled_samples_total} labeled items)")

    mixmatch_history = mixmatch_trainer.train(
        data_paths=data_paths,
        num_epochs=args.num_epochs,
        steps_per_epoch=calculated_steps_per_epoch # Pass calculated steps
    )
    
    tf.print(f"MixMatch training completed.")
    tf.print(f"Results, checkpoints, and plots saved in: {mixmatch_trainer.output_dir}")

    # Final plot (MixMatchTrainer's train method already calls plot_progress)
    # mixmatch_trainer.plot_progress(args.num_epochs, final_plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MixMatch for pancreas segmentation.")
    # Common arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./MixMatch_Experiments_Root", help="Root directory for all MixMatch experiment outputs.")
    parser.add_argument("--experiment_name", type=str, default=f"MixMatchRun_{time.strftime('%Y%m%d_%H%M%S')}", help="Specific name for this experiment run.")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_labeled", type=int, default=30, help="Number of labeled patient volumes/files for training.")
    parser.add_argument("--num_validation", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4) # MixMatch often uses larger batches if possible
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.002) # MixMatch often uses higher LR with schedulers
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2]) # For Keras Progbar in trainer
    parser.add_argument("--early_stopping_patience", type=int, default=30)


    # MixMatch specific arguments
    parser.add_argument("--mixmatch_T", type=float, default=0.5, help="Temperature for sharpening pseudo-labels.")
    parser.add_argument("--mixmatch_K", type=int, default=2, help="Number of augmentations for guessing pseudo-labels.")
    parser.add_argument("--mixmatch_alpha", type=float, default=0.75, help="Alpha parameter for Beta distribution in MixUp.")
    parser.add_argument("--mixmatch_consistency_max", type=float, default=100.0, help="Max weight for unlabeled consistency loss.")
    parser.add_argument("--mixmatch_rampup_steps", type=int, default=10000, help="Number of training steps for lambda_u ramp-up (e.g. 16 epochs * 700 steps/epoch).")
    
    args = parser.parse_args()
    
    # Update experiment name to include key parameters dynamically AFTER parsing args
    args.experiment_name = f"MixMatch_L{args.num_labeled}_B{args.batch_size}_K{args.mixmatch_K}_Alpha{args.mixmatch_alpha}_Temp{args.mixmatch_T}_ConsMax{args.mixmatch_consistency_max}_{time.strftime('%Y%m%d_%H%M%S')}"

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    main(args)