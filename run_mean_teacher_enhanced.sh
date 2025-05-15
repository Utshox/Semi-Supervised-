#!/bin/bash
# SLURM batch job script for Mean Teacher semi-supervised learning with advanced techniques

#SBATCH -p gpu                        # Use gpu partition 
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH -n 8                          # Request 8 CPU cores
#SBATCH --mem=64G                     # Request 64GB RAM
#SBATCH --time=00:10:00               # Time limit to 20 minutes
#SBATCH -o meanteacher-adv-%j.out     # Output file name with job ID
#SBATCH -e meanteacher-adv-%j.err     # Error file name with job ID
#SBATCH --mail-type=END,FAIL          # Send email when job ends or fails
#SBATCH --job-name=pancreas_mt_adv    # Add descriptive job name
# Path to your data directory - using your HPC paths
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2/"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mean_teacher_advanced_results"

echo "========================================================"
echo "Running Advanced Mean Teacher Learning - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory: $WORK_DIR"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/visualizations

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Add user's local bin to PATH to find pip-installed executables
export PATH="/scratch/lustre/home/mdah0000/.local/bin:$PATH"

# Memory optimization settings
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_GPU_HOST_MEM_LIMIT_IN_MB=4096
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"  # Enable XLA JIT compilation

# Find Python executable
PYTHON_CMD=""
for cmd in python3 python /usr/bin/python3 /usr/bin/python; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        echo "Found Python: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python executable not found."
    exit 1
fi

# Install required packages
$PYTHON_CMD -m pip install --quiet tensorflow==2.15.0 tqdm matplotlib psutil scipy albumentations # Changed to base tensorflow package

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create an enhanced Mean Teacher implementation script
echo "Creating Advanced Mean Teacher script..."
cat > $WORK_DIR/run_mean_teacher_enhanced.py << 'EOF'
#!/usr/bin/env python3
# Advanced Mean Teacher implementation for pancreas segmentation with strong augmentations

# --- Project Setup and Configuration Comments ---
# This script is designed for the HPC environment at /scratch/lustre/home/mdah0000/
# Key paths are typically passed as arguments (data_dir, output_dir) or defined in the calling shell script.
# - WORK_DIR (for scripts and base modules, from shell): /scratch/lustre/home/mdah0000/smm/v14
# - DATA_DIR (for input images, from shell/arg): /scratch/lustre/home/mdah0000/images/preprocessed_v2/
# - OUTPUT_DIR (for results, from shell/arg): $WORK_DIR/mean_teacher_advanced_results (or as specified)
# The sys.path.append in the original script (if used) would ensure modules from WORK_DIR are imported.
# This generated script relies on Python's default module search path, assuming necessary modules
# (config, train_ssl_tf2n, data_loader_tf2, models_tf2, main) are accessible.
# ---

import tensorflow as tf
from pathlib import Path
import argparse
import time
from datetime import datetime
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
import sys # Import sys module

# Import your modules
# --- Comment: Module Imports ---
# The following modules are part of this project and are expected to be in the Python path.
# - config: Contains configuration classes like StableSSLConfig and ExperimentConfig.
# - train_ssl_tf2n: Provides the base StableSSLTrainer class.
# - data_loader_tf2: Contains DataPipeline for creating TensorFlow datasets.
# - models_tf2: Defines the PancreasSeg model architecture.
# - main: Includes utility functions like prepare_data_paths and setup_gpu.
# ---
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import StableSSLTrainer
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from main import prepare_data_paths, setup_gpu

print("TensorFlow version:", tf.__version__)

# --- Comment: GPU Setup (via main.setup_gpu) ---
# The setup_gpu() function (imported from main.py) is responsible for configuring GPU resources.
# This typically involves enabling memory growth to prevent TensorFlow from allocating all GPU
# memory at once, which is crucial for running on shared HPC resources.
# Mixed precision (TF_ENABLE_AUTO_MIXED_PRECISION=1) might be enabled in the calling shell script
# for performance, and the Python code should be compatible with it (e.g., using LossScaleOptimizer).
# ---

def collect_garbage():
    """Force garbage collection to free memory"""
    gc.collect()
    tf.keras.backend.clear_session()

class AdvancedMeanTeacherTrainer(StableSSLTrainer):
    """Advanced version of Mean Teacher trainer with memory optimization and strong augmentations"""
    # --- Comment: Class Initialization & Supervised Baseline ---
    # This AdvancedMeanTeacherTrainer builds upon StableSSLTrainer.
    # The supervised aspects of the model (PancreasSeg architecture, Dice/BCE loss)
    # have demonstrated strong performance (Val Dice ~0.8577), providing a robust
    # foundation for the student model. The Mean Teacher components aim to leverage
    # this by using unlabeled data for consistency training.
    # ---
    
    def __init__(self, config):
        # Set warmup_epochs before calling super().__init__
        self.warmup_epochs = 5  # Warmup epochs for training stability
        self.best_dice = 0
        self.validation_plateau = 0
        
        # Now call parent's init which will indirectly call _setup_training_params
        super().__init__(config)
        
        # Initialize use_tqdm, can be controlled by config
        self.use_tqdm = getattr(self.config, 'use_tqdm', True) # Use getattr for safe access

        # The rest of your init logic
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': []
        }
        self._setup_model()
        
    def _setup_model(self):
        """Setup the U-Net model with improved architecture"""
        print("Setting up improved U-Net model for Mean Teacher...")
        
        # --- Comment: Model Architecture (PancreasSeg) ---
        # Both student and teacher models utilize the PancreasSeg architecture.
        # This architecture was successful in the supervised baseline (Val Dice ~0.8577).
        # Key characteristics (e.g., n_filters=32) are set here.
        # The teacher model starts as an identical copy of the student.
        # ---
        # Create the model
        self.config.n_filters = 32  # Base filters
        
        # Create student model with dropout for regularization
        self.model = PancreasSeg(self.config)
        
        # Create teacher model (no dropout during inference)
        self.teacher_model = PancreasSeg(self.config)
        
        # Initialize models with dummy input using (height, width, channels)
        dummy_input = tf.zeros((1, self.config.img_size_y, self.config.img_size_x, self.config.num_channels))
        # --- Build student model with training=True ---
        print("Building student model (self.model) with training=True...")
        _ = self.model(dummy_input, training=True)
        # --- Build teacher model with training=True initially, as it gets student weights ---
        # It will be CALLED with training=False during inference.
        print("Building teacher model (self.teacher_model) with training=True...")
        _ = self.teacher_model(dummy_input, training=True)
        
        # Copy weights from student to teacher initially
        self.teacher_model.set_weights(self.model.get_weights())
        
        print(f"Model created with input shape: ({self.config.img_size_y}, {self.config.img_size_x}, {self.config.num_channels})")
        print(f"Base filters: {self.config.n_filters}")
        
        # Print model parameters
        total_params = self.model.count_params()
        print(f"Total model parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
        
    def _setup_training_params(self, steps_per_epoch):
        """Setup training parameters with advanced learning rate scheduler"""
        
        # --- Comment: Learning Rate Schedule (CosineWarmupSchedule) ---
        # A CosineWarmupSchedule is used for the learning rate.
        # This involves a linear ramp-up from min_lr to max_lr over warmup_steps,
        # followed by a cosine decay down to min_lr over the remaining total_steps.
        # This schedule is often effective for stabilizing training early and achieving
        # good convergence. Parameters like max_lr, min_lr, and warmup_epochs are critical.
        # The supervised model's success might inform initial LR choices, but tuning
        # for semi-supervised learning is often necessary.
        # ---
        class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            """Learning rate scheduler with linear warmup and cosine decay"""
            def __init__(self, max_lr, min_lr, warmup_steps, total_steps):
                super(CosineWarmupSchedule, self).__init__()
                self.max_lr = tf.cast(max_lr, tf.float32)
                self.min_lr = tf.cast(min_lr, tf.float32)
                self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                self.total_steps = tf.cast(total_steps, tf.float32)
                
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                
                # Warmup phase
                warmup_lr = self.min_lr + step * (self.max_lr - self.min_lr) / self.warmup_steps
                
                # Cosine decay phase
                decay_steps = self.total_steps - self.warmup_steps
                cosine_decay = 0.5 * (1 + tf.cos(
                    tf.constant(np.pi) * (step - self.warmup_steps) / decay_steps))
                cosine_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
                
                # Return either warmup or cosine decay based on current step
                return tf.cond(step <= self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)
            
            def get_config(self):
                return {
                    "max_lr": self.max_lr,
                    "min_lr": self.min_lr,
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps
                }
        
        # Calculate total steps with more epochs
        total_steps = self.config.num_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        # Learning rate parameters (slightly higher max learning rate for better exploration)
        max_lr = 6e-5  # Closer to supervised 5e-5 (was 8e-4)
        min_lr = 1e-6  # Matches supervised 1e-6 (was 5e-6)
        
        # Create learning rate scheduler
        self.lr_schedule = CosineWarmupSchedule(
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Set Mean Teacher specific parameters
        # --- Comment: Mean Teacher Hyperparameters - EMA Decay ---
        # ema_decay: Controls the update speed of the teacher model.
        # This implementation uses a phased EMA decay:
        # 1. Warmup Phase (first `self.warmup_epochs`): Teacher weights are a direct copy of student's.
        #    The `_get_ema_decay` returns `ema_decay_start` for logging but isn't used for EMA calc.
        # 2. Stabilization Phase (next `self.ema_stabilization_duration` epochs):
        #    A fixed, high `ema_stabilization_decay` (e.g., 0.999) is used to create a very stable teacher
        #    once the student has some initial training.
        # 3. Ramp Phase (after stabilization, up to `self.consistency_rampup_epochs`):
        #    EMA decay ramps from `ema_stabilization_decay` to `ema_decay_end`.
        # This phased approach aims to balance teacher stability with responsiveness.
        # ---
        self.ema_decay_start = 0.95    # Original start, for reference
        self.ema_stabilization_duration = 15 # NEW: Increased from 10 to 15 epochs
        self.ema_stabilization_decay = 0.999  # NEW: Increased from 0.99 to 0.999 for extreme stability
        self.ema_decay_end = 0.999     # NEW: Match stabilization decay for a fixed EMA after warmup
        self.consistency_weight_max = 20.0  # Maximum consistency weight
        # self.consistency_rampup_epochs = 80  # Ramp up consistency weight over epochs (original definition)
        # NEW: consistency_rampup_epochs now means the epoch index by which ramps complete.
        # --- Comment: Mean Teacher Hyperparameters - Consistency Weight ---
        # consistency_weight_max: The maximum weight applied to the consistency loss component.
        # consistency_rampup_epochs: The epoch by which the consistency weight ramps up to its maximum.
        # The ramp-up starts *after* the EMA stabilization phase (see `get_consistency_weight`).
        # This delay allows the teacher to become stable before its predictions heavily influence the student.
        # Strategies:
        # - Start with a moderate `consistency_weight_max` (e.g., 10-50) and tune.
        # - The ramp-up period should be long enough for the teacher to stabilize but not so long
        #   that unlabeled data isn't leveraged effectively for a significant portion of training.
        # ---
        self.consistency_rampup_epochs = 80 
        
        # Optimizer with weight decay
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            weight_decay=1e-5  # L2 regularization through weight decay
        )
        
        # --- Wrap optimizer for mixed precision if enabled ---
        if os.environ.get('TF_ENABLE_AUTO_MIXED_PRECISION') == '1':
            print("INFO: Wrapping optimizer with LossScaleOptimizer for mixed precision.")
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        else:
            print("INFO: Not using LossScaleOptimizer as TF_ENABLE_AUTO_MIXED_PRECISION is not '1'.")

        print(f"Learning rate schedule: Cosine with warmup from {min_lr} to {max_lr}")
        
        # Updated print statement for EMA decay logic
        if self.ema_stabilization_decay == self.ema_decay_end:
            print(f"EMA decay: Direct copy (warmup epochs 0-{self.warmup_epochs-1}), then fixed at {self.ema_stabilization_decay} (epochs {self.warmup_epochs}-onwards)")
        else:
            print(f"EMA decay: Direct copy (warmup epochs 0-{self.warmup_epochs-1}), then {self.ema_stabilization_decay} (stabilization epochs {self.warmup_epochs}-{self.warmup_epochs + self.ema_stabilization_duration-1}), then ramp to {self.ema_decay_end}")
        
        print(f"Consistency weight max: {self.consistency_weight_max}")
        
    def _get_ema_decay(self, epoch):
        """Get EMA decay rate based on current epoch with stabilization phase"""
        # epoch is 0-indexed current epoch
        
        # --- Comment: EMA Decay Logic ---
        # This function implements the phased EMA decay strategy:
        # - Warmup: Returns `ema_decay_start` (for logging; actual update is direct copy).
        # - Stabilization: Returns `ema_stabilization_decay`.
        # - Ramp: Linearly interpolates from `ema_stabilization_decay` to `ema_decay_end`.
        # This complex schedule is an area for potential simplification or tuning if issues arise.
        # The core idea is to have a very stable teacher after initial student learning.
        # ---
        # Phase 1: Warmup (direct copy is handled in update_teacher_model)
        # This value is returned for logging during warmup but not used for EMA calculation then.
        if epoch < self.warmup_epochs: # e.g., epochs 0-4
             return self.ema_decay_start 

        # Phase 2: EMA Stabilization (e.g., epochs 5-14)
        stabilization_phase_end_epoch = self.warmup_epochs + self.ema_stabilization_duration
        
        if epoch < stabilization_phase_end_epoch: # e.g., 5 <= epoch < 15
            return self.ema_stabilization_decay # e.g. 0.99
            
        # Phase 3: Ramp from self.ema_stabilization_decay to self.ema_decay_end
        # This ramp starts at stabilization_phase_end_epoch (e.g., 15)
        # and ends by self.consistency_rampup_epochs (e.g., 80).
        ramp_start_epoch = stabilization_phase_end_epoch
        
        # Ensure consistency_rampup_epochs is after ramp_start_epoch for a valid ramp
        if self.consistency_rampup_epochs <= ramp_start_epoch:
             # If ramp duration is zero or negative, return the target end decay or stabilization decay
             return self.ema_decay_end 

        # Calculate progress in the current ramp segment
        # total_ramp_duration = self.consistency_rampup_epochs - ramp_start_epoch
        # steps_into_ramp = epoch - ramp_start_epoch
        # progress_in_ramp = steps_into_ramp / total_ramp_duration
        progress_in_ramp = (epoch - ramp_start_epoch) / (self.consistency_rampup_epochs - ramp_start_epoch)
        
        # Clamp progress to [0, 1] to handle edge cases (e.g., epoch >= self.consistency_rampup_epochs)
        progress_in_ramp = tf.minimum(tf.maximum(progress_in_ramp, 0.0), 1.0)

        current_decay = self.ema_stabilization_decay + progress_in_ramp * (self.ema_decay_end - self.ema_stabilization_decay)
        return current_decay

    def update_teacher_model(self, epoch):
        """Update teacher model weights using adaptive EMA or direct copy during warmup"""
        # --- Comment: Teacher Model Update ---
        # This is a critical step in Mean Teacher.
        # - During `warmup_epochs`: Teacher weights are a direct copy of the student's. This allows
        #   the teacher to quickly reflect the student's initial learning from supervised signals,
        #   leveraging the strong supervised baseline.
        # - After warmup: Teacher weights are updated using an Exponential Moving Average (EMA)
        #   of the student's weights, with the decay rate determined by `_get_ema_decay(epoch)`.
        #   The EMA makes the teacher a more stable, time-averaged version of the student.
        # Strategies for Improvement:
        # - Ensure `warmup_epochs` is sufficient for initial student stability.
        # - The EMA decay schedule (`_get_ema_decay`) is complex; monitor its behavior.
        #   Simpler schedules (e.g., constant high decay after warmup) could be tested.
        # - Ensure student weights are valid before copying/averaging.
        # ---
        student_weights = self.model.get_weights()
        if not student_weights:
            tf.print("DEBUG Teacher Update: Student model has no weights!", output_stream=sys.stderr)
            return self._get_ema_decay(epoch) # Return based on current phase for logging consistency

        if epoch < self.warmup_epochs:
            # During warmup epochs, directly copy student weights to teacher
            tf.print(f"DEBUG Teacher Update: Epoch {epoch} (Warmup Phase) - Directly copying student weights to teacher.", output_stream=sys.stderr)
            self.teacher_model.set_weights(student_weights)
            return self._get_ema_decay(epoch) # For logging, will return self.ema_decay_start
        else:
            # After warmup, use EMA based on the new phased logic
            ema_decay = self._get_ema_decay(epoch) 
            tf.print(f"DEBUG Teacher Update: Epoch {epoch} (EMA Phase), EMA Decay: {ema_decay}", output_stream=sys.stderr)

            teacher_weights = self.teacher_model.get_weights()

            if not teacher_weights:
                tf.print("DEBUG Teacher Update: Teacher model has no weights BEFORE EMA update! Initializing from student.", output_stream=sys.stderr)
                self.teacher_model.set_weights(student_weights)
                teacher_weights = self.teacher_model.get_weights()
                if not teacher_weights: 
                     tf.print("DEBUG Teacher Update: CRITICAL - Teacher model still has no weights after attempting re-initialization.", output_stream=sys.stderr)
                     return ema_decay 

            new_weights = []
            for sw, tw in zip(student_weights, teacher_weights):
                new_weights.append(ema_decay * tw + (1 - ema_decay) * sw)
            
            self.teacher_model.set_weights(new_weights)
            return ema_decay

    def _dice_loss(self, y_true, y_pred):
        """Enhanced Dice loss with smoother handling of edge cases and debugging."""
        # --- Comment: Supervised Loss - Dice Loss ---
        # This Dice loss function is a key component of the supervised signal.
        # Its effectiveness was part of the supervised model's success (Val Dice ~0.8577).
        # Enhancements here include:
        # - Slicing `y_true` and `y_pred` to select the foreground channel (channel 1) if multiple channels exist.
        # - Resizing `y_true` to match `y_pred` spatial dimensions using nearest neighbor interpolation.
        # - Applying sigmoid to `y_pred` logits before calculation.
        # - Using a small epsilon (1e-6) for numerical stability.
        # This loss directly guides the student model on labeled data.
        # ---
        tf.print("DEBUG DICE LOSS: Entry. y_true shape:", tf.shape(y_true), "y_pred (logits) shape:", tf.shape(y_pred), output_stream=sys.stderr)

        # --- Slice y_true if it has multiple channels ---
        if tf.shape(y_true)[-1] > 1:
            y_true_sliced = y_true[..., 1:2] # Select channel 1 (foreground ground truth)
        else:
            y_true_sliced = y_true # Assuming single channel ground truth
        tf.print("DEBUG DICE LOSS: y_true_sliced shape:", tf.shape(y_true_sliced), "min/max/mean:", tf.reduce_min(tf.cast(y_true_sliced, tf.float32)), tf.reduce_max(tf.cast(y_true_sliced, tf.float32)), tf.reduce_mean(tf.cast(y_true_sliced, tf.float32)), output_stream=sys.stderr)

        # --- Determine target size from y_pred ---
        # Ensure y_pred_for_shape is defined before accessing its shape for target_size
        if tf.shape(y_pred)[-1] > 1:
             y_pred_for_shape = y_pred[..., 1:2] 
        else:
            y_pred_for_shape = y_pred
        target_size = tf.shape(y_pred_for_shape)[1:3] # Get H, W from prediction

        # --- Resize y_true to match y_pred spatial dimensions ---
        y_true_resized = tf.image.resize(y_true_sliced, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.print("DEBUG DICE LOSS: y_true_resized shape:", tf.shape(y_true_resized), "min/max/mean:", tf.reduce_min(tf.cast(y_true_resized, tf.float32)), tf.reduce_max(tf.cast(y_true_resized, tf.float32)), tf.reduce_mean(tf.cast(y_true_resized, tf.float32)), output_stream=sys.stderr)

        # --- Slice y_pred if it has multiple channels ---
        if tf.shape(y_pred)[-1] > 1:
             y_pred_sliced = y_pred[..., 1:2] # Select channel 1 (FOREGROUND logit)
        else:
            y_pred_sliced = y_pred # Assuming single channel logits
        tf.print("DEBUG DICE LOSS: y_pred_sliced (logits) shape:", tf.shape(y_pred_sliced), "min/max/mean:", tf.reduce_min(y_pred_sliced), tf.reduce_max(y_pred_sliced), tf.reduce_mean(y_pred_sliced), output_stream=sys.stderr)

        # Apply sigmoid to logits
        y_pred_sigmoid = tf.sigmoid(y_pred_sliced)
        tf.print("DEBUG DICE LOSS: y_pred_sigmoid shape:", tf.shape(y_pred_sigmoid), "min/max/mean:", tf.reduce_min(y_pred_sigmoid), tf.reduce_max(y_pred_sigmoid), tf.reduce_mean(y_pred_sigmoid), output_stream=sys.stderr)

        # Cast to float32 for stable calculation
        y_true_f32 = tf.cast(y_true_resized, tf.float32)
        y_pred_sigmoid_f32 = tf.cast(y_pred_sigmoid, tf.float32)

        # Flatten the predictions and targets
        y_true_flat = tf.reshape(y_true_f32, [-1])
        y_pred_flat = tf.reshape(y_pred_sigmoid_f32, [-1])

        # Calculate Dice score components with epsilon for numerical stability
        smooth = 1e-6 # Define smooth factor
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        sum_true = tf.reduce_sum(y_true_flat)
        sum_pred = tf.reduce_sum(y_pred_flat)
        
        numerator = 2.0 * intersection + smooth
        denominator = sum_true + sum_pred + smooth
        tf.print("DEBUG DICE LOSS: intersection_val:", intersection, "sum_true:", sum_true, "sum_pred:", sum_pred, "numerator:", numerator, "denominator:", denominator, output_stream=sys.stderr)
        
        # Return Dice loss (1 - Dice coefficient)
        loss = 1.0 - (numerator / denominator)
        tf.print("DEBUG DICE LOSS: calculated loss:", loss, output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(loss, 0.0, message="Dice loss is negative.")
        tf.debugging.assert_less_equal(loss, 1.00001, message="Dice loss is greater than 1 (allowing for float precision).")

        policy = tf.keras.mixed_precision.global_policy()
        return tf.cast(loss, policy.compute_dtype)

    def _weighted_bce_loss(self, y_true, y_pred):
        """Weighted binary cross-entropy with focus on boundary regions and debugging."""
        # --- Comment: Supervised Loss - Weighted BCE Loss ---
        # This complements the Dice loss for the supervised signal.
        # BCE can provide smoother gradients, especially early in training.
        # Key aspects:
        # - Slicing `y_true` and `y_pred` for the foreground channel.
        # - Resizing `y_true` to match `y_pred` spatial dimensions.
        # - Applying sigmoid to `y_pred` logits.
        # - Applying a `pos_weight` (e.g., 4.0) to give more importance to positive (foreground) pixels,
        #   which can be helpful in imbalanced segmentation tasks.
        # The combination of Dice and BCE (e.g., 0.8*Dice + 0.2*BCE in `train_step`)
        # leverages the strengths of both and contributed to the supervised model's success.
        # ---
        tf.print("DEBUG BCE LOSS: Entry. y_true shape:", tf.shape(y_true), "y_pred (logits) shape:", tf.shape(y_pred), output_stream=sys.stderr)

        # --- Slice y_true and y_pred FIRST ---
        if tf.shape(y_true)[-1] > 1:
            y_true_sliced = y_true[..., 1:2] 
        else:
            y_true_sliced = y_true
        tf.print("DEBUG BCE LOSS: y_true_sliced shape:", tf.shape(y_true_sliced), "min/max/mean:", tf.reduce_min(tf.cast(y_true_sliced, tf.float32)), tf.reduce_max(tf.cast(y_true_sliced, tf.float32)), tf.reduce_mean(tf.cast(y_true_sliced, tf.float32)), output_stream=sys.stderr)

        if tf.shape(y_pred)[-1] > 1:
            y_pred_sliced = y_pred[..., 1:2] 
        else:
            y_pred_sliced = y_pred
        tf.print("DEBUG BCE LOSS: y_pred_sliced (logits) shape:", tf.shape(y_pred_sliced), "min/max/mean:", tf.reduce_min(y_pred_sliced), tf.reduce_max(y_pred_sliced), tf.reduce_mean(y_pred_sliced), output_stream=sys.stderr)

        # --- Resize y_true to match y_pred spatial dimensions ---
        target_size = tf.shape(y_pred_sliced)[1:3] # Get H, W from y_pred_sliced
        y_true_resized = tf.image.resize(y_true_sliced, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.print("DEBUG BCE LOSS: y_true_resized shape:", tf.shape(y_true_resized), "min/max/mean:", tf.reduce_min(tf.cast(y_true_resized, tf.float32)), tf.reduce_max(tf.cast(y_true_resized, tf.float32)), tf.reduce_mean(tf.cast(y_true_resized, tf.float32)), output_stream=sys.stderr)

        # Cast to float32 for calculation stability
        y_true_f32 = tf.cast(y_true_resized, tf.float32)
        y_pred_f32 = tf.cast(y_pred_sliced, tf.float32) 

        tf.debugging.assert_greater_equal(y_true_f32, 0.0, message="y_true_f32 for BCE has values less than 0")
        tf.debugging.assert_less_equal(y_true_f32, 1.0, message="y_true_f32 for BCE has values greater than 1")

        # Apply sigmoid activation (logits -> probabilities)
        y_pred_sigmoid_f32 = tf.sigmoid(y_pred_f32)
        tf.print("DEBUG BCE LOSS: y_pred_sigmoid_f32 shape:", tf.shape(y_pred_sigmoid_f32), "min/max/mean:", tf.reduce_min(y_pred_sigmoid_f32), tf.reduce_max(y_pred_sigmoid_f32), tf.reduce_mean(y_pred_sigmoid_f32), output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(y_pred_sigmoid_f32, 0.0, message="y_pred_sigmoid_f32 for BCE has values less than 0")
        tf.debugging.assert_less_equal(y_pred_sigmoid_f32, 1.0, message="y_pred_sigmoid_f32 for BCE has values greater than 1")

        # Calculate standard BCE loss
        bce = tf.keras.losses.binary_crossentropy(y_true_f32, y_pred_sigmoid_f32, from_logits=False)
        tf.print("DEBUG BCE LOSS: raw bce per-pixel min/max/mean:", tf.reduce_min(bce), tf.reduce_max(bce), tf.reduce_mean(bce), output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(bce, 0.0, message="Raw BCE per-pixel is negative")

        # Apply weight to positive examples
        pos_weight = 4.0 
        weight_map = y_true_f32 * (pos_weight - 1.0) + 1.0
        weight_map_squeezed = tf.squeeze(weight_map, axis=-1)

        weighted_bce = bce * weight_map_squeezed
        tf.print("DEBUG BCE LOSS: weighted_bce per-pixel min/max/mean:", tf.reduce_min(weighted_bce), tf.reduce_max(weighted_bce), tf.reduce_mean(weighted_bce), output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(weighted_bce, 0.0, message="Weighted BCE per-pixel is negative")

        loss_f32 = tf.reduce_mean(weighted_bce)
        tf.print("DEBUG BCE LOSS: final mean loss_f32:", loss_f32, output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(loss_f32, 0.0, message="Final BCE loss_f32 is negative")
        
        policy = tf.keras.mixed_precision.global_policy()
        return tf.cast(loss_f32, policy.compute_dtype)
    
    def _consistency_loss(self, student_logits, teacher_logits):
        """Advanced consistency loss combining MSE and structural similarity"""
        # --- Comment: Mean Teacher - Consistency Loss ---
        # This loss measures the difference between student and teacher predictions on unlabeled data.
        # It encourages the student to produce outputs consistent with the more stable teacher.
        # This implementation uses:
        # 1. Mean Squared Error (MSE) on probabilities (after sigmoid).
        # 2. Gradient Consistency: Penalizes differences in image gradients (approximated by Sobel-like finite differences)
        #    of the student and teacher probability maps. This encourages structural similarity in predictions.
        # The losses are combined with weighting (0.7 * MSE + 0.3 * GradLoss).
        # Strategies for Improvement:
        # - Input for Teacher: Ensure `teacher_logits` come from the teacher model processing an
        #   appropriately augmented version of the unlabeled input that the student sees. The `train_step`
        #   uses `_apply_augmentation(unlabeled_images, is_teacher=True)`.
        # - Alternative Metrics: KL Divergence could be an alternative to MSE.
        # - Weighting: The 0.7/0.3 balance between MSE and gradient loss can be tuned.
        # - Gradient Calculation: The Sobel approximation is simple; more advanced gradient/feature
        #   matching could be explored but adds complexity.
        # ---
        # Apply sigmoid
        student_probs = tf.sigmoid(student_logits)
        teacher_probs = tf.sigmoid(teacher_logits)

        # --- Add tf.print for debugging shapes before potential resize ---
        tf.print("Consistency Loss Input: student_probs shape:", tf.shape(student_probs), "teacher_probs shape:", tf.shape(teacher_probs), output_stream=sys.stderr)

        # Resize student_probs to match teacher_probs if spatial dimensions differ
        # Based on error, student_probs might be [B, 128, 128, C] and teacher_probs [B, 256, 256, C]
        if tf.shape(student_probs)[1] != tf.shape(teacher_probs)[1] or \
           tf.shape(student_probs)[2] != tf.shape(teacher_probs)[2]:
            tf.print("Consistency Loss: Resizing student_probs from", tf.shape(student_probs)[1:3], 
                     "to match teacher_probs dimensions", tf.shape(teacher_probs)[1:3], output_stream=sys.stderr)
            student_probs = tf.image.resize(student_probs, 
                                            tf.shape(teacher_probs)[1:3], # target H, W from teacher_probs
                                            method=tf.image.ResizeMethod.BILINEAR)
            tf.print("Consistency Loss: student_probs shape after resize:", tf.shape(student_probs), output_stream=sys.stderr)
        
        # Mean squared error component
        mse_loss = tf.reduce_mean(tf.square(student_probs - teacher_probs))
        
        # Also penalize differences in higher-order structure via gradient differences
        # This encourages prediction of similar edges/structures
        def _image_gradients(images):
            # Calculate gradients using Sobel operator approximation
            batch_size = tf.shape(images)[0]
            dy = images[:, 1:, :, :] - images[:, :-1, :, :]
            dx = images[:, :, 1:, :] - images[:, :, :-1, :]
            # Zero-pad the gradients to match original size
            shape = tf.shape(images)
            dy = tf.pad(dy, [[0, 0], [0, 1], [0, 0], [0, 0]])
            dx = tf.pad(dx, [[0, 0], [0, 0], [0, 1], [0, 0]])
            return dx, dy
        
        # Get gradients of both prediction maps
        student_dx, student_dy = _image_gradients(student_probs)
        teacher_dx, teacher_dy = _image_gradients(teacher_probs)
        
        # Calculate gradient consistency loss
        grad_dx_loss = tf.reduce_mean(tf.abs(student_dx - teacher_dx))
        grad_dy_loss = tf.reduce_mean(tf.abs(student_dy - teacher_dy))
        grad_loss = grad_dx_loss + grad_dy_loss
        
        # Combine losses with weighting
        combined_loss = 0.7 * mse_loss + 0.3 * grad_loss
        
        return combined_loss
    
    def get_consistency_weight(self, epoch):
        """Get ramped consistency weight, delayed until after EMA stabilization phase"""
        # epoch is 0-indexed
        
        # --- Comment: Mean Teacher - Consistency Weight Ramp-up ---
        # This function controls the influence of the `_consistency_loss`.
        # The weight is kept at 0.0 during:
        #   - Student warmup (`self.warmup_epochs`).
        #   - Teacher EMA stabilization (`self.ema_stabilization_duration`).
        # After these phases, the weight ramps linearly from 0.0 to `self.consistency_weight_max`
        # by `self.consistency_rampup_epochs`.
        # This delayed ramp-up is crucial:
        #   - It allows the student to learn from supervised signals first.
        #   - It ensures the teacher model is stable (due to EMA stabilization) before its
        #     predictions are used as targets for the consistency loss.
        # Strategies for Improvement:
        # - Timing: Ensure `consistency_start_epoch` aligns with a point where the teacher
        #   is reasonably accurate.
        # - Ramp Shape: Linear ramp-up is common. Sigmoid or exponential ramps are alternatives.
        # - Max Weight: `self.consistency_weight_max` needs tuning based on the dataset,
        #   amount of labeled data, and reliability of teacher predictions.
        # ---
        # Consistency loss starts ramping up *after* the EMA stabilization phase.
        # self.warmup_epochs (e.g., 5 for epochs 0-4: direct copy)
        # self.ema_stabilization_duration (e.g., 10 for epochs 5-14: fixed high EMA decay)
        consistency_start_epoch = self.warmup_epochs + self.ema_stabilization_duration # e.g., 15
        
        if epoch < consistency_start_epoch: # e.g., epochs 0-14
            return 0.0
            
        # Linear ramp up for consistency weight
        # Ramp from consistency_start_epoch (e.g. 15) to self.consistency_rampup_epochs (e.g. 80)
        # Total duration of this ramp: self.consistency_rampup_epochs - consistency_start_epoch
        # Current step into this ramp: epoch - consistency_start_epoch
        
        ramp_duration = self.consistency_rampup_epochs - consistency_start_epoch
        
        if ramp_duration <= 0: # Should not happen with reasonable settings
            # If ramp duration is invalid, return max weight if current epoch is at or beyond ramp_end
            return self.consistency_weight_max if epoch >= self.consistency_rampup_epochs else 0.0

        current_progress = (epoch - consistency_start_epoch) / ramp_duration
        # Clamp progress to [0, 1] to handle edge cases (e.g., epoch >= self.consistency_rampup_epochs)
        current_progress = tf.minimum(tf.maximum(current_progress, 0.0), 1.0)

        return self.consistency_weight_max * current_progress
        
    def _apply_augmentation(self, image, is_teacher=False):
        """Apply advanced augmentations with TensorFlow operations"""
        # --- Comment: Data Augmentation ---
        # Augmentation is applied differently to student and teacher inputs for unlabeled data.
        # - Student: Receives stronger augmentation (controlled by `strength = 0.5`).
        #   Includes random brightness, contrast, and Gaussian noise.
        # - Teacher: Receives weaker augmentation (`strength = 0.2`) or potentially just basic
        #   preprocessing if `is_teacher=True` leads to fewer ops. The current code applies
        #   noise with reduced strength but not brightness/contrast if `is_teacher=True`.
        # This differential augmentation is a common strategy: the student should be robust
        # to various perturbations, while the teacher provides a more stable target.
        # All images are resized to `(config.img_size_y, config.img_size_x)`.
        # Strategies for Improvement:
        # - Augmentation Policy: Experiment with different/more advanced augmentations (e.g., elastic
        #   deformations, cutout, mixup) for the student.
        # - Teacher Augmentation: Some Mean Teacher variants use no augmentation or only very
        #   weak augmentation (e.g., flip/crop) for the teacher on unlabeled data. The goal is
        #   a stable, reliable target. The current setup applies some noise.
        # - Ensure augmentations are suitable for medical images and don't distort crucial features.
        # ---
        # --- Add tf.print for debugging ---
        tf.print("Augmentation input shape:", tf.shape(image), "dtype:", image.dtype, "rank:", tf.rank(image), output_stream=sys.stderr)

        # --- Remove internal rank check, rely on check in train loop ---
        # if tf.rank(image) == 3:
        #     tf.print("Augmentation: Adding batch dimension", output_stream=sys.stderr)
        #     image = tf.expand_dims(image, axis=0)

        # --- Ensure float32 dtype ---
        image = tf.cast(image, tf.float32)
        tf.print("Shape before resize:", tf.shape(image), "dtype:", image.dtype, output_stream=sys.stderr)

        # --- Explicitly cast size to tf.int32 using (height, width) ---
        target_size = tf.cast([self.config.img_size_y, self.config.img_size_x], dtype=tf.int32)

        # --- Resize ---
        try:
            image = tf.image.resize(image, target_size)
        except Exception as e:
            tf.print("ERROR during tf.image.resize:", e, output_stream=sys.stderr)
            # Re-raise the exception to halt execution if resize fails
            raise e
        # --- End resize ---

        tf.print("Shape after resize:", tf.shape(image), output_stream=sys.stderr)

        batch_size = tf.shape(image)[0]

        # Different augmentation strength for student and teacher
        strength = 0.5 if not is_teacher else 0.2

        # Random brightness adjustment
        if not is_teacher:
            image = tf.image.random_brightness(image, max_delta=0.1 * strength)

        # Random contrast adjustment
        if not is_teacher:
            image = tf.image.random_contrast(image, lower=1.0-0.2*strength, upper=1.0+0.2*strength)

        # Random Gaussian noise
        noise_stddev = 0.02 * strength
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev, dtype=image.dtype)
        image = image + noise

        # Make sure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
        
    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images, consistency_weight, epoch):
        """Execute one training step with different augmentations for student and teacher"""
        # --- Comment: Mean Teacher Training Step ---
        # This is the core of the training process, combining supervised and unsupervised learning.
        # 1. Augmentation:
        #    - `labeled_images_student`: Labeled images with student augmentation.
        #    - `unlabeled_images_student`: Unlabeled images with student augmentation.
        #    - `unlabeled_images_teacher`: Same unlabeled images but with teacher-specific (weaker) augmentation.
        # 2. Supervised Path (Student Model):
        #    - `labeled_logits = self.model(labeled_images_student, training=True)`
        #    - `supervised_loss = 0.8 * _dice_loss + 0.2 * _weighted_bce_loss`
        #      This leverages the strong, proven supervised loss components.
        # 3. Unsupervised Path (Consistency):
        #    - `unlabeled_student_logits = self.model(unlabeled_images_student, training=True)`
        #    - `unlabeled_teacher_logits = self.teacher_model(unlabeled_images_teacher, training=False)`
        #      (Teacher model in inference mode).
        #    - `consistency_loss = _consistency_loss(unlabeled_student_logits, unlabeled_teacher_logits)`
        # 4. Total Loss:
        #    - `total_loss = supervised_loss + consistency_weight * consistency_loss`
        #      The `consistency_weight` is ramped up according to `get_consistency_weight(epoch)`.
        # 5. Optimization: Gradients from `total_loss` update the student model (`self.model`).
        #    The teacher model (`self.teacher_model`) is updated via EMA in `update_teacher_model` (called in `train` loop).
        #
        # Strategies for Improvement (building on successful supervised model):
        # - Augmentation Strategy: The quality and nature of augmentations for student vs. teacher
        #   are paramount. The student needs diverse, strong augmentations. The teacher needs
        #   inputs that allow it to make stable, reliable predictions.
        # - Consistency Loss Formulation: The `_consistency_loss` (MSE + GradDiff) can be tuned or
        #   replaced (e.g., with KL divergence, or simpler MSE if GradDiff is unstable).
        # - Loss Weighting: The balance between `supervised_loss` and `consistency_loss` (via
        #   `consistency_weight`) is critical. If labeled data is scarce, higher consistency weight
        #   might be needed, but only if teacher predictions are reliable.
        # - Teacher Stability: The EMA update (`update_teacher_model`) and its schedule are key to
        #   teacher stability. A noisy teacher can harm student learning.
        # - Input Validation: Assertions for tensor ranks and shapes are good for debugging.
        # ---
        # --- Assertion removed, check moved to train loop ---

        with tf.GradientTape() as tape:
            # Apply augmentations specific to student
            labeled_images_student = self._apply_augmentation(labeled_images, is_teacher=False)
            unlabeled_images_student = self._apply_augmentation(unlabeled_images, is_teacher=False)

            # Apply different augmentations for teacher (weaker)
            unlabeled_images_teacher = self._apply_augmentation(unlabeled_images, is_teacher=True)

            # --- Add assertions AFTER augmentation ---
            tf.debugging.assert_rank(labeled_images_student, 4, message="labeled_images_student lost batch dim after augmentation!")
            tf.debugging.assert_rank(unlabeled_images_student, 4, message="unlabeled_images_student lost batch dim after augmentation!")
            tf.debugging.assert_rank(unlabeled_images_teacher, 4, message="unlabeled_images_teacher lost batch dim after augmentation!")

            # --- Check spatial dimensions AFTER augmentation (using config values) ---
            expected_spatial_shape = tf.constant([self.config.img_size_y, self.config.img_size_x], dtype=tf.int32)
            tf.debugging.assert_equal(tf.shape(unlabeled_images_student)[1:3], expected_spatial_shape,
                                      message=f"unlabeled_images_student has wrong spatial size after augmentation! Expected {self.config.img_size_y}x{self.config.img_size_x}")


            # Forward pass through student model for labeled data
            labeled_logits = self.model(labeled_images_student, training=True) # Shape [B, H, W, 1] or [B, H, W, C]? Assume 1
            tf.print("DEBUG train_step: Shape of labeled_logits (student):", tf.shape(labeled_logits), output_stream=sys.stderr)

            # Calculate supervised losses - combine Dice and BCE for better performance
            # Pass original labels [B, H, W, 2] and logits [B, H, W, 1]
            dice_loss = self._dice_loss(labeled_labels, labeled_logits) # Returns policy.compute_dtype
            bce_loss = self._weighted_bce_loss(labeled_labels, labeled_logits) # Returns policy.compute_dtype

            # Weighted combination of losses
            supervised_loss = 0.5 * dice_loss + 0.5 * bce_loss # Potentially float16 if mixed precision is on

            # Forward pass for unlabeled data through student model
            unlabeled_student_logits = self.model(unlabeled_images_student, training=True)
            tf.print("DEBUG train_step: Shape of unlabeled_student_logits (student):", tf.shape(unlabeled_student_logits), output_stream=sys.stderr)

            # Forward pass for differently augmented unlabeled data through teacher model
            unlabeled_teacher_logits = self.teacher_model(unlabeled_images_teacher, training=False)
            tf.print("DEBUG train_step: Shape of unlabeled_teacher_logits (teacher):", tf.shape(unlabeled_teacher_logits), output_stream=sys.stderr)

            # Calculate consistency loss
            consistency_loss = self._consistency_loss(unlabeled_student_logits, unlabeled_teacher_logits) # Assumed to be float32

            # --- Explicitly cast loss components to float32 before combining, as per README findings ---
            supervised_loss_f32 = tf.cast(supervised_loss, tf.float32)
            consistency_loss_f32 = tf.cast(consistency_loss, tf.float32) # Ensure consistency_loss is float32
            
            # consistency_weight is already float32 from get_consistency_weight()
            total_loss = supervised_loss_f32 + consistency_weight * consistency_loss_f32

            # --- Apply loss scaling if using mixed precision ---
            # Check if optimizer is LossScaleOptimizer before calling get_scaled_loss
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss # Ensure scaled_loss is assigned

        # Get gradients and update student model
        # --- Use scaled loss for gradients ---
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        # --- Unscale gradients ---
        # Check if optimizer is LossScaleOptimizer before calling get_unscaled_gradients
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients # Ensure gradients is assigned

        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 2.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Calculate dice score for monitoring (using original labels and logits)
        dice_score = 1.0 - self._dice_loss(labeled_labels, labeled_logits) # Re-calculate using original dice loss logic

        return {
            'total_loss': total_loss,
            'dice_score': dice_score,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss
        }

    def _dice_metric(self, y_true, y_pred, model_name=""): # Added model_name argument
        """Dice metric for validation with debugging and correct return type."""
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): Entry. y_true shape:", tf.shape(y_true), "y_pred (logits) shape:", tf.shape(y_pred), output_stream=sys.stderr)

        y_true_casted = tf.cast(y_true, tf.float32)
        tf.debugging.assert_greater_equal(y_true_casted, 0.0, message="y_true for metric has values < 0 before processing")
        tf.debugging.assert_less_equal(y_true_casted, 1.0, message="y_true for metric has values > 1 before processing (should be 0 or 1)")

        if tf.shape(y_true_casted)[-1] > 1:
            y_true_processed = y_true_casted[..., 1:2]
        else:
            y_true_processed = y_true_casted
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): y_true_processed (after potential slice) shape:", tf.shape(y_true_processed), "min/max/mean:", tf.reduce_min(y_true_processed), tf.reduce_max(y_true_processed), tf.reduce_mean(y_true_processed), output_stream=sys.stderr)

        if tf.shape(y_pred)[-1] > 1:
            y_pred_logits_sliced = y_pred[..., 1:2]
        else:
            y_pred_logits_sliced = y_pred
        
        y_pred_probs = tf.sigmoid(y_pred_logits_sliced)
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): y_pred_probs shape:", tf.shape(y_pred_probs), "min/max/mean:", tf.reduce_min(y_pred_probs), tf.reduce_max(y_pred_probs), tf.reduce_mean(y_pred_probs), output_stream=sys.stderr)

        y_pred_mask = tf.cast(y_pred_probs > 0.5, tf.float32)
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): y_pred_mask shape:", tf.shape(y_pred_mask), "min/max/mean:", tf.reduce_min(y_pred_mask), tf.reduce_max(y_pred_mask), tf.reduce_mean(y_pred_mask), output_stream=sys.stderr)

        pred_h = tf.shape(y_pred_mask)[1]
        pred_w = tf.shape(y_pred_mask)[2]
        
        y_true_resized = y_true_processed 
        if tf.shape(y_true_processed)[1] != pred_h or tf.shape(y_true_processed)[2] != pred_w:
            # Updated tf.print to include model_name
            tf.print(f"DEBUG DICE METRIC ({model_name}): Resizing y_true_processed from {tf.shape(y_true_processed)[1:3]} to {[pred_h, pred_w]} for metric", output_stream=sys.stderr)
            y_true_resized = tf.image.resize(y_true_processed, [pred_h, pred_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): y_true_resized (for metric) shape:", tf.shape(y_true_resized), "min/max/mean:", tf.reduce_min(y_true_resized), tf.reduce_max(y_true_resized), tf.reduce_mean(y_true_resized), output_stream=sys.stderr)
        tf.debugging.assert_greater_equal(y_true_resized, 0.0, message="y_true_resized for metric has values < 0")
        tf.debugging.assert_less_equal(y_true_resized, 1.0, message="y_true_resized for metric has values > 1")

        y_true_flat_per_image = tf.reshape(y_true_resized, [tf.shape(y_true_resized)[0], -1])
        y_pred_flat_per_image = tf.reshape(y_pred_mask, [tf.shape(y_pred_mask)[0], -1])

        intersection_per_image = tf.reduce_sum(y_true_flat_per_image * y_pred_flat_per_image, axis=1)
        sum_true_per_image = tf.reduce_sum(y_true_flat_per_image, axis=1)
        sum_pred_per_image = tf.reduce_sum(y_pred_flat_per_image, axis=1)
        epsilon_metric = 1e-6

        def standard_dice_calc(inter, s_true, s_pred):
            return (2.0 * inter + epsilon_metric) / (s_true + s_pred + epsilon_metric)

        num_images_in_batch_tf = tf.shape(y_true_resized)[0]
        dice_scores_list = []
        for i in tf.range(num_images_in_batch_tf):
            inter = intersection_per_image[i]
            s_true = sum_true_per_image[i]
            s_pred = sum_pred_per_image[i]
            
            # Updated tf.print to include model_name
            tf.print(f"DEBUG DICE METRIC ({model_name}) (Image ", i, " of ", num_images_in_batch_tf, "): s_true:", s_true, "s_pred:", s_pred, "inter:", inter, output_stream=sys.stderr)
            
            current_image_dice = tf.cond(
                tf.equal(s_true, 0.0),
                lambda: tf.cond(tf.equal(s_pred, 0.0),
                                lambda: tf.constant(1.0, dtype=tf.float32),
                                lambda: tf.constant(0.0, dtype=tf.float32)),
                lambda: standard_dice_calc(inter, s_true, s_pred)
            )
            dice_scores_list.append(current_image_dice)
        
        if not dice_scores_list: 
            dice_scores_per_image_tensor = tf.constant([], dtype=tf.float32)
        else:
            dice_scores_per_image_tensor = tf.stack(dice_scores_list)
        
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): dice_scores_per_image_tensor:", dice_scores_per_image_tensor, " (shape:", tf.shape(dice_scores_per_image_tensor), ")", output_stream=sys.stderr)

        valid_mask_per_image = tf.greater(sum_true_per_image, 0.0)
        dice_scores_for_valid_gt_images = tf.boolean_mask(dice_scores_per_image_tensor, valid_mask_per_image)
        num_valid_images_in_batch = tf.reduce_sum(tf.cast(valid_mask_per_image, tf.float32))

        mean_dice_for_reporting = tf.cond(
            tf.greater(num_valid_images_in_batch, 0.0),
            lambda: tf.reduce_mean(dice_scores_for_valid_gt_images),
            lambda: 0.0
        )
        # Updated tf.print to include model_name
        tf.print(f"DEBUG DICE METRIC ({model_name}): mean_dice_for_reporting (avg over valid GT images):", mean_dice_for_reporting, output_stream=sys.stderr)
        tf.print(f"DEBUG DICE METRIC ({model_name}): num_valid_images_in_batch (those with GT > 0):", num_valid_images_in_batch, output_stream=sys.stderr)
        
        tf.debugging.assert_greater_equal(mean_dice_for_reporting, 0.0, message="Mean Dice score for reporting (valid GT images) is negative.")
        tf.debugging.assert_less_equal(mean_dice_for_reporting, 1.00001, message="Mean Dice score for reporting (valid GT images) is > 1.")

        return {
            'dice': mean_dice_for_reporting,
            'valid_images_count': num_valid_images_in_batch
        }

    def validate(self, val_ds):
        # Determine num_val_steps regardless of tqdm usage first
        val_cardinality = tf.data.experimental.cardinality(val_ds)
        if val_cardinality == tf.data.experimental.INFINITE_CARDINALITY or val_cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
            tf.print("Warning: Validation dataset cardinality is unknown or infinite. Using fallback validation steps.", output_stream=sys.stderr)
            num_val_steps = getattr(self.config, 'validation_steps_fallback', 100)
        else:
            num_val_steps = val_cardinality.numpy()

        if self.use_tqdm: 
            val_progbar = tqdm(total=num_val_steps, desc="Validating", unit="batch", disable=not self.use_tqdm)

        student_batch_mean_dices = [] 
        teacher_batch_mean_dices = [] 
        
        val_iter = iter(val_ds)

        for step in range(num_val_steps): 
            try:
                images, labels = next(val_iter)
                # Added print to inspect raw labels from the validation dataset
                tf.print("DEBUG VALIDATE: Raw labels from val_ds - shape:", tf.shape(labels),
                         "dtype:", labels.dtype,
                         "min:", tf.reduce_min(labels), # Assuming labels are numeric
                         "max:", tf.reduce_max(labels), # Assuming labels are numeric
                         "mean:", tf.reduce_mean(tf.cast(labels, tf.float32)), # Cast to float for mean
                         output_stream=sys.stderr)
            except StopIteration:
                tf.print("Validation data exhausted before completing all num_val_steps.", output_stream=sys.stderr)
                break 

            if step == 0 and getattr(self.config, 'debug_validation_batch_0', False):
                tf.print("\n--- Validate Function Debug (Batch 0) ---", output_stream=sys.stderr)
                tf.print("Raw labels shape:", tf.shape(labels), "dtype:", labels.dtype, "sum:", tf.reduce_sum(tf.cast(labels, tf.float32)), output_stream=sys.stderr)
                tf.print("--- End Validate Function Debug ---", output_stream=sys.stderr)

            img_h, img_w = self.config.img_size_y, self.config.img_size_x 
            images_resized = tf.image.resize(images, [img_h, img_w])
            
            labels_processed = tf.cast(labels, tf.float32)
            if self.config.num_classes == 1 and tf.shape(labels_processed)[-1] != 1: 
                if len(tf.shape(labels_processed)) == 3: 
                    labels_processed = labels_processed[..., tf.newaxis]
                elif tf.shape(labels_processed)[-1] > 1: 
                    labels_processed = labels_processed[..., :1]
            
            if getattr(self.config, 'debug_validation_processing', False):
                tf.print("Validate input: images_resized shape:", tf.shape(images_resized), "dtype:", images_resized.dtype, output_stream=sys.stderr)
                tf.print("Validate input: labels_processed shape:", tf.shape(labels_processed), "dtype:", labels_processed.dtype, output_stream=sys.stderr)

            student_pred = self.model(images_resized, training=False) 
            teacher_pred = self.teacher_model(images_resized, training=False)

            # Pass model_name to _dice_metric
            student_metrics_dict = self._dice_metric(labels_processed, student_pred, model_name="Student")
            teacher_metrics_dict = self._dice_metric(labels_processed, teacher_pred, model_name="Teacher")
            
            s_valid_count = float(student_metrics_dict['valid_images_count']) 
            if s_valid_count > 0:
                student_batch_mean_dices.append(float(student_metrics_dict['dice'])) 
            
            t_valid_count = float(teacher_metrics_dict['valid_images_count'])
            if t_valid_count > 0:
                teacher_batch_mean_dices.append(float(teacher_metrics_dict['dice']))

            if self.use_tqdm:
                val_progbar.update(1)
                postfix_update = {}
                if s_valid_count > 0:
                    postfix_update['s_dice_batch'] = student_metrics_dict['dice'].numpy() 
                if t_valid_count > 0:
                    postfix_update['t_dice_batch'] = teacher_metrics_dict['dice'].numpy()
                if postfix_update:
                    val_progbar.set_postfix(postfix_update)
        
        if self.use_tqdm:
            val_progbar.close()

        student_dice = np.mean(student_batch_mean_dices) if student_batch_mean_dices else 0.0
        teacher_dice = np.mean(teacher_batch_mean_dices) if teacher_batch_mean_dices else 0.0
        
        student_dice = float(student_dice) 
        teacher_dice = float(teacher_dice) 

        tf.print(f"Validation Student Dice: {student_dice:.4f}, Teacher Dice: {teacher_dice:.4f}", output_stream=sys.stderr)
        return student_dice, teacher_dice

    def plot_learning_curves(self, filepath=None):
        """Plot learning curves for training process"""
        if not self.history['train_loss'] or not self.history['val_dice']:
            print("Not enough data to plot learning curves.")
            return
            
        plt.figure(figsize=(20, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['supervised_loss'], label='Supervised Loss')
        plt.plot(self.history['consistency_loss'], label='Consistency Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Plot Dice scores
        plt.subplot(2, 2, 2)
        plt.plot(self.history['val_dice'], label='Student Validation Dice')
        plt.plot(self.history['teacher_dice'], label='Teacher Validation Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        # Save the figure if filepath is provided
        if filepath:
            plt.savefig(filepath)
            print(f"Learning curves saved to {filepath}")
        
        plt.tight_layout()
        plt.close()
        
    def train(self, data_paths):
        """Train with curriculum learning, adaptive EMA and advanced augmentations"""
        print("\nStarting Advanced Mean Teacher training...")
        
        # --- Comment: Main Training Loop ---
        # This loop orchestrates the epochs and steps for Mean Teacher training.
        # Building on the successful supervised model, key strategies for Mean Teacher are:
        # 1. Data Handling:
        #    - `train_labeled`: Dataset of (image, label) pairs for supervised loss.
        #    - `train_unlabeled`: Dataset of unlabeled images, repeated and iterated.
        #      Crucially, these unlabeled images are augmented differently for student and teacher
        #      within the `train_step` function via `_apply_augmentation`.
        # 2. EMA Teacher Update: `self.update_teacher_model(epoch)` is called each epoch.
        #    Its phased EMA (warmup copy -> stabilization EMA -> ramped EMA) is critical.
        # 3. Consistency Weight Scheduling: `self.get_consistency_weight(epoch)` provides the
        #    ramped weight for the consistency loss, ensuring it's applied judiciously.
        # 4. Combined Loss: The `train_step` combines supervised loss (Dice+BCE, from the
        #    strong baseline) with the consistency loss.
        # 5. Monitoring:
        #    - Supervised loss, consistency loss, total loss.
        #    - Validation Dice for both student and (more importantly) teacher models.
        #    - Learning rate, EMA decay, consistency weight.
        # 6. Checkpointing: Saves best student and teacher models based on validation Dice.
        #
        # Areas for Fine-tuning/Improvement:
        # - Augmentation policies in `_apply_augmentation`.
        # - Hyperparameters of EMA decay (`warmup_epochs`, `ema_stabilization_duration`, decay values).
        # - Hyperparameters of consistency weight (`consistency_weight_max`, `consistency_rampup_epochs`, start delay).
        # - Balance between supervised and unsupervised loss components.
        # - Optimizer choice and learning rate schedule (currently AdamW with CosineWarmup).
        # ---
        # Using a small batch size to prevent memory issues
        batch_size = self.config.batch_size
        print(f"Using batch size: {batch_size}")
        
        # Create datasets
        print("Creating datasets...")
        
        # Check if we have labeled data
        if not data_paths['labeled']['images'] or not data_paths['labeled']['labels']:
            raise ValueError("No labeled training data found!")
            
        train_labeled = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=batch_size,
            shuffle=True
        ).prefetch(tf.data.AUTOTUNE) # Add prefetch

        # Calculate actual steps_per_epoch
        # steps_per_epoch = sum(1 for _ in train_labeled) # OLD INEFFICIENT WAY
        num_labeled_samples = len(data_paths['labeled']['images'])
        steps_per_epoch = math.ceil(num_labeled_samples / batch_size)
        print(f"Number of labeled samples: {num_labeled_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Calculated steps per epoch (labeled data): {steps_per_epoch}")

        # Call setup_training_params here with correct steps_per_epoch
        self._setup_training_params(steps_per_epoch)
        
        # Check if we have unlabeled data
        has_unlabeled = False
        train_unlabeled = None
        unlabeled_iterator = None
        if 'unlabeled' in data_paths and data_paths['unlabeled']['images']:
            print(f"Found {len(data_paths['unlabeled']['images'])} unlabeled images")
            has_unlabeled = True
            train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['unlabeled']['images'],
                None,  # No labels for unlabeled data
                batch_size=batch_size,
                shuffle=True
            ).repeat().prefetch(tf.data.AUTOTUNE) # Repeat indefinitely and prefetch
            unlabeled_iterator = iter(train_unlabeled) # Create iterator
        else:
            print("WARNING: No unlabeled data found. Running in supervised mode with teacher model.")
            
        # Validation dataset
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=batch_size, # Use same batch size for consistency
            shuffle=False
        ).prefetch(tf.data.AUTOTUNE) # Add prefetch
        
        # Setup training parameters (moved setup call above)
        best_dice = 0
        patience = 20  # Increased patience for better training
        patience_counter = 0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create experiment log file
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f'mean_teacher_advanced_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Create visualization directory
        vis_dir = Path(self.config.output_dir) / 'visualizations'
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,supervised_loss,consistency_loss,student_dice,teacher_dice,learning_rate,ema_decay,consistency_weight,time\n')
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Set up metrics tracking for this epoch
            epoch_losses = []
            epoch_supervised_losses = []
            epoch_consistency_losses = []
            epoch_dice_scores = []
            
            # Get current consistency weight based on epoch
            consistency_weight = self.get_consistency_weight(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Consistency weight: {consistency_weight:.4f}")
            
            # Update teacher model with adaptive EMA decay
            ema_decay = self.update_teacher_model(epoch)
            print(f"EMA decay: {ema_decay:.6f}")
            
            # Training steps for this epoch
            num_batches = steps_per_epoch
            
            # Use progress bar for cleaner output
            progress_bar = tqdm(train_labeled.take(num_batches), total=num_batches, desc=f"Training Epoch {epoch+1}") # Iterate over labeled data
            
            for labeled_batch in progress_bar:
                labeled_images, labeled_labels = labeled_batch

                # Get unlabeled batch from iterator
                if has_unlabeled and unlabeled_iterator is not None:
                    unlabeled_batch_data = next(unlabeled_iterator) # This is (image_tensor,)
                    unlabeled_images = unlabeled_batch_data[0] # Extract the image tensor
                else:
                    # Fallback if no unlabeled data
                    unlabeled_images = labeled_images # Use labeled images as placeholder

                # --- Add check and fix for missing batch dimension ---
                if tf.rank(unlabeled_images) == 3:
                    print(f"Warning: Adding batch dimension to unlabeled_images (original shape: {tf.shape(unlabeled_images)})", file=sys.stderr) # Print to stderr
                    unlabeled_images = tf.expand_dims(unlabeled_images, axis=0)
                # --- End check ---

                # Run training step
                metrics = self.train_step(
                    labeled_images, labeled_labels, unlabeled_images,
                    consistency_weight, epoch)
                
                # Record metrics
                epoch_losses.append(metrics['total_loss'].numpy().item())
                epoch_supervised_losses.append(metrics['supervised_loss'].numpy().item())
                epoch_consistency_losses.append(metrics['consistency_loss'].numpy().item())
                epoch_dice_scores.append(metrics['dice_score'].numpy().item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['total_loss'].numpy().item():.4f}",
                    'dice': f"{metrics['dice_score'].numpy().item():.4f}",
                    'cons': f"{metrics['consistency_loss'].numpy().item():.4f}"
                })
                
                # Update teacher model every N batches for faster adaptation
            
            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation
            student_dice, teacher_dice = self.validate(val_ds)
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            mean_supervised_loss = sum(epoch_supervised_losses) / len(epoch_supervised_losses) if epoch_supervised_losses else 0
            mean_consistency_loss = sum(epoch_consistency_losses) / len(epoch_consistency_losses) if epoch_consistency_losses else 0
            mean_train_dice = sum(epoch_dice_scores) / len(epoch_dice_scores) if epoch_dice_scores else 0
            
            # Get current learning rate
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            
            # Update history with all metrics
            self.history['train_loss'].append(mean_loss)
            self.history['val_dice'].append(student_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(current_lr)
            self.history['supervised_loss'].append(mean_supervised_loss)
            self.history['consistency_loss'].append(mean_consistency_loss)
            
            # Detailed epoch summary
            print(f"Epoch {epoch+1} summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {mean_loss:.4f}")
            print(f"  Supervised Loss: {mean_supervised_loss:.4f}")
            print(f"  Consistency Loss: {mean_consistency_loss:.4f}")
            print(f"  Train Dice: {mean_train_dice:.4f}")
            print(f"  Student Val Dice: {student_dice:.4f}")
            print(f"  Teacher Val Dice: {teacher_dice:.4f}")
            print(f"  Learning Rate: {current_lr:.6e}")
            print(f"  Consistency Weight: {consistency_weight:.4f}")
            print(f"  EMA Decay: {ema_decay:.6f}")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_supervised_loss:.6f},"
                        f"{mean_consistency_loss:.6f},{student_dice:.6f},{teacher_dice:.6f},"
                        f"{current_lr:.6e},{ema_decay:.6f},{consistency_weight:.4f},{epoch_time:.2f}\n")
            
            # Save learning curve plot every 5 epochs
            if (epoch + 1) % 5 == 0:
                plot_path = vis_dir / f"learning_curves_epoch_{epoch+1}.png"
                self.plot_learning_curves(plot_path)
            
            # Save best models (both student and teacher)
            best_model_metric = teacher_dice # Prioritize teacher model for saving best
            if best_model_metric > best_dice: # Use teacher_dice for best_dice tracking
                best_dice = best_model_metric
                
                # Save teacher model
                teacher_path = checkpoint_dir / f"best_teacher_model_epoch{epoch+1}_dice{best_dice:.4f}"
                self.teacher_model.save_weights(str(teacher_path))
                
                # Save student model
                student_path = checkpoint_dir / f"best_student_model_epoch{epoch+1}_dice{student_dice:.4f}" # Save student too
                self.model.save_weights(str(student_path))
                
                print(f"✓ New best TEACHER model saved! Dice: {best_dice:.4f}")
                patience_counter = 0
                self.validation_plateau = 0
            else:
                patience_counter += 1
                print(f"No improvement based on teacher Dice. Patience: {patience_counter}/{patience}")
                
                # Check for plateau
                if len(self.history['teacher_dice']) > 5: # Check teacher dice for plateau
                    recent_dices = self.history['teacher_dice'][-5:]
                    if max(recent_dices) - min(recent_dices) < 0.005:  # Very small improvement
                        self.validation_plateau += 1
                        print(f"Possible validation plateau detected (teacher): {self.validation_plateau}")
                        
                        # If we're plateauing, try more aggressive learning rate adjustment
                        if self.validation_plateau >= 3:
                            print("Applying learning rate boost to escape plateau")
                            # Will be reset after this epoch
            
            # Early stopping with adaptive patience
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
            
            # Save intermediate model every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_teacher_epoch_{epoch+1}"
                self.teacher_model.save_weights(str(checkpoint_path))
                print(f"Saved intermediate teacher checkpoint at epoch {epoch+1}")
        
        # Save final models
        final_teacher_path = checkpoint_dir / f"final_teacher_model"
        self.teacher_model.save_weights(str(final_teacher_path))
        
        final_student_path = checkpoint_dir / f"final_student_model"
        self.model.save_weights(str(final_student_path))
        
        # Final learning curves
        final_plot_path = vis_dir / f"final_learning_curves.png" 
        self.plot_learning_curves(final_plot_path)
        
        print(f"\nTraining completed! Best validation Dice (Teacher): {best_dice:.4f}")
        return self.history

def main():
    parser = argparse.ArgumentParser()
    # --- Comment: HPC Path Configuration & Arguments ---
    # Default paths (`data_dir`, `output_dir`) are set for the HPC environment
    # `/scratch/lustre/home/mdah0000/`. These are overridden by command-line arguments.
    # The script expects preprocessed data in `args.data_dir`.
    # Results, checkpoints, and logs are saved under `args.output_dir`.
    # ---
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/preprocessed_v2')
    parser.add_argument('--output_dir', type=Path, required=False,
                      help='Path to output directory',
                      default='/scratch/lustre/home/mdah0000/smm/v14/mean_teacher_advanced_results')
    parser.add_argument('--experiment_name', type=str, default='mean_teacher_advanced',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=150,  # More epochs for better convergence
                      help='Number of epochs to train')
    parser.add_argument('--img_size', type=int, default=256,
                      help='Image size (both dimensions)')
    parser.add_argument('--num_labeled', type=int, default=30,  # Try with fewer labeled images
                      help='Number of labeled training images')
    parser.add_argument('--num_validation', type=int, default=56,
                      help='Number of validation images')
    # --- Comment: Mean Teacher Hyperparameter Arguments ---
    # While some MT params are hardcoded in `AdvancedMeanTeacherTrainer._setup_training_params`
    # (like EMA decay schedule values, consistency ramp-up start logic tied to EMA phases),
    # core training parameters (epochs, batch size, image size, num_labeled) are CLI args.
    # For deeper MT tuning, exposing more of those internal params (EMA decays, ramp timings,
    # consistency_weight_max) as CLI arguments would be beneficial for experimentation.
    # The current script uses fixed internal schedules for EMA/consistency ramp-up.
    # ---
    args = parser.parse_args()

    # Setup GPU
    setup_gpu()
    
    # Create config
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.img_size_x = args.img_size
    config.img_size_y = args.img_size
    config.n_filters = 32
    config.num_classes = 1 # Changed from 2 to 1 to match supervised setup
    config.num_channels = 1 # Assuming 1 channel input images (e.g. grayscale)
    config.output_dir = args.output_dir
    config.checkpoint_dir = f"{args.output_dir}/checkpoints"
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='mean_teacher_advanced',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=args.num_labeled, num_validation=args.num_validation)
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer with advanced functionality
    print("Creating Advanced Mean Teacher trainer...")
    trainer = AdvancedMeanTeacherTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Create experiment directory
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model with advanced options
    trainer.train(data_paths)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == '__main__': # Corrected syntax
    main()
EOF

# Make the script executable
chmod +x $WORK_DIR/run_mean_teacher_enhanced.py
echo "Made script executable"

# Run the advanced Mean Teacher training script
echo "Running Advanced Mean Teacher training..."
$PYTHON_CMD $WORK_DIR/run_mean_teacher_enhanced.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --experiment_name "mean_teacher_advanced_slurm" \
    --batch_size 2 \
    --num_epochs 150 \
    --img_size 256 \
    --num_labeled 30 \
    --num_validation 56

echo "========================================================"
echo "Advanced Mean Teacher Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="