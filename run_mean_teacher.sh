#!/bin/bash
# SLURM batch job script for Mean Teacher semi-supervised learning with enhancements

#SBATCH -p gpu                     # Use gpu partition 
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 8                       # Request 8 CPU cores for better data loading
#SBATCH --mem=64G                  # Request 64GB RAM
#SBATCH --time=12:00:00            # Time limit increased to 24 hours
#SBATCH -o meanteacher-enh-%j.out  # Output file name with job ID
#SBATCH -e meanteacher-enh-%j.err  # Error file name with job ID
#SBATCH --mail-type=END,FAIL       # Send email when job ends or fails
#SBATCH --job-name=pancreas_mt     # Add descriptive job name

# Path to your data directory - using consistent paths as other models
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mean_teacher_results"

echo "========================================================"
echo "Running Enhanced Mean Teacher Learning - $(date)"
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
mkdir -p $OUTPUT_DIR/plots 

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_AUTO_MIXED_PRECISION=1 

# Memory optimization settings - using async allocator for better memory handling
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Find Python executable - try multiple options
PYTHON_CMD=""
for cmd in python3 python /usr/bin/python3 /usr/bin/python; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        echo "Found Python: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python executable not found. Please make sure Python is installed and in your PATH."
    exit 1
fi

# Install the correct TensorFlow version if needed
$PYTHON_CMD -m pip install --user --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib psutil

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create a customized version of main.py to support our enhanced Mean Teacher
echo "Creating Enhanced Mean Teacher Python script..."
cat > $WORK_DIR/run_mean_teacher_enhanced.py << 'EOF'
#!/usr/bin/env python3
# Enhanced Mean Teacher implementation for pancreas segmentation

import tensorflow as tf
from pathlib import Path
import argparse
import time
from datetime import datetime
import numpy as np
import gc
import os # Added for environment variable check
import json # Added for metadata saving
import matplotlib.pyplot as plt # Added for plotting
from tqdm import tqdm # Added for progress bar

# Import your modules - use the existing path
import sys
sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import StableSSLTrainer # Keep base class for structure
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from main import prepare_data_paths
# --- Remove setup_gpu import ---
# from main import prepare_data_paths, setup_gpu 

print("TensorFlow version:", tf.__version__)

# --- Add custom_setup_gpu function (same as supervised) ---
def custom_setup_gpu():
    """Setup GPU with memory growth enabled and enable mixed precision"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found. Using CPU.")
            return
            
        print(f"Found {len(gpus)} GPU(s)")
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {gpu}")
            
        # Enable mixed precision if TF_ENABLE_AUTO_MIXED_PRECISION is set
        if os.environ.get('TF_ENABLE_AUTO_MIXED_PRECISION') == '1':
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision (float16) enabled.")
            except Exception as e:
                print(f"Could not enable mixed precision: {e}")
        else:
            print("Mixed precision not enabled.")

        print("GPU setup completed successfully")
        
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        print("Continuing with default GPU settings")

# Define a function to periodically collect garbage
def collect_garbage():
    """Force garbage collection to free memory"""
    gc.collect()
    try:
        # Try to get memory stats if available
        for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
            try:
                with tf.device(f'/GPU:{i}'):
                    # Create and delete a small tensor to force memory stats update
                    temp = tf.random.normal([1])
                    del temp
                print(f"Garbage collection performed on GPU:{i}")
            except Exception as e:
                pass
    except Exception as e:
        print(f"Error during memory collection: {e}")

# --- Add create_augmented_dataset function (similar to supervised) ---
def create_augmented_dataset(dataloader, image_paths, label_paths=None, batch_size=4, shuffle=True, augment=True, is_unlabeled=False):
    """Create a dataset with optional augmentation and prefetching."""
    print(f"Creating dataset: batch_size={batch_size}, shuffle={shuffle}, augment={augment}, unlabeled={is_unlabeled}")
    
    # Create the base dataset
    dataset = dataloader.create_dataset(
        image_paths,
        label_paths, # Will be None for unlabeled
        batch_size=batch_size,
        shuffle=shuffle,
        augment=False # Augmentation handled below if enabled
    )
    
    if augment:
        # Simple augmentation (can be expanded)
        def augment_custom(image, label=None): # Label is optional
            image = tf.cast(image, tf.float32)
            # --- Cast label only if it's not None ---
            if label is not None:
                label = tf.cast(label, tf.float32)
            
            # --- Apply random flip conditionally ---
            flip_cond = tf.random.uniform(()) > 0.5
            image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(image), lambda: image)
            # --- Only flip label if it exists and flip_cond is true ---
            if label is not None:
                 label = tf.cond(flip_cond, lambda: tf.image.flip_left_right(label), lambda: label)

            # Normalize image
            image = tf.clip_by_value(image, 0.0, 1.0)

            if label is not None:
                return image, label
            else:
                # For unlabeled data, map function expects only one output
                return image 
            
        # Apply augmentation
        dataset = dataset.map(
            augment_custom,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 
    
    return dataset


class EnhancedMeanTeacherTrainer(StableSSLTrainer):
    """Enhanced version of Mean Teacher trainer with memory optimization and mixed precision"""
    
    def __init__(self, config, output_dir): # Added output_dir
        super().__init__(config)
        self.config = config
        self.output_dir = Path(output_dir) # Use output_dir
        self._setup_model()
        self._setup_training_params()
        self.history = {
            'train_loss': [],
            'sup_loss': [], # Added
            'cons_loss': [], # Added
            'val_dice': [],
            'learning_rate': []
        }
        
    def _setup_model(self):
        """Setup the U-Net model with memory-efficient parameters"""
        print("Setting up enhanced U-Net model for Mean Teacher...")
        
        # Create a more memory efficient model by reducing filters
        self.config.n_filters = 32  # Match supervised settings
        
        # Create the model directly using PancreasSeg
        self.model = PancreasSeg(self.config)
        self.teacher_model = PancreasSeg(self.config)
        
        # Initialize models with dummy input
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        _ = self.model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy weights from student to teacher
        self.teacher_model.set_weights(self.model.get_weights())
        
        print(f"Model created with input shape: ({self.config.img_size_x}, {self.config.img_size_y}, {self.config.num_channels})")
        print(f"Initial filters: {self.config.n_filters}")
        
        # Print model summary to understand the memory footprint
        total_params = self.model.count_params()
        print(f"Total model parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")

    # --- Add stable _dice_loss (same as supervised) ---
    def _dice_loss(self, y_true, y_pred):
        """Memory-efficient dice loss function with float32 stability and debugging"""
        policy = tf.keras.mixed_precision.global_policy()
        y_true = tf.cast(y_true, policy.compute_dtype)
        y_pred = tf.cast(y_pred, policy.compute_dtype) # Logits

        # --- Debug Prints Start ---
        tf.print("--- _dice_loss ---")
        tf.print("y_true shape:", tf.shape(y_true), "dtype:", y_true.dtype, "min/max:", tf.reduce_min(y_true), tf.reduce_max(y_true), summarize=-1)
        tf.print("y_pred (logits) shape:", tf.shape(y_pred), "dtype:", y_pred.dtype, "min/max:", tf.reduce_min(y_pred), tf.reduce_max(y_pred), summarize=-1)
        # --- Debug Prints End ---

        y_pred_f32 = tf.cast(y_pred, tf.float32)
        y_true_f32 = tf.cast(y_true, tf.float32)

        if tf.shape(y_true_f32)[-1] > 1:
            y_true_f32 = y_true_f32[..., 1:2]
        if tf.shape(y_pred_f32)[-1] > 1:
            y_pred_f32 = y_pred_f32[..., 0:1]
        
        y_pred_sigmoid_f32 = tf.sigmoid(y_pred_f32)
        epsilon = 1e-5

        # --- Debug Prints Start ---
        tf.print("y_true_f32 shape:", tf.shape(y_true_f32), "min/max:", tf.reduce_min(y_true_f32), tf.reduce_max(y_true_f32), summarize=-1)
        tf.print("y_pred_sigmoid_f32 shape:", tf.shape(y_pred_sigmoid_f32), "min/max:", tf.reduce_min(y_pred_sigmoid_f32), tf.reduce_max(y_pred_sigmoid_f32), summarize=-1)
        # --- Debug Prints End ---

        numerator = 2.0 * tf.reduce_sum(y_true_f32 * y_pred_sigmoid_f32, axis=[1, 2, 3]) 
        denominator = tf.reduce_sum(y_true_f32, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_sigmoid_f32, axis=[1, 2, 3]) + epsilon
        
        # --- Debug Prints Start ---
        tf.print("Numerator:", numerator, summarize=-1)
        tf.print("Denominator:", denominator, summarize=-1)
        # --- Debug Prints End ---

        dice_per_sample = numerator / denominator
        dice_coef = tf.reduce_mean(dice_per_sample)
        loss = 1.0 - dice_coef

        # --- Debug Prints Start ---
        tf.print("Dice Coef:", dice_coef, "Loss:", loss, summarize=-1)
        tf.print("--- /_dice_loss ---")
        # --- Debug Prints End ---

        return tf.cast(loss, policy.compute_dtype)

    # --- Add stable _dice_metric (same as supervised) ---
    def _dice_metric(self, y_true, y_pred):
        """Memory-efficient dice metric function for evaluation with debugging"""
        # --- Debug Prints Start ---
        tf.print("--- _dice_metric ---")
        tf.print("y_true shape:", tf.shape(y_true), "dtype:", y_true.dtype, "min/max:", tf.reduce_min(y_true), tf.reduce_max(y_true), summarize=-1)
        tf.print("y_pred (logits) shape:", tf.shape(y_pred), "dtype:", y_pred.dtype, "min/max:", tf.reduce_min(y_pred), tf.reduce_max(y_pred), summarize=-1)
        # --- Debug Prints End ---

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32) # Logits
        
        if y_true.shape[-1] > 1:
            y_true = y_true[..., 1:2]
        
        if y_pred.shape[-1] == 1:
            y_pred_prob = tf.sigmoid(y_pred)
        else: 
             y_pred_prob = tf.nn.softmax(y_pred)[..., 1:2]

        # --- Debug Prints Start ---
        tf.print("y_true_f32 shape:", tf.shape(y_true), "min/max:", tf.reduce_min(y_true), tf.reduce_max(y_true), summarize=-1)
        tf.print("y_pred_prob shape:", tf.shape(y_pred_prob), "min/max:", tf.reduce_min(y_pred_prob), tf.reduce_max(y_pred_prob), summarize=-1)
        # --- Debug Prints End ---

        y_pred_binary = tf.cast(y_pred_prob > 0.5, tf.float32)
        epsilon = 1e-5
        numerator = 2.0 * tf.reduce_sum(y_true * y_pred_binary, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_binary, axis=[1, 2, 3]) + epsilon

        # --- Debug Prints Start ---
        tf.print("Numerator (metric):", numerator, summarize=-1)
        tf.print("Denominator (metric):", denominator, summarize=-1)
        dice_score = tf.reduce_mean(numerator / denominator)
        tf.print("Dice Score (metric):", dice_score, summarize=-1)
        tf.print("--- /_dice_metric ---")
        # --- Debug Prints End ---

        return dice_score

    def _setup_training_params(self):
        """Setup training parameters with improved learning rate scheduler"""
        # --- Remove OneCycleLR class definition ---
        
        # Calculate steps per epoch and total steps
        num_labeled_images = self.config.num_labeled if hasattr(self.config, 'num_labeled') else 30 # Use the actual num_labeled
        steps_per_epoch = num_labeled_images // self.config.batch_size + (1 if num_labeled_images % self.config.batch_size else 0)
        total_steps = self.config.num_epochs * steps_per_epoch # Can be used for decay_steps if needed

        # --- Use ExponentialDecay schedule ---
        initial_lr = 5e-4 # Start with the previous max_lr
        # Decay LR by 10% every 10 epochs (adjust decay_steps and decay_rate as needed)
        decay_steps = 10 * steps_per_epoch 
        decay_rate = 0.90 
        
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True # Apply decay at discrete intervals
        )
        
        # Set Mean Teacher specific parameters (can be overridden by args)
        self.ema_decay = getattr(self.config, 'ema_decay', 0.999)
        self.consistency_weight = getattr(self.config, 'consistency_weight', 100.0)
        # Keep the adjusted consistency_rampup
        self.consistency_rampup = getattr(self.config, 'consistency_rampup', 120) 
        
        print(f"Using batch size: {self.config.batch_size}")
        print(f"Steps per epoch (based on {num_labeled_images} labeled images): {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        # --- Update LR schedule print statement ---
        print(f"Learning rate schedule: ExponentialDecay from {initial_lr}, decay rate {decay_rate} every {decay_steps} steps")
        print(f"EMA decay: {self.ema_decay}")
        print(f"Consistency weight: {self.consistency_weight} (rampup over {self.consistency_rampup} steps)")
        
        # Setup optimizer with proper learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08
        )
        # --- Apply loss scaling for mixed precision ---
        if os.environ.get('TF_ENABLE_AUTO_MIXED_PRECISION') == '1':
             self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

    def update_teacher_model(self):
        """Update teacher model weights using exponential moving average (TF graph compatible)"""
        if not hasattr(self, 'ema_decay'):
            self.ema_decay = 0.999  # Default EMA decay

        # --- Use TensorFlow variable assignment ---
        student_vars = self.model.weights
        teacher_vars = self.teacher_model.weights

        # Ensure the number of weights match
        if len(student_vars) != len(teacher_vars):
             raise ValueError("Student and Teacher models have different numbers of weights.")

        for student_var, teacher_var in zip(student_vars, teacher_vars):
            # Ensure types match before assignment if necessary, though assign should handle it
            # student_var_casted = tf.cast(student_var, teacher_var.dtype)
            # new_teacher_var = self.ema_decay * teacher_var + (1 - self.ema_decay) * student_var_casted
            
            # Direct assignment using EMA formula
            new_teacher_var = self.ema_decay * teacher_var + (1 - self.ema_decay) * student_var
            teacher_var.assign(new_teacher_var)

    def _consistency_loss(self, student_logits, teacher_logits):
        """Calculate consistency loss between student and teacher model predictions"""
        # --- Cast inputs based on policy ---
        policy = tf.keras.mixed_precision.global_policy()
        student_logits = tf.cast(student_logits, policy.compute_dtype)
        teacher_logits = tf.cast(teacher_logits, policy.compute_dtype)

        # --- Perform calculations in float32 ---
        student_logits_f32 = tf.cast(student_logits, tf.float32)
        teacher_logits_f32 = tf.cast(teacher_logits, tf.float32)

        # Apply sigmoid since this is binary segmentation
        student_probs_f32 = tf.sigmoid(student_logits_f32)
        teacher_probs_f32 = tf.sigmoid(teacher_logits_f32)
        
        # Calculate MSE between predictions (in float32)
        mse_loss_f32 = tf.reduce_mean(tf.square(student_probs_f32 - teacher_probs_f32))

        # --- Cast final loss back to policy dtype ---
        return tf.cast(mse_loss_f32, policy.compute_dtype)
    
    def get_consistency_weight(self, step):
        """Get ramped consistency weight based on current step"""
        # --- Ensure step is float32 ---
        step = tf.cast(step, tf.float32) 
        
        if not hasattr(self, 'consistency_rampup'):
            return tf.cast(self.consistency_weight, tf.float32) # Ensure consistent dtype
            
        # Ramp up the consistency weight
        rampup_length = tf.cast(self.consistency_rampup, tf.float32) # Cast rampup length
        
        # --- Use tf.minimum instead of Python min ---
        rampup = tf.minimum(1.0, step / rampup_length) 
        
        return tf.cast(self.consistency_weight, tf.float32) * rampup # Ensure consistent dtype

    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images):
        """Execute one training step with both labeled and unlabeled data"""
        step = self.optimizer.iterations
        
        # --- Simple augmentation for unlabeled images (consistency target) ---
        # Consider using stronger augmentations if needed
        unlabeled_images_aug = unlabeled_images # Use original for teacher input for now
        # Example: Add noise
        # noise_stddev = 0.03 
        # unlabeled_images_aug = unlabeled_images + tf.random.normal(
        #     shape=tf.shape(unlabeled_images), mean=0.0, stddev=noise_stddev)
        
        with tf.GradientTape() as tape:
            # Labeled data -> Student
            labeled_logits = self.model(labeled_images, training=True)
            supervised_loss = self._dice_loss(labeled_labels, labeled_logits) # Likely float16
            
            # Unlabeled data -> Student
            unlabeled_student_logits = self.model(unlabeled_images, training=True)
            
            # Augmented Unlabeled data -> Teacher (no training)
            unlabeled_teacher_logits = self.teacher_model(unlabeled_images_aug, training=False)
            
            # Consistency Loss
            consistency_loss = self._consistency_loss(unlabeled_student_logits, unlabeled_teacher_logits) # Likely float16
            
            # Get current consistency weight
            cons_weight = self.get_consistency_weight(step) # Returns float32
            
            # --- Cast components to float32 before combining ---
            supervised_loss_f32 = tf.cast(supervised_loss, tf.float32)
            # --- Cast consistency_loss to float32 BEFORE multiplying ---
            consistency_loss_f32 = cons_weight * tf.cast(consistency_loss, tf.float32) 
            # cons_weight is already float32
            
            # Combined loss (in float32)
            total_loss = supervised_loss_f32 + consistency_loss_f32 
            
            # --- Apply loss scaling (operates on the float32 total_loss) ---
            scaled_total_loss = self.optimizer.get_scaled_loss(total_loss) if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else total_loss
        
        # Gradients using scaled loss
        scaled_gradients = tape.gradient(scaled_total_loss, self.model.trainable_variables)
        # --- Unscale gradients ---
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients) if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else scaled_gradients
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update teacher model (now graph compatible)
        self.update_teacher_model()
        
        # Calculate dice score for monitoring (using original supervised loss dtype)
        dice_score = 1.0 - supervised_loss 
        
        # Return losses (use original dtypes or float32 as preferred for logging)
        # --- Return the float32 version of consistency loss for logging consistency ---
        return total_loss, dice_score, supervised_loss, tf.cast(consistency_loss, tf.float32) # Return float32 consistency loss

    def validate(self, val_dataset):
        """Validate model performance on validation dataset"""
        student_dice_scores = []
        teacher_dice_scores = []
        
        # --- Ensure val_dataset is not empty and is iterable ---
        if val_dataset is None:
            tf.print("Validation dataset is None. Skipping validation.", output_stream=sys.stderr)
            return 0.0, 0.0 # Or appropriate default

        # --- Wrap in try-except to catch potential iteration issues ---
        try:
            for images, labels in val_dataset: # Iterates over batches
                # Ensure images and labels are float32 for model and metric
                images = tf.cast(images, tf.float32)
                labels = tf.cast(labels, tf.float32) # Ensure labels are float32 for metric

                student_logits = self.model(images, training=False) 
                teacher_logits = self.teacher_model(images, training=False)
                
                # --- ADD DEBUG PRINT HERE ---
                # True Foreground Sum
                y_true_for_sum = tf.cast(labels, tf.float32)
                if self.config.num_classes == 2 and tf.shape(y_true_for_sum)[-1] == 2:
                    y_true_fg_sum_per_sample = tf.reduce_sum(y_true_for_sum[..., 1], axis=[1,2])
                elif tf.shape(y_true_for_sum)[-1] == 1:
                    y_true_fg_sum_per_sample = tf.reduce_sum(tf.squeeze(y_true_for_sum, axis=-1), axis=[1,2])
                else: # Assuming y_true_for_sum is already [B,H,W]
                    y_true_fg_sum_per_sample = tf.reduce_sum(y_true_for_sum, axis=[1,2])

                # Predicted Foreground Sum (Teacher)
                y_pred_logits_teacher = tf.cast(teacher_logits, tf.float32)
                y_pred_probs_teacher_fg_channel = tf.zeros_like(y_true_fg_sum_per_sample) # Default to 0 if logic fails

                if self.config.num_classes == 2 and tf.shape(y_pred_logits_teacher)[-1] == 2:
                    y_pred_probs_teacher_fg_channel = tf.nn.softmax(y_pred_logits_teacher)[..., 1]
                elif tf.shape(y_pred_logits_teacher)[-1] == 1:
                    y_pred_probs_teacher_fg_channel = tf.sigmoid(tf.squeeze(y_pred_logits_teacher, axis=-1))
                elif len(tf.shape(y_pred_logits_teacher)) == 3: # Assuming [B,H,W] logits for foreground
                    y_pred_probs_teacher_fg_channel = tf.sigmoid(y_pred_logits_teacher)
                
                y_pred_binary_teacher = tf.cast(y_pred_probs_teacher_fg_channel > 0.5, tf.float32)
                y_pred_fg_sum_teacher_per_sample = tf.reduce_sum(y_pred_binary_teacher, axis=[1,2])

                tf.print("Validation Batch Debug (Teacher):",
                         "Batch True FG Pixel Sums (per sample):", y_true_fg_sum_per_sample,
                         "Batch Pred FG Pixel Sums (Teacher, per sample):", y_pred_fg_sum_teacher_per_sample,
                         "Labels shape:", tf.shape(labels), "Labels dtype:", labels.dtype,
                         "Teacher logits shape:", tf.shape(teacher_logits), "Teacher logits dtype:", teacher_logits.dtype,
                         output_stream=sys.stderr, summarize=-1)
                # --- END DEBUG PRINT ---
                
                # --- Use _dice_metric for evaluation ---
                current_student_dice = self._dice_metric(labels, student_logits)
                current_teacher_dice = self._dice_metric(labels, teacher_logits)
                
                if not tf.math.is_nan(current_student_dice): # Avoid NaNs
                    student_dice_scores.append(float(current_student_dice))
                if not tf.math.is_nan(current_teacher_dice): # Avoid NaNs
                    teacher_dice_scores.append(float(current_teacher_dice))
        except Exception as e:
            tf.print(f"Error during validation loop: {str(e)}", output_stream=sys.stderr)
            # Return current mean or 0 if no scores yet
            mean_student_dice = np.mean(student_dice_scores) if student_dice_scores else 0.0
            mean_teacher_dice = np.mean(teacher_dice_scores) if teacher_dice_scores else 0.0
            return float(mean_student_dice), float(mean_teacher_dice)


        mean_student_dice = np.mean(student_dice_scores) if student_dice_scores else 0.0
        mean_teacher_dice = np.mean(teacher_dice_scores) if teacher_dice_scores else 0.0
        
        # --- Debug print for final validation scores ---
        tf.print(f"Validation Complete. Mean Student Dice: {mean_student_dice:.4f}, Mean Teacher Dice: {mean_teacher_dice:.4f}", output_stream=sys.stderr)

        return float(mean_student_dice), float(mean_teacher_dice)

    # --- Add save_checkpoint method (similar to supervised) ---
    def save_checkpoint(self, name='checkpoint'):
        """Save a model checkpoint with more metadata"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # Save teacher model weights
        checkpoint_path = checkpoint_dir / f"{name}_teacher_{timestamp}"
        self.teacher_model.save_weights(str(checkpoint_path))
        
        metadata = {
            'timestamp': timestamp,
            'epoch': len(self.history['val_dice']),
            'best_val_dice': max(self.history['val_dice']) if self.history['val_dice'] else 0,
            'current_val_dice': self.history['val_dice'][-1] if self.history['val_dice'] else 0,
            'learning_rate': float(self.lr_schedule(self.optimizer.iterations)),
            'img_size': (self.config.img_size_x, self.config.img_size_y),
            'num_classes': self.config.num_classes,
            'n_filters': self.config.n_filters,
            'ema_decay': self.ema_decay,
            'consistency_weight': self.consistency_weight,
            'consistency_rampup': self.consistency_rampup,
            'num_labeled': self.config.num_labeled
        }
        with open(f"{checkpoint_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved teacher checkpoint to {checkpoint_path}")
        return str(checkpoint_path)

    # --- Add plot_progress method (similar to supervised) ---
    def plot_progress(self):
        """Plot training progress"""
        if len(self.history['train_loss']) < 2: return
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.plot(self.history['train_loss'], label='Total Loss')
        plt.plot(self.history['sup_loss'], label='Supervised Loss')
        plt.plot(self.history['cons_loss'], label='Consistency Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()

        plt.subplot(1, 4, 2)
        plt.plot(self.history['val_dice'])
        plt.title(f'Validation Dice (Best: {max(self.history["val_dice"]):.4f})')
        plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(self.history['learning_rate'])
        plt.title('Learning Rate'); plt.xlabel('Epoch'); plt.ylabel('LR')
        plt.yscale('log'); plt.grid(True)
        
        # Add consistency weight plot
        steps_per_epoch = self.config.num_labeled // self.config.batch_size + 1
        epochs = len(self.history['train_loss'])
        steps = [e * steps_per_epoch for e in range(epochs)]
        cons_weights = [self.get_consistency_weight(s) for s in steps]
        plt.subplot(1, 4, 4)
        plt.plot(cons_weights)
        plt.title('Consistency Weight'); plt.xlabel('Epoch'); plt.ylabel('Weight'); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{plots_dir}/mt_progress_{timestamp}.png')
        plt.savefig(f'{plots_dir}/mt_latest_progress.png')
        plt.close()

    def train(self, data_paths):
        """Train with both labeled and unlabeled data"""
        print("\nStarting Enhanced Mean Teacher training...")
        
        # --- Use batch size from config ---
        batch_size = self.config.batch_size 
        print(f"Using batch size: {batch_size}")
        
        # --- Remove LR schedule recalculation - already done in _setup_training_params ---
        
        # Create datasets using the new function
        train_labeled_ds = create_augmented_dataset(
            self.data_pipeline.dataloader,
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=batch_size,
            shuffle=True,
            augment=True, # Augment labeled data
            is_unlabeled=False
        )
        
        train_unlabeled_ds = create_augmented_dataset(
            self.data_pipeline.dataloader,
            data_paths['unlabeled']['images'],
            None, # No labels
            batch_size=batch_size,
            shuffle=True,
            augment=True, # Augment unlabeled data for student input
            is_unlabeled=True
        )
        
        val_ds = create_augmented_dataset(
            self.data_pipeline.dataloader,
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=batch_size,
            shuffle=False,
            augment=False, # No augmentation for validation
            is_unlabeled=False
        )
        
        # Setup training parameters
        best_dice = 0
        patience = self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 15
        patience_counter = 0
        checkpoint_freq = 5 # Save every 5 epochs
        
        # Create experiment log file
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f'mean_teacher_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        
        with open(log_file, 'w') as f:
            f.write('epoch,loss,supervised_loss,consistency_loss,val_dice,learning_rate,time\n')
        
        # Log config
        print(f"\nTraining Configuration:")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Log file: {log_file}")
        print(f"- Batch size: {self.config.batch_size}")
        print(f"- Epochs: {self.config.num_epochs}")
        print(f"- Early stopping patience: {patience}")
        print(f"- Image size: {self.config.img_size_x}x{self.config.img_size_y}")
        print(f"- Num Labeled: {self.config.num_labeled}")
        print(f"- EMA Decay: {self.ema_decay}")
        print(f"- Consistency Weight: {self.consistency_weight} (Rampup: {self.consistency_rampup})")

        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            epoch_losses, epoch_sup_losses, epoch_cons_losses = [], [], []
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Zip datasets - repeat unlabeled if shorter
            train_combined_ds = tf.data.Dataset.zip((train_labeled_ds, train_unlabeled_ds.repeat()))
            
            progress_bar = tqdm(train_combined_ds, desc="Training", total=self.config.num_labeled // batch_size +1)

            for labeled_batch, unlabeled_batch in progress_bar:
                labeled_images, labeled_labels = labeled_batch
                unlabeled_images = unlabeled_batch # Dataset map already handles the tuple for unlabeled

                # Training step
                loss, dice, sup_loss, cons_loss = self.train_step(
                    labeled_images, labeled_labels, unlabeled_images)
                
                # Record metrics
                epoch_losses.append(float(loss))
                epoch_sup_losses.append(float(sup_loss))
                epoch_cons_losses.append(float(cons_loss))
                
                progress_bar.set_postfix({
                    "loss": f"{float(loss):.4f}", 
                    "sup": f"{float(sup_loss):.4f}", 
                    "cons": f"{float(cons_loss):.4f}",
                    "dice": f"{float(dice):.4f}" 
                })
            
            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation
            val_dice = self.validate(val_ds)
            
            # Force garbage collection again
            collect_garbage()
            
            # Update history
            epoch_time = time.time() - start_time
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0
            mean_sup_loss = np.mean(epoch_sup_losses) if epoch_sup_losses else 0
            mean_cons_loss = np.mean(epoch_cons_losses) if epoch_cons_losses else 0
            
            self.history['train_loss'].append(mean_loss)
            self.history['sup_loss'].append(mean_sup_loss)
            self.history['cons_loss'].append(mean_cons_loss)
            self.history['val_dice'].append(val_dice)
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            print(f"Time: {epoch_time:.2f}s | Loss: {mean_loss:.4f} | Sup: {mean_sup_loss:.4f} | Cons: {mean_cons_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.8e}")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_sup_loss:.6f},{mean_cons_loss:.6f},{val_dice:.6f},{current_lr:.8e},{epoch_time:.2f}\n")
            
            # Save best model (based on validation dice)
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mean_teacher') # Use new save method
                print(f"✓ New best model saved! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            # Save periodic checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
                print(f"✓ Epoch {epoch+1} checkpoint saved")

            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

            # Plot progress
            if (epoch + 1) % 5 == 0:
                 collect_garbage()
                 self.plot_progress()
                 collect_garbage()

        # Save final model
        self.save_checkpoint('final_mean_teacher')
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        self.plot_progress() # Plot final progress
        return self.history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False, default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--output_dir', type=Path, required=False, default='/scratch/lustre/home/mdah0000/smm/v14/mean_teacher_results')
    parser.add_argument('--experiment_name', type=str, default='mean_teacher_enhanced_v2') # Updated name
    parser.add_argument('--batch_size', type=int, default=8) # Default batch size
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--consistency_weight', type=float, default=100.0)
    parser.add_argument('--consistency_rampup', type=int, default=120) # Example: 120 steps (30 epochs * 4 steps/epoch)
    parser.add_argument('--num_labeled', type=int, default=30) # Default small labeled set for MT
    parser.add_argument('--num_validation', type=int, default=56)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--early_stopping_patience', type=int, default=15) # Add patience arg
    args = parser.parse_args()

    # --- Use custom GPU setup ---
    custom_setup_gpu() 
    
    # Create config
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size # Use arg
    config.img_size_x = args.img_size
    config.img_size_y = args.img_size
    config.n_filters = args.n_filters
    config.num_classes = 2
    # --- Pass MT specific args to config ---
    config.ema_decay = args.ema_decay
    config.consistency_weight = args.consistency_weight
    config.consistency_rampup = args.consistency_rampup
    config.num_labeled = args.num_labeled # Store num_labeled for LR schedule calc
    config.early_stopping_patience = args.early_stopping_patience # Store patience

    # Create experiment config (not strictly needed if output_dir is passed directly)
    # ...

    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=args.num_labeled, num_validation=args.num_validation)
    print(f"Using {len(data_paths['labeled']['images'])} labeled, {len(data_paths['unlabeled']['images'])} unlabeled, {len(data_paths['validation']['images'])} validation images.")
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer, passing output_dir
    print("Creating memory-optimized Mean Teacher trainer...")
    trainer = EnhancedMeanTeacherTrainer(config, args.output_dir) 
    trainer.data_pipeline = data_pipeline
    # Trainer already reads MT params from config in _setup_training_params
    
    # Create experiment directory
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    trainer.train(data_paths)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
EOF

# Make the script executable
chmod +x $WORK_DIR/run_mean_teacher_enhanced.py
echo "Made script executable"

# Run the enhanced Mean Teacher script explicitly with python interpreter
echo "Starting enhanced Mean Teacher training with optimized parameters..."
# --- Update arguments passed to the script ---
$PYTHON_CMD $WORK_DIR/run_mean_teacher_enhanced.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --experiment_name "mean_teacher_enhanced_v2" \
    --ema_decay 0.999 \
    --consistency_weight 100.0 \
    --consistency_rampup 120 \
    --batch_size 8 \
    --num_epochs 100 \
    --n_filters 32 \
    --img_size 256 \
    --num_labeled 30 \
    --num_validation 56 \
    --early_stopping_patience 15

echo "========================================================"
echo "Mean Teacher Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="