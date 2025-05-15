#!/bin/bash
# SLURM batch job script for high-performance Mean Teacher semi-supervised learning

#SBATCH -p gpu                         # Use gpu partition 
#SBATCH --gres=gpu:2                   # Request 2 GPUs
#SBATCH -n 8                           # Request 8 CPU cores
#SBATCH --mem=64G                      # Request 64GB RAM
#SBATCH --time=24:00:00                # Time limit to 24 hours (max allowed)
#SBATCH -o meanteacher-highperf-%j.out # Output file name with job ID
#SBATCH -e meanteacher-highperf-%j.err # Error file name with job ID
#SBATCH --mail-type=END,FAIL           # Send email when job ends or fails
#SBATCH --job-name=mt_highperf         # Add descriptive job name

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mean_teacher_results_highperf"

echo "========================================================"
echo "Running High-Performance Mean Teacher Learning - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory: $WORK_DIR"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================"

# Create output directory structure
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/visualizations
mkdir -p $OUTPUT_DIR/cached_data

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings - optimized for Mean Teacher
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# Memory optimization settings
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_MEMORY_ALLOCATION=0.85
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Enable mixed precision for better performance and lower memory usage
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Install required packages
echo "Installing necessary packages..."
pip install --user --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib pandas h5py

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create high-performance version of Mean Teacher
echo "Creating high-performance Mean Teacher implementation..."
cat > $WORK_DIR/run_mean_teacher_highperf.py << 'EOF'
#!/usr/bin/env python3
# High-performance Mean Teacher implementation with memory optimization and fixed dimensions

import tensorflow as tf
import numpy as np
import time
import os
import gc
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import psutil # Import psutil

# Import modules from codebase
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import StableSSLTrainer
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from main import prepare_data_paths

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Setup GPU with memory optimization
def setup_gpu():
    """Setup GPU with memory optimization for Mean Teacher"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU.")
        return False
    
    print(f"Found {len(physical_devices)} GPU(s)")
    
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
            
            if os.environ.get('TF_MEMORY_ALLOCATION'):
                mem_limit = float(os.environ.get('TF_MEMORY_ALLOCATION'))
                print(f"Using up to {mem_limit*100}% of available memory for {device}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
    
    # Enable mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision (float16) for better performance")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    return True

# Force garbage collection
def collect_garbage():
    """Force garbage collection and clear TF memory"""
    gc.collect()
    tf.keras.backend.clear_session()
    
    for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
        try:
            with tf.device(f'/GPU:{i}'):
                dummy = tf.random.normal([1])
                del dummy
        except Exception:
            pass
    
    print("Memory cleared")

# Cache preprocessed data class - shared with MixMatch
class DataCache:
    """Cache preprocessed data to disk for faster training"""
    
    def __init__(self, cache_dir, batch_size):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.cache_file = self.cache_dir / f"mean_teacher_data_cache_bs{batch_size}.h5"
        
    def cache_exists(self):
        return self.cache_file.exists()
    
    def cache_datasets(self, train_labeled, train_unlabeled, val_ds):
        """Cache datasets to disk in HDF5 format"""
        print(f"Caching datasets to {self.cache_file}...")
        
        with h5py.File(self.cache_file, 'w') as f:
            # Create groups
            labeled_grp = f.create_group('labeled')
            unlabeled_grp = f.create_group('unlabeled')
            val_grp = f.create_group('validation')
            
            # Cache labeled data
            labeled_images = []
            labeled_labels = []
            
            print("Caching labeled data...")
            for i, (images, labels) in enumerate(tqdm(train_labeled)):
                labeled_images.append(images.numpy())
                labeled_labels.append(labels.numpy())
                
            labeled_grp.create_dataset('images', data=np.concatenate(labeled_images, axis=0))
            labeled_grp.create_dataset('labels', data=np.concatenate(labeled_labels, axis=0))
            
            # Cache unlabeled data if available
            unlabeled_images = []
            
            try:
                print("Caching unlabeled data...")
                for i, batch in enumerate(tqdm(train_unlabeled)):
                    # Handle different return types
                    if isinstance(batch, tuple):
                        images = batch[0]
                    else:
                        images = batch
                    unlabeled_images.append(images.numpy())
                    
                unlabeled_grp.create_dataset('images', data=np.concatenate(unlabeled_images, axis=0))
            except Exception as e:
                print(f"Error caching unlabeled data: {e}")
            
            # Cache validation data
            val_images = []
            val_labels = []
            
            print("Caching validation data...")
            for i, (images, labels) in enumerate(tqdm(val_ds)):
                val_images.append(images.numpy())
                val_labels.append(labels.numpy())
                
            val_grp.create_dataset('images', data=np.concatenate(val_images, axis=0))
            val_grp.create_dataset('labels', data=np.concatenate(val_labels, axis=0))
            
        print("Dataset caching complete.")
    
    def load_cached_datasets(self):
        """Load cached datasets from disk"""
        if not self.cache_exists():
            raise FileNotFoundError(f"Cache file {self.cache_file} not found")
            
        print(f"Loading cached datasets from {self.cache_file}...")
        
        with h5py.File(self.cache_file, 'r') as f:
            # Load labeled data
            labeled_images = f['labeled/images'][:]
            labeled_labels = f['labeled/labels'][:]
            
            # Load validation data
            val_images = f['validation/images'][:]
            val_labels = f['validation/labels'][:]
            
            # Load unlabeled data if available
            has_unlabeled = 'unlabeled/images' in f
            if has_unlabeled:
                unlabeled_images = f['unlabeled/images'][:]
                
        # Create TensorFlow datasets
        train_labeled = tf.data.Dataset.from_tensor_slices((labeled_images, labeled_labels))
        train_labeled = train_labeled.batch(self.batch_size).prefetch(1) # Reduced prefetch buffer
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_ds = val_ds.batch(self.batch_size).prefetch(1) # Reduced prefetch buffer
        
        # Create unlabeled dataset if available
        if has_unlabeled:
            train_unlabeled = tf.data.Dataset.from_tensor_slices(unlabeled_images)
            train_unlabeled = train_unlabeled.batch(self.batch_size).prefetch(1) # Reduced prefetch buffer
            print("Cached datasets loaded successfully (with unlabeled data).")
            return train_labeled, train_unlabeled, val_ds
        else:
            print("Cached datasets loaded successfully (without unlabeled data).")
            return train_labeled, None, val_ds

# High-performance Mean Teacher trainer
class HighPerfMeanTeacherTrainer(StableSSLTrainer):
    """Enhanced Mean Teacher trainer with memory optimizations and fixed dimensions"""
    
    def __init__(self, config):
        self.warmup_epochs = 5  # Set before super to avoid attribute error
        self.best_dice = 0
        self.validation_plateau = 0
        
        # Call parent's init
        super().__init__(config)
        
        self.config = config
        
        # --- Explicitly call setup methods ---
        self._setup_model()
        self._setup_training_params()
        # --- End explicit calls ---
        
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': [],
            'memory_usage': []
        }
    
    def _setup_model(self):
        """Setup student and teacher models with memory efficiency"""
        print("Setting up memory-efficient Mean Teacher models...")
        
        # Set model parameters - smaller filters to save memory
        self.config.n_filters = 16 # Reduced filters further from 32 to 16
        
        # Create student model
        self.model = PancreasSeg(self.config)
        
        # Create teacher model
        self.teacher_model = PancreasSeg(self.config)
        
        # Initialize models with dummy input
        # Ensure config attributes are set before this point
        if not hasattr(self.config, 'img_size_x'): self.config.img_size_x = 256 # Default if not set
        if not hasattr(self.config, 'img_size_y'): self.config.img_size_y = 256 # Default if not set
        if not hasattr(self.config, 'num_channels'): self.config.num_channels = 1 # Default if not set
        
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        
        try:
            _ = self.model(dummy_input)
            _ = self.teacher_model(dummy_input)
            
            # Copy weights from student to teacher only after successful initialization
            self.teacher_model.set_weights(self.model.get_weights())
            
            print(f"Model created with input shape: ({self.config.img_size_x}, {self.config.img_size_y}, {self.config.num_channels})")
            print(f"Initial filters: {self.config.n_filters}")
            
            # Print model summary
            total_params = self.model.count_params()
            print(f"Total model parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
            
        except Exception as e:
             print(f"Error during model initialization or weight setting: {e}")
             # Decide how to handle this - maybe raise the error or exit
             raise e # Re-raise the error to stop execution if model setup fails

    def _setup_training_params(self):
        """Setup training parameters with learning rate scheduler"""
        # Define CosineWarmupSchedule class
        class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
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
        steps_per_epoch = 225 // self.config.batch_size
        total_steps = self.config.num_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        # Learning rate parameters
        max_lr = 8e-4
        min_lr = 5e-6
        
        # Create learning rate scheduler
        self.lr_schedule = CosineWarmupSchedule(
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Set Mean Teacher specific parameters
        self.ema_decay_start = 0.99
        self.ema_decay_end = 0.999
        self.consistency_weight_max = 20.0
        self.consistency_rampup_epochs = 80
        
        # Optimizer with weight decay
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            weight_decay=1e-5
        )
        
        print(f"Learning rate schedule: Cosine with warmup from {min_lr} to {max_lr}")
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        print(f"EMA decay: {self.ema_decay_start} to {self.ema_decay_end}")
        print(f"Consistency weight max: {self.consistency_weight_max}")
    
    def _get_ema_decay(self, epoch):
        """Get adaptive EMA decay based on current epoch"""
        if epoch >= self.consistency_rampup_epochs:
            return self.ema_decay_end
        
        alpha = epoch / self.consistency_rampup_epochs
        return self.ema_decay_start + alpha * (self.ema_decay_end - self.ema_decay_start)
    
    def update_teacher_model(self, epoch):
        """Update teacher model with EMA of student weights"""
        ema_decay = self._get_ema_decay(epoch)
        
        # Get model weights
        student_weights = self.model.get_weights()
        teacher_weights = self.teacher_model.get_weights()
        
        # Update teacher weights with EMA
        new_weights = []
        for sw, tw in zip(student_weights, teacher_weights):
            new_weights.append(ema_decay * tw + (1 - ema_decay) * sw)
            
        # Apply new weights to teacher model
        self.teacher_model.set_weights(new_weights)
        
        return ema_decay
    
    def _dice_loss(self, y_true, y_pred):
        """Memory-efficient Dice loss implementation with float32 intermediate calculations and denominator clipping"""
        # --- Add input checks ---
        tf.debugging.check_numerics(y_true, "Input y_true to _dice_loss contains NaN/Inf")
        tf.debugging.check_numerics(y_pred, "Input y_pred to _dice_loss contains NaN/Inf")
        
        # Apply sigmoid for binary segmentation
        y_pred_sigmoid = tf.sigmoid(y_pred)
        tf.debugging.check_numerics(y_pred_sigmoid, "Sigmoid output in _dice_loss contains NaN/Inf")

        # Cast y_true to prediction dtype
        y_true = tf.cast(y_true, y_pred_sigmoid.dtype)
        
        epsilon = 1e-4 
        
        # Flatten predictions and targets
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred_sigmoid, [-1])
        
        # Perform calculations in float32 for stability
        y_true_flat_f32 = tf.cast(y_true_flat, tf.float32)
        y_pred_flat_f32 = tf.cast(y_pred_flat, tf.float32)
        
        intersection = tf.reduce_sum(y_true_flat_f32 * y_pred_flat_f32)
        # --- Calculate denominator components separately for checking ---
        sum_true = tf.reduce_sum(y_true_flat_f32)
        sum_pred = tf.reduce_sum(y_pred_flat_f32)
        
        # --- Check sums before calculating denominator ---
        tf.debugging.check_numerics(sum_true, "Sum of y_true in _dice_loss contains NaN/Inf")
        tf.debugging.check_numerics(sum_pred, "Sum of y_pred in _dice_loss contains NaN/Inf")

        denominator = sum_true + sum_pred
        
        # --- Clip denominator to avoid division by zero ---
        # Use tf.maximum to ensure denominator is at least epsilon
        dice_score = (2.0 * intersection + tf.cast(epsilon, tf.float32)) / (tf.maximum(denominator, 0.0) + tf.cast(epsilon, tf.float32))
        
        # Check final dice_score for NaN/Inf before returning loss
        tf.debugging.check_numerics(dice_score, "Final Dice score calculation resulted in NaN/Inf")
        
        return 1.0 - dice_score
    
    def _weighted_bce_loss(self, y_true, y_pred):
        """Fixed weighted BCE loss with proper dimension handling"""
        # Apply sigmoid activation
        y_pred_sigmoid = tf.sigmoid(y_pred)
        
        # --- Cast y_true to prediction dtype ---
        y_true = tf.cast(y_true, y_pred_sigmoid.dtype)
        
        # --- Use larger epsilon for float16 stability ---
        epsilon = 1e-4 
        
        # --- Explicitly handle channel dimensions ---
        # Ensure y_true is [B, H, W, 1] (taking foreground if multi-channel)
        if tf.rank(y_true) == 4 and y_true.shape[-1] > 1:
            y_true_target = y_true[..., 1:2]
        elif tf.rank(y_true) == 3: # Add channel dim if missing
             y_true_target = tf.expand_dims(y_true, axis=-1)
        else: # Assume already [B, H, W, 1] or similar single channel
            y_true_target = y_true

        # Ensure y_pred_sigmoid is [B, H, W, 1] (taking first channel if multi-channel)
        if tf.rank(y_pred_sigmoid) == 4 and y_pred_sigmoid.shape[-1] > 1:
            y_pred_target = y_pred_sigmoid[..., 0:1]
        elif tf.rank(y_pred_sigmoid) == 3: # Add channel dim if missing (unlikely for logits)
             y_pred_target = tf.expand_dims(y_pred_sigmoid, axis=-1)
        else: # Assume already [B, H, W, 1] or similar single channel
            y_pred_target = y_pred_sigmoid

        # Ensure shapes match before BCE calculation
        # This reshape might still be needed if spatial dimensions differ unexpectedly, but let's rely on explicit channel handling first.
        # y_true_target = tf.reshape(y_true_target, tf.shape(y_pred_target)) # Commented out for now

        # Calculate standard BCE loss - use TF's binary_crossentropy for numerical stability
        # Input shapes: [B, H, W, 1], Output shape: [B, H, W]
        bce = tf.keras.losses.binary_crossentropy(
            y_true_target, 
            y_pred_target,
            from_logits=False,
            label_smoothing=epsilon # Use label smoothing as a form of epsilon
        )
        
        # Add focal loss component (focus more on hard examples)
        gamma = 2.0
        # Use target shapes for p_t calculation
        p_t = y_true_target * y_pred_target + (1 - y_true_target) * (1 - y_pred_target) # Shape [B, H, W, 1]
        # --- Clip p_t to avoid issues with pow ---
        p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)
        focal_weight = tf.pow(1.0 - p_t, gamma) # Shape [B, H, W, 1]
        
        # Apply additional weight to positive pixels (foreground class)
        pos_weight = 4.0
        class_weight = 1.0 + (pos_weight - 1.0) * y_true_target # Shape [B, H, W, 1]
        
        # Combine focal and class weights
        weight_map = class_weight * focal_weight # Shape [B, H, W, 1]
        
        # Apply weights to BCE loss
        # --- Squeeze weight_map to match bce shape [B, H, W] ---
        weighted_bce = bce * tf.squeeze(weight_map, axis=-1) # Shape [B, H, W]
        
        # Return mean over all dimensions
        mean_bce = tf.reduce_mean(weighted_bce)
        # --- Add check for NaN/Inf in BCE loss ---
        tf.debugging.check_numerics(mean_bce, "Weighted BCE calculation resulted in NaN/Inf")
        return mean_bce
    
    def _consistency_loss(self, student_logits, teacher_logits):
        """Improved consistency loss with structural component"""
        # Apply sigmoid
        student_probs = tf.sigmoid(student_logits)
        teacher_probs = tf.sigmoid(teacher_logits)
        
        # MSE loss component
        mse_loss = tf.reduce_mean(tf.square(student_probs - teacher_probs))
        
        # --- Temporarily disabled gradient difference loss for memory testing ---
        # # Gradient difference loss for structural consistency
        # def _image_gradients(images):
        #     # Simple finite difference approximation
        #     batch_size = tf.shape(images)[0]
        #     dy = images[:, 1:, :, :] - images[:, :-1, :, :]
        #     dx = images[:, :, 1:, :] - images[:, :, :-1, :]
            
        #     # Zero-pad to match original size
        #     dy = tf.pad(dy, [[0, 0], [0, 1], [0, 0], [0, 0]])
        #     dx = tf.pad(dx, [[0, 0], [0, 0], [0, 1], [0, 0]])
            
        #     return dx, dy
        
        # # Get gradients
        # student_dx, student_dy = _image_gradients(student_probs)
        # teacher_dx, teacher_dy = _image_gradients(teacher_probs)
        
        # # Gradient consistency loss
        # grad_dx_loss = tf.reduce_mean(tf.abs(student_dx - teacher_dx))
        # grad_dy_loss = tf.reduce_mean(tf.abs(student_dy - teacher_dy))
        # grad_loss = grad_dx_loss + grad_dy_loss
        
        # # Combine with weighting
        # combined_loss = 0.7 * mse_loss + 0.3 * grad_loss
        # return combined_loss
        # --- End of disabled section ---

        return mse_loss # Return only MSE loss for now
    
    def get_consistency_weight(self, epoch):
        """Calculate consistency weight with warmup period"""
        # No consistency in very beginning
        if epoch < self.warmup_epochs:
            return 0.0
            
        # Linear ramp up
        if epoch < self.consistency_rampup_epochs:
            return self.consistency_weight_max * (epoch - self.warmup_epochs) / (self.consistency_rampup_epochs - self.warmup_epochs)
            
        # Full weight after ramp-up
        return self.consistency_weight_max
    
    def _apply_augmentation(self, image, is_teacher=False):
        """Lightweight image augmentation for Mean Teacher"""
        # Different strength for student vs teacher
        strength = 0.5 if not is_teacher else 0.2
        
        # Random brightness (student only)
        if not is_teacher:
            image = tf.image.random_brightness(image, max_delta=0.1 * strength)
        
        # Random contrast (student only)
        if not is_teacher:
            image = tf.image.random_contrast(image, lower=1.0-0.2*strength, upper=1.0+0.2*strength)
        
        # Gaussian noise for both (different strength)
        noise_stddev = 0.02 * strength
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev, dtype=image.dtype)
        image = image + noise
        
        # Clip values to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images, consistency_weight, epoch):
        """Memory-efficient training step with fixed dimension handling"""
        with tf.GradientTape() as tape:
            # Apply augmentations specific to student and teacher
            labeled_images_student = self._apply_augmentation(labeled_images, is_teacher=False)
            unlabeled_images_student = self._apply_augmentation(unlabeled_images, is_teacher=False)
            unlabeled_images_teacher = self._apply_augmentation(unlabeled_images, is_teacher=True)
            
            # Forward pass through student model for labeled data
            labeled_logits = self.model(labeled_images_student, training=True) # Output might be float16
            
            # Calculate supervised losses
            # _dice_loss likely returns float32 due to internal casting
            dice_loss = self._dice_loss(labeled_labels, labeled_logits) 
            
            bce_loss = tf.cast(0.0, tf.float32) # Initialize bce_loss as float32
            try:
                # Determine labels for BCE
                if labeled_labels.shape[-1] > 1:
                    if labeled_logits.shape[-1] == 1:
                        bce_labels = labeled_labels[..., 1:2]
                    else:
                        bce_labels = labeled_labels
                else:
                    bce_labels = labeled_labels
                
                # _weighted_bce_loss might return float16 or float32
                calculated_bce_loss = self._weighted_bce_loss(bce_labels, labeled_logits)
                # Explicitly cast the result to float32
                bce_loss = tf.cast(calculated_bce_loss, tf.float32) 

            except Exception as e:
                tf.print(f"Warning: Error in BCE loss calculation, using only dice loss: {e}") # Use tf.print inside tf.function
                # Ensure bce_loss remains 0.0 (float32)
                bce_loss = tf.cast(0.0, tf.float32) 
            
            # --- Cast both losses to float32 before combining ---
            dice_loss_f32 = tf.cast(dice_loss, tf.float32)
            # bce_loss is already float32 from the try/except block
            
            supervised_loss = 0.8 * dice_loss_f32 + 0.2 * bce_loss 
            
            # Consistency part
            consistency_loss = tf.cast(0.0, tf.float32) # Initialize consistency_loss as float32
            if consistency_weight > 0:
                unlabeled_student_logits = self.model(unlabeled_images_student, training=True)
                unlabeled_teacher_logits = self.teacher_model(unlabeled_images_teacher, training=False)
                # _consistency_loss might return float16 or float32
                calculated_consistency_loss = self._consistency_loss(unlabeled_student_logits, unlabeled_teacher_logits)
                # Explicitly cast the result to float32
                consistency_loss = tf.cast(calculated_consistency_loss, tf.float32)
            
            # Combined loss (all components are now float32)
            consistency_weight_f32 = tf.cast(consistency_weight, tf.float32)
            total_loss = supervised_loss + consistency_weight_f32 * consistency_loss
            # --- Check total_loss for NaN/Inf ---
            tf.debugging.check_numerics(total_loss, "Total loss calculation resulted in NaN/Inf")

        # Get gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # --- Check gradients for NaN/Inf ---
        for grad in gradients:
            if grad is not None:
                 # Cast gradient check to handle potential None values gracefully
                 try:
                     tf.debugging.check_numerics(grad, f"Gradient for variable {getattr(grad, 'name', 'unknown')} has NaN/Inf")
                 except AttributeError: # Handle cases where grad might not have a 'name'
                     tf.debugging.check_numerics(grad, "Gradient has NaN/Inf")

        # Clip gradients to prevent explosion
        gradients = [grad if grad is not None else tf.zeros_like(var) 
                     for grad, var in zip(gradients, self.model.trainable_variables)] # Handle None gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0) # Reduced clipping norm

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate training dice score for monitoring (use float32 dice loss)
        dice_score = 1.0 - dice_loss_f32 
        
        return {
            'total_loss': total_loss, 
            'dice_score': dice_score, 
            'supervised_loss': supervised_loss, 
            'consistency_loss': consistency_loss
        }
    
    def validate(self, val_dataset):
        """Validate with both student and teacher models"""
        student_dice_scores = []
        teacher_dice_scores = []
        
        for images, labels in val_dataset:
            # Student validation
            student_logits = self.model(images, training=False)
            student_dice = 1 - self._dice_loss(labels, student_logits)
            student_dice_scores.append(float(student_dice))
            
            # Teacher validation
            teacher_logits = self.teacher_model(images, training=False)
            teacher_dice = 1 - self._dice_loss(labels, teacher_logits)
            teacher_dice_scores.append(float(teacher_dice))
        
        # Calculate mean scores
        student_mean_dice = np.mean(student_dice_scores) if student_dice_scores else 0.0
        teacher_mean_dice = np.mean(teacher_dice_scores) if teacher_dice_scores else 0.0
        
        print(f"Mean validation Dice - Student: {student_mean_dice:.4f}, Teacher: {teacher_mean_dice:.4f}")
        
        return student_mean_dice, teacher_mean_dice
    
    def _compute_memory_usage(self):
        """Get current GPU and RAM memory usage"""
        gpu_mem_gb = 0.0
        ram_mem_gb = 0.0
        try:
            # GPU Memory (assuming GPU:0 is primary)
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            gpu_mem_gb = memory_info['current'] / (1024 * 1024 * 1024)  # GB
        except Exception as e:
            # print(f"Could not get GPU memory info: {e}") # Optional: uncomment for debugging
            pass 
            
        try:
            # RAM Usage for the current process
            process = psutil.Process(os.getpid())
            ram_mem_gb = process.memory_info().rss / (1024 * 1024 * 1024) # GB (Resident Set Size)
        except Exception as e:
            # print(f"Could not get RAM usage info: {e}") # Optional: uncomment for debugging
            pass
            
        return gpu_mem_gb, ram_mem_gb

    def plot_learning_curves(self, filepath=None):
        """Plot and save learning curves"""
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
        
        # Plot memory usage if available
        if self.history['memory_usage']:
            gpu_mem = [mem[0] for mem in self.history['memory_usage']]
            ram_mem = [mem[1] for mem in self.history['memory_usage']]
            
            plt.subplot(2, 2, 4)
            plt.plot(gpu_mem, label='GPU Memory (GB)')
            plt.plot(ram_mem, label='Process RAM (GB)')
            plt.xlabel('Epoch')
            plt.ylabel('Memory Usage (GB)')
            plt.title('Resource Usage')
            plt.legend()
            plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=150)
            print(f"Learning curves saved to {filepath}")
        
        plt.close()
    
    def train(self, data_paths, use_cache=True, cache_dir=None):
        """Train with high-performance optimizations"""
        print("\nStarting high-performance Mean Teacher training...")
        
        # Use a small batch size to prevent memory issues
        batch_size = self.config.batch_size
        print(f"Using batch size: {batch_size}")
        
        # Create or load datasets based on caching preference
        if use_cache and cache_dir:
            # Initialize cache handler
            cache = DataCache(cache_dir, batch_size)
            
            if cache.cache_exists():
                # Load from cache
                train_labeled, train_unlabeled, val_ds = cache.load_cached_datasets()
                has_unlabeled = train_unlabeled is not None
            else:
                # Create datasets for caching
                print("Creating datasets for caching...")
                
                # Create labeled dataset
                train_labeled = self.data_pipeline.dataloader.create_dataset(
                    data_paths['labeled']['images'],
                    data_paths['labeled']['labels'],
                    batch_size=batch_size,
                    shuffle=True
                )
                
                # Create unlabeled dataset if available
                has_unlabeled = False
                train_unlabeled = None
                if 'unlabeled' in data_paths and data_paths['unlabeled']['images']:
                    print(f"Found {len(data_paths['unlabeled']['images'])} unlabeled images")
                    has_unlabeled = True
                    train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                        data_paths['unlabeled']['images'],
                        None,  # No labels for unlabeled data
                        batch_size=batch_size,
                        shuffle=True
                    )
                
                # Create validation dataset
                val_ds = self.data_pipeline.dataloader.create_dataset(
                    data_paths['validation']['images'],
                    data_paths['validation']['labels'],
                    batch_size=batch_size,
                    shuffle=False
                )
                
                # Cache datasets
                cache.cache_datasets(train_labeled, train_unlabeled, val_ds)
                
                # Load cached datasets
                train_labeled, train_unlabeled, val_ds = cache.load_cached_datasets()
                has_unlabeled = train_unlabeled is not None
        else:
            # Create datasets without caching
            print("Creating datasets without caching...")
            
            # Create labeled dataset
            train_labeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['labeled']['images'],
                data_paths['labeled']['labels'],
                batch_size=batch_size,
                shuffle=True
            )
            
            # Create unlabeled dataset if available
            has_unlabeled = False
            train_unlabeled = None
            if 'unlabeled' in data_paths and data_paths['unlabeled']['images']:
                print(f"Found {len(data_paths['unlabeled']['images'])} unlabeled images")
                has_unlabeled = True
                train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                    data_paths['unlabeled']['images'],
                    None,  # No labels for unlabeled data
                    batch_size=batch_size,
                    shuffle=True
                )
            
            # Create validation dataset
            val_ds = self.data_pipeline.dataloader.create_dataset(
                data_paths['validation']['images'],
                data_paths['validation']['labels'],
                batch_size=batch_size,
                shuffle=False
            )
        
        # Verify datasets
        print("Validating datasets...")
        try:
            for batch in train_labeled.take(1):
                labeled_images, labeled_labels = batch
                print(f"Labeled batch - Images shape: {labeled_images.shape}, Labels shape: {labeled_labels.shape}")
            
            if has_unlabeled:
                for batch in train_unlabeled.take(1):
                    print(f"Unlabeled batch - Shape: {batch.shape}")
            
            for batch in val_ds.take(1):
                val_images, val_labels = batch
                print(f"Validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")
        except Exception as e:
            print(f"Error validating datasets: {e}")
        
        # Training setup
        best_dice = 0
        patience = 20  # Increased patience for better training
        patience_counter = 0
        checkpoint_dir = Path(self.config.checkpoint_dir) / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file
        log_dir = Path(self.config.checkpoint_dir) / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f'mean_teacher_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Create visualization directory
        vis_dir = Path(self.config.checkpoint_dir) / 'visualizations'
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Write CSV header
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,supervised_loss,consistency_loss,student_dice,teacher_dice,'
                    'learning_rate,ema_decay,consistency_weight,gpu_memory_usage,ram_memory_usage,time\n')
        
        # If no unlabeled data, cache labeled data as unlabeled
        if not has_unlabeled:
            print("No unlabeled data available. Using labeled images without labels as unlabeled data.")
            unlabeled_data = []
            for labeled_batch in train_labeled:
                unlabeled_data.append(labeled_batch[0])  # Only save images
            
            print(f"Cached {len(unlabeled_data)} batches as 'unlabeled' data")
        elif has_unlabeled:
            # Store unlabeled data in memory for faster access
            print("Caching unlabeled data in memory...")
            unlabeled_data = []
            for unlabeled_batch in train_unlabeled:
                unlabeled_data.append(unlabeled_batch)  # Save full batch
            
            print(f"Cached {len(unlabeled_data)} unlabeled batches")
        
        # --- Optimized Unlabeled Data Handling ---
        # Create repeatable iterators instead of loading all to memory
        train_labeled_iter = iter(train_labeled.repeat()) # Repeat indefinitely
        
        if has_unlabeled:
            train_unlabeled_iter = iter(train_unlabeled.repeat()) # Repeat indefinitely
            print("Using repeatable iterator for unlabeled data.")
        else:
            # If no unlabeled data, use labeled images without labels
            train_unlabeled_iter = iter(train_labeled.map(lambda img, lbl: img).repeat())
            print("No unlabeled data found. Using labeled images via repeatable iterator.")
        # --- End of Optimized Unlabeled Data Handling ---

        # --- Add nvidia-smi check before loop ---
        print("\n--- GPU Status Before Training Loop ---")
        os.system('nvidia-smi')
        print("-------------------------------------\n")
        # --- End nvidia-smi check ---

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Setup metrics tracking
            epoch_losses = []
            epoch_supervised_losses = []
            epoch_consistency_losses = []
            epoch_dice_scores = []
            epoch_memory_usage_samples = [] # Store multiple samples per epoch

            # Get consistency weight for this epoch
            consistency_weight = self.get_consistency_weight(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Consistency weight: {consistency_weight:.4f}")
            
            # Update teacher model with adaptive EMA decay
            ema_decay = self.update_teacher_model(epoch)
            print(f"EMA decay: {ema_decay:.6f}")
            
            # Record memory usage at start of epoch
            gpu_mem, ram_mem = self._compute_memory_usage()
            epoch_memory_usage_samples.append((gpu_mem, ram_mem))
            print(f"Memory usage (Start Epoch): GPU={gpu_mem:.2f} GB, RAM={ram_mem:.2f} GB")
            
            # Progress bar for this epoch
            # Calculate steps per epoch based on the labeled dataset size
            steps_per_epoch = sum(1 for _ in train_labeled) 
            progress_bar = tqdm(range(steps_per_epoch), desc="Training")
            
            for batch_idx in progress_bar: # Iterate based on steps_per_epoch
                try:
                    # Get next batch from repeatable iterators
                    labeled_images, labeled_labels = next(train_labeled_iter)
                    unlabeled_batch = next(train_unlabeled_iter)
                    
                    # Training step
                    metrics = self.train_step(
                        labeled_images, labeled_labels, unlabeled_batch, 
                        consistency_weight, epoch)
                    
                    # Record metrics
                    epoch_losses.append(float(metrics['total_loss']))
                    epoch_supervised_losses.append(float(metrics['supervised_loss']))
                    epoch_consistency_losses.append(float(metrics['consistency_loss']))
                    epoch_dice_scores.append(float(metrics['dice_score']))
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{float(metrics['total_loss']):.4f}",
                        'dice': f"{float(metrics['dice_score']):.4f}",
                        'cons': f"{float(metrics['consistency_loss']):.4f}"
                    })
                    
                    # Update teacher model every N batches for faster adaptation
                    if batch_idx % 10 == 0 and batch_idx > 0:
                        self.update_teacher_model(epoch)
                    
                    # Clear memory and record usage periodically
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        collect_garbage()
                        gpu_mem, ram_mem = self._compute_memory_usage()
                        epoch_memory_usage_samples.append((gpu_mem, ram_mem))
                        # print(f"Memory usage (Batch {batch_idx}): GPU={gpu_mem:.2f} GB, RAM={ram_mem:.2f} GB") # Optional verbose logging
                        
                except tf.errors.ResourceExhaustedError:
                    print("\nOUT OF MEMORY ERROR: Skipping batch and collecting garbage...")
                    collect_garbage()
                    continue
                    
                except Exception as e:
                    print(f"\nError during training step {batch_idx}: {e}")
                    # Attempt to recover by clearing memory and skipping batch
                    collect_garbage()
                    continue
            
            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation
            student_dice, teacher_dice = self.validate(val_ds)
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0
            mean_supervised_loss = np.mean(epoch_supervised_losses) if epoch_supervised_losses else 0
            mean_consistency_loss = np.mean(epoch_consistency_losses) if epoch_consistency_losses else 0
            mean_train_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0
            mean_memory = np.mean(epoch_memory_usage_samples) if epoch_memory_usage_samples else self._compute_memory_usage()
            
            # Calculate average memory usage for the epoch
            if epoch_memory_usage_samples:
                avg_gpu_mem = np.mean([mem[0] for mem in epoch_memory_usage_samples])
                avg_ram_mem = np.mean([mem[1] for mem in epoch_memory_usage_samples])
            else:
                avg_gpu_mem, avg_ram_mem = self._compute_memory_usage() # Get current if no samples
            
            # Get current learning rate
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            
            # Update history
            self.history['train_loss'].append(mean_loss)
            self.history['val_dice'].append(student_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(current_lr)
            self.history['supervised_loss'].append(mean_supervised_loss)
            self.history['consistency_loss'].append(mean_consistency_loss)
            self.history['memory_usage'].append((avg_gpu_mem, avg_ram_mem)) # Store tuple
            
            # Detailed summary
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
            print(f"  Avg Memory: GPU={avg_gpu_mem:.2f} GB, RAM={avg_ram_mem:.2f} GB")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_supervised_loss:.6f},"
                        f"{mean_consistency_loss:.6f},{student_dice:.6f},{teacher_dice:.6f},"
                        f"{current_lr:.6e},{ema_decay:.6f},{consistency_weight:.4f},"
                        f"{avg_gpu_mem:.4f},{avg_ram_mem:.4f},{epoch_time:.2f}\n") # Added RAM to log
            
            # Save learning curve plot
            plot_path = vis_dir / f"learning_curves_epoch_{epoch+1}.png"
            self.plot_learning_curves(plot_path)
            
            # Save latest plot
            latest_plot_path = vis_dir / "latest_learning_curves.png"
            self.plot_learning_curves(latest_plot_path)
            
            # Save best models
            best_model = max(student_dice, teacher_dice)
            better_model = "teacher" if teacher_dice > student_dice else "student"
            
            if best_model > best_dice:
                best_dice = best_model
                
                # Save teacher model
                teacher_path = checkpoint_dir / f"best_teacher_model_{time.strftime('%Y%m%d_%H%M%S')}"
                self.teacher_model.save_weights(str(teacher_path))
                
                # Save student model
                student_path = checkpoint_dir / f"best_student_model_{time.strftime('%Y%m%d_%H%M%S')}"
                self.model.save_weights(str(student_path))
                
                print(f"✓ New best model saved! Dice: {best_dice:.4f} ({better_model} model)")
                patience_counter = 0
                self.validation_plateau = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Check for plateau
                if len(self.history['val_dice']) > 5:
                    recent_dices = self.history['val_dice'][-5:]
                    if max(recent_dices) - min(recent_dices) < 0.005:  # Very small improvement
                        self.validation_plateau += 1
                        print(f"Possible validation plateau detected: {self.validation_plateau}")
            
            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
            
            # Save intermediate model every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}"
                self.teacher_model.save_weights(str(checkpoint_path))
                print(f"Saved intermediate checkpoint at epoch {epoch+1}")
        
        # Save final models
        final_teacher_path = checkpoint_dir / f"final_teacher_model_{time.strftime('%Y%m%d_%H%M%S')}"
        self.teacher_model.save_weights(str(final_teacher_path))
        
        final_student_path = checkpoint_dir / f"final_student_model_{time.strftime('%Y%m%d_%H%M%S')}"
        self.model.save_weights(str(final_student_path))
        
        # Final learning curves
        final_plot_path = vis_dir / f"final_learning_curves.png" 
        self.plot_learning_curves(final_plot_path)
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history

def main():
    parser = argparse.ArgumentParser(description='High-performance Mean Teacher training')
    parser.add_argument('--data_dir', type=Path, default='/scratch/lustre/home/mdah0000/images/cropped',
                      help='Path to data directory')
    parser.add_argument('--output_dir', type=Path, default='/scratch/lustre/home/mdah0000/smm/v14/mean_teacher_results_highperf',
                      help='Path to output directory')
    parser.add_argument('--num_labeled', type=int, default=30,  # Use fewer labeled images
                      help='Number of labeled training images')
    parser.add_argument('--batch_size', type=int, default=2,  # Small batch size to prevent OOM
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--img_size', type=int, default=256,  # Reduced from 512 to save memory
                      help='Image size (both dimensions)')
    parser.add_argument('--use_cache', action='store_true',
                      help='Whether to cache preprocessed data')
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
    config.num_channels = 1
    config.num_classes = 2
    config.checkpoint_dir = str(args.output_dir)
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name="mean_teacher_highperf",
        experiment_type="mean_teacher",
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=args.num_labeled, num_validation=56)
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer
    print("Creating high-performance Mean Teacher trainer...")
    trainer = HighPerfMeanTeacherTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Define cache directory
    cache_dir = f"{args.output_dir}/cached_data"
    
    # Create experiment directory
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    trainer.train(data_paths, use_cache=args.use_cache, cache_dir=cache_dir)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x $WORK_DIR/run_mean_teacher_highperf.py
echo "Created high-performance Mean Teacher implementation"

# Run the high-performance Mean Teacher script
echo "Starting high-performance Mean Teacher training..."
echo "--- GPU Status Before Python Execution ---"
nvidia-smi # Add nvidia-smi call here
echo "-----------------------------------------"
python3 $WORK_DIR/run_mean_teacher_highperf.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --img_size 256 \
    --num_labeled 30 \
    --num_epochs 100

echo "========================================================"
echo "High-Performance Mean Teacher Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="