#!/bin/bash
# SLURM batch job script for supervised learning with enhancements aligned with Mean Teacher

#SBATCH -p gpu                        # Use gpu partition (from Mean Teacher)
#SBATCH --gres=gpu:2                  # Request 2 GPUs (from Mean Teacher)
#SBATCH -n 8                          # Request 8 CPU cores (from Mean Teacher)
#SBATCH --mem=64G                     # Request 64GB RAM (from Mean Teacher)
#SBATCH --time=00:20:00               # Time limit (from Mean Teacher)
#SBATCH -o supervised-adv-%j.out      # Output file name (pattern from Mean Teacher)
#SBATCH -e supervised-adv-%j.err      # Error file name (pattern from Mean Teacher)
#SBATCH --mail-type=END,FAIL          # Send email when job ends or fails (from Mean Teacher)
#SBATCH --job-name=pancreas_sup_adv   # Job name (pattern from Mean Teacher)

# Path to your data directory - using your HPC paths
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2/" # ALIGNED with Mean Teacher
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/supervised_advanced_results" # ALIGNED with Mean Teacher pattern

echo "========================================================"
echo "Running Advanced Supervised Learning - $(date)" # ALIGNED with Mean Teacher
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
mkdir -p $OUTPUT_DIR/visualizations # ALIGNED with Mean Teacher (was plots)

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings (from Mean Teacher)
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Memory optimization settings (from Mean Teacher)
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_GPU_HOST_MEM_LIMIT_IN_MB=4096
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Find Python executable (from Mean Teacher)
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

# Install required packages (from Mean Teacher)
echo "Installing required packages..."
$PYTHON_CMD -m pip install --quiet "tensorflow[and-cuda]==2.15.0" tqdm matplotlib psutil scipy albumentations

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# ========================================================================
# CREATING PYTHON FILE: This section creates the Python script file that
# will be executed. The Python code is embedded directly in this SLURM script.
# (Content of run_supervised_enhanced.py is preserved from original)
# ========================================================================
echo "Creating Python script file: $WORK_DIR/run_supervised_enhanced.py"

cat > $WORK_DIR/run_supervised_enhanced.py << 'EOF'
import sys
import os
import numpy as np
import gc
import json # Added for metadata saving

# Add the directory containing your module to Python's path
# sys.path.append('/scratch/lustre/home/mdah0000/smm/v14') # Assuming modules are in the WORK_DIR

import tensorflow as tf
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import your modules
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import SupervisedTrainer
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg  # Import PancreasSeg model directly
from main import prepare_data_paths

print("TensorFlow version:", tf.__version__)

# Our own GPU setup function that won't conflict with virtual devices
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

# Add a custom augmented dataset creation function to handle the shape mismatch
def create_augmented_dataset(dataloader, image_paths, label_paths, batch_size=4, shuffle=True, augment=True):
    """Create a dataset that properly handles num_classes mismatch with smaller batches"""
    print(f"Creating dataset with batch size {batch_size}, shuffle={shuffle}, augment={augment}")
    
    # Create the base dataset
    dataset = dataloader.create_dataset(
        image_paths,
        label_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=False  # We'll handle augmentation separately
    )
    
    if augment:
        # Simple augmentation that preserves memory
        def augment_custom(image, label):
            # Ensure inputs are tensors with proper shapes
            image = tf.cast(image, tf.float32)
            label = tf.cast(label, tf.float32)
            
            # Random flip (single operation to save memory)
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(label)

            # Normalize image values to [0,1] range for better learning
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
            
        # Apply our custom augmentation
        dataset = dataset.map(
            augment_custom,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch for better performance but limit prefetching to avoid OOM
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Use AUTOTUNE for prefetch buffer size
    
    return dataset

class EnhancedSupervisedTrainer(SupervisedTrainer):
    """Enhanced version of supervised trainer with memory optimization"""
    
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        self.output_dir = Path(output_dir)
        self._setup_model()  # Override parent's model setup
        self._setup_training_params()
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
    def _setup_model(self):
        """Setup the U-Net model with memory-efficient parameters"""
        print("Setting up enhanced U-Net model...")
        
        # Create a more memory efficient model by reducing filters
        self.config.n_filters = 32  # Increased from 16 but still much less than original 64
        
        # Create the model directly using PancreasSeg
        self.model = PancreasSeg(self.config)
        
        # Initialize model with dummy input (small size for memory efficiency)
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        _ = self.model(dummy_input)
        
        print(f"Model created with input shape: ({self.config.img_size_x}, {self.config.img_size_y}, {self.config.num_channels})")
        print(f"Initial filters: {self.config.n_filters}")
        
        # Print model summary to understand the memory footprint
        total_params = self.model.count_params()
        print(f"Total model parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
        
    def _dice_loss(self, y_true, y_pred):
        """Memory-efficient dice loss function with float32 stability"""
        policy = tf.keras.mixed_precision.global_policy()
        y_true = tf.cast(y_true, policy.compute_dtype)
        y_pred = tf.cast(y_pred, policy.compute_dtype) # Logits

        y_pred_f32 = tf.cast(y_pred, tf.float32)
        y_true_f32 = tf.cast(y_true, tf.float32)

        # Slice first
        if tf.shape(y_true_f32)[-1] > 1:
            y_true_f32 = y_true_f32[..., 1:2] # Assuming channel 1 is the target mask
        if tf.shape(y_pred_f32)[-1] > 1:
            # y_pred contains logits for each class. Channel 0 for background, Channel 1 for foreground.
            # We need the logits for the foreground class.
            y_pred_f32 = y_pred_f32[..., 1:2] # CORRECTED: Use channel 1 for foreground logits
        # If y_pred_f32 was already (B, H, W, 1), it means logits for foreground, so no change needed in the 'if' block above.

        y_pred_sigmoid_f32 = tf.sigmoid(y_pred_f32)
        epsilon = 1e-5

        # Calculate intersection and sum of pixels per sample
        intersection = tf.reduce_sum(y_true_f32 * y_pred_sigmoid_f32, axis=[1, 2, 3]) # Shape (B,)
        sum_true_pixels = tf.reduce_sum(y_true_f32, axis=[1, 2, 3])                   # Shape (B,)
        sum_pred_pixels = tf.reduce_sum(y_pred_sigmoid_f32, axis=[1, 2, 3])           # Shape (B,)

        numerator_per_sample = 2.0 * intersection
        denominator_per_sample = sum_true_pixels + sum_pred_pixels

        # Calculate Dice coefficient per sample
        # Ensure (0 + eps) / (0 + eps) = 1, not 0 or NaN
        dice_per_sample = (numerator_per_sample + epsilon) / (denominator_per_sample + epsilon) # Shape (B,)

        dice_coef = tf.reduce_mean(dice_per_sample)
        loss = 1.0 - dice_coef

        return tf.cast(loss, policy.compute_dtype)
        
    def _setup_training_params(self):
        """Setup training parameters with improved learning rate scheduler and loss function"""
        # Define a proper LearningRateSchedule class that TensorFlow understands
        class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, max_lr, min_lr, total_steps):
                super(OneCycleLR, self).__init__()
                self.max_lr = tf.cast(max_lr, tf.float32)
                self.min_lr = tf.cast(min_lr, tf.float32)
                self.total_steps = tf.cast(total_steps, tf.float32)
                self.half_steps = self.total_steps / 2
                
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                
                # First half: linear increase
                condition = step <= self.half_steps
                
                # Calculate the learning rate during ramp-up phase
                ramp_up_lr = self.min_lr + (self.max_lr - self.min_lr) * (step / self.half_steps)
                
                # Calculate the learning rate during cosine decay phase
                decay_steps = step - self.half_steps
                cosine_decay = 0.5 * (1 + tf.cos(
                    tf.constant(np.pi) * decay_steps / (self.total_steps - self.half_steps)))
                ramp_down_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
                
                # Return either ramp-up or ramp-down learning rate based on step
                return tf.cond(condition, lambda: ramp_up_lr, lambda: ramp_down_lr)
        
            def get_config(self):
                return {
                    "max_lr": self.max_lr,
                    "min_lr": self.min_lr,
                    "total_steps": self.total_steps
                }
        
        # Calculate total steps for entire training using config batch size
        # Assuming 225 labeled images as per previous context
        num_labeled_images = 225 
        steps_per_epoch = num_labeled_images // self.config.batch_size + (1 if num_labeled_images % self.config.batch_size else 0)
        total_steps = self.config.num_epochs * steps_per_epoch
        
        # Create learning rate schedule with properly configured low learning rates
        max_lr = 5e-4  # More reasonable maximum learning rate
        min_lr = 1e-5  # Higher minimum to prevent stagnation
        
        # Create an instance of our custom learning rate scheduler
        self.lr_schedule = OneCycleLR(
            max_lr=max_lr,
            min_lr=min_lr,
            total_steps=total_steps
        )
        
        # Print clear information about learning rate schedule
        print(f"Using batch size: {self.config.batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Learning rate schedule: One-cycle from {min_lr} to {max_lr} and back to {min_lr}")
        
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
        
        # Setup loss function
        self.loss_fn = self._combined_loss
    
    @tf.function
    def train_step(self, images, labels):
        """Modified training step with gradient clipping and loss scaling"""
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            # Ensure logits are float32 for loss calculation if using mixed precision
            # This is often handled automatically by TF layers, but explicit cast can help
            # logits_f32 = tf.cast(logits, tf.float32) 
            loss = self.loss_fn(labels, logits) # Use original logits
            # --- Apply loss scaling if using mixed precision ---
            scaled_loss = self.optimizer.get_scaled_loss(loss) if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else loss
        
        # Compute gradients using scaled loss
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        # --- Unscale gradients ---
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients) if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else scaled_gradients
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate dice score (using the unscaled loss)
        # Note: Dice loss might not be exactly (1 - dice_score) if BCE is included
        # Re-calculate dice metric for reporting if needed, or use loss directly
        dice_score = 1.0 - self._dice_loss(labels, logits) # Use dice loss part for score
        
        return loss, dice_score
    
    def _dice_metric(self, y_true, y_pred): # Removed epsilon from parameters
        """Memory-efficient dice metric function for evaluation, aligned with loss probability calculation."""
        # y_true is one-hot (Batch, H, W, NumClasses=2)
        # y_pred is logits (Batch, H, W, NumClasses=2)

        # Get foreground channel from true labels (channel 1)
        if tf.shape(y_true)[-1] > 1:
            y_true_fg = y_true[..., 1:2]
        else: # Should not happen if y_true is one-hot with num_classes=2
            y_true_fg = y_true
        y_true_f32 = tf.cast(y_true_fg, tf.float32) # Shape: (B, H, W, 1)

        # Cast predictions (logits) to float32
        y_pred_logits_f32 = tf.cast(y_pred, tf.float32)

        # Select foreground logits (channel 1) - Model outputs [logit_bg, logit_fg]
        # This assumes num_classes is 2 and channel 1 is foreground.
        if tf.shape(y_pred_logits_f32)[-1] > 1:
            foreground_logits = y_pred_logits_f32[..., 1:2] # Shape: (B, H, W, 1)
        else: # This case would be for a model outputting a single logit for the foreground class
            foreground_logits = y_pred_logits_f32
        
        # Apply sigmoid to get foreground probability (consistent with loss functions)
        y_pred_prob_fg = tf.sigmoid(foreground_logits) # Shape: (B, H, W, 1)

        # Threshold probability to get binary mask for the foreground class
        y_pred_binary_fg = tf.cast(y_pred_prob_fg > 0.5, tf.float32) # Shape: (B, H, W, 1)

        epsilon = 1e-5 # Standardized epsilon, consistent with _dice_loss
        # Calculate intersection and sum of pixels per sample
        intersection = tf.reduce_sum(y_true_f32 * y_pred_binary_fg, axis=[1, 2, 3]) # Shape (B,)
        sum_true = tf.reduce_sum(y_true_f32, axis=[1, 2, 3])                         # Shape (B,)
        sum_pred = tf.reduce_sum(y_pred_binary_fg, axis=[1, 2, 3])                   # Shape (B,)

        numerator_per_sample = 2.0 * intersection
        denominator_per_sample = sum_true + sum_pred
        
        # Calculate Dice coefficient per sample
        # (numerator + epsilon) / (denominator + epsilon) handles 0/0 case as 1.0
        dice_per_sample = (numerator_per_sample + epsilon) / (denominator_per_sample + epsilon)

        return tf.reduce_mean(dice_per_sample) # Mean Dice over the batch

    def validate(self, val_dataset):
        """Memory-efficient validation step"""
        val_dice_scores = []
        
        for images, labels in val_dataset:
            # Get predictions
            logits = self.model(images, training=False)
            
            # Use the dedicated metric function, not the loss function
            dice_score = self._dice_metric(labels, logits)
            val_dice_scores.append(float(dice_score))
        
        # Return mean dice score if we have values, otherwise 0
        if val_dice_scores:
            mean_dice = float(sum(val_dice_scores) / len(val_dice_scores))
            # Print information about the best dice score in this validation batch
            best_dice = max(val_dice_scores)
            print(f"Mean validation Dice: {mean_dice:.4f}, Best batch Dice: {best_dice:.4f}")
            return mean_dice
        else:
            print("WARNING: No validation batches processed!")
            return 0.0
    
    def save_checkpoint(self, name='checkpoint'):
        """Save a model checkpoint with more metadata"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_path = checkpoint_dir / f"{name}_{timestamp}"
        
        # Save the model weights
        self.model.save_weights(str(checkpoint_path))
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'epoch': len(self.history['val_dice']),
            'best_val_dice': max(self.history['val_dice']) if self.history['val_dice'] else 0,
            'current_val_dice': self.history['val_dice'][-1] if self.history['val_dice'] else 0,
            'learning_rate': float(self.lr_schedule(self.optimizer.iterations)),
            'img_size': (self.config.img_size_x, self.config.img_size_y),
            'num_classes': self.config.num_classes,
            'n_filters': self.config.n_filters
        }
        
        # Save metadata as JSON
        import json
        with open(f"{checkpoint_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def train(self, data_paths):
        """Memory-efficient training loop with progress tracking"""
        print("\nStarting memory-optimized training...")
        
        # --- Use batch size from config ---
        batch_size = self.config.batch_size 
        print(f"Using batch size: {batch_size}")
        
        # --- Remove LR schedule recalculation - already done in _setup_training_params ---
        
        # Create datasets with configured batch size
        train_ds = create_augmented_dataset(
            self.data_pipeline.dataloader,
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=batch_size, # Use config batch size
            shuffle=True,
            augment=True
        )
        
        val_ds = create_augmented_dataset(
            self.data_pipeline.dataloader,
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=batch_size, # Use config batch size
            shuffle=False,
            augment=False
        )
        
        # Create experiment log file
        log_file = self.output_dir / 'logs' / f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(log_file, 'w') as f:
            f.write('epoch,loss,val_dice,learning_rate,time\n')
        
        # Log training config
        print(f"\nTraining Configuration:")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Log file: {log_file}")
        print(f"- Batch size: {self.config.batch_size}") # Use config value
        print(f"- Epochs: {self.config.num_epochs}")
        print(f"- Early stopping patience: {self.config.early_stopping_patience}")
        print(f"- Image size: {self.config.img_size_x}x{self.config.img_size_y}")
        print(f"- Number of classes: {self.config.num_classes}")
        
        # Setup for training
        best_dice = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        checkpoint_freq = 5
        
        # --- Optional: Setup TensorFlow Profiler ---
        # profiler_logdir = self.output_dir / 'logs' / 'profiler'
        # profiler_logdir.mkdir(exist_ok=True, parents=True)
        # print(f"Profiler logs will be saved to: {profiler_logdir}")
        # --- End Profiler Setup ---

        # Run a quick test batch to check for errors before main training loop
        for batch in train_ds.take(1):
            images, labels = batch
            print(f"Test batch - images: {images.shape}, labels: {labels.shape}")
            print(f"Memory usage after loading batch: {tf.config.experimental.get_memory_info('GPU:0')['current']/1e9:.2f}GB" 
                  if hasattr(tf.config.experimental, 'get_memory_info') else "Memory info not available")
            
            # Run a test training step
            try:
                with tf.device('/GPU:0'):
                    loss, dice = self.train_step(images, labels)
                    print(f"Test training step successful - Loss: {float(loss):.4f}, Dice: {float(dice):.4f}")
            except Exception as e:
                print(f"Error during test training step: {e}")
                print("Will attempt to continue with training anyway")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # --- Optional: Start Profiler for specific epochs ---
            # if epoch == 1 or epoch == 5: # Profile early and later epochs
            #     tf.profiler.experimental.start(str(profiler_logdir))
            # --- End Profiler Start ---

            # Training
            epoch_losses = []
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            progress_bar = tqdm(train_ds, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                images, labels = batch
                
                # Run training step
                loss, dice = self.train_step(images, labels)
                epoch_losses.append(float(loss))
                progress_bar.set_postfix({"loss": f"{float(loss):.4f}", "dice": f"{float(dice):.4f}"})

                # --- Optional: Add profiler trace for a few steps ---
                # if epoch == 1 and batch_idx < 5:
                #      tf.summary.trace_on(graph=True, profiler=True)
                #      # Rerun step with tracing
                #      loss, dice = self.train_step(images, labels)
                #      with tf.summary.create_file_writer(str(profiler_logdir)).as_default():
                #          tf.summary.trace_export(name=f"train_step_epoch{epoch}_batch{batch_idx}", step=batch_idx, profiler_outdir=str(profiler_logdir))
                #      tf.summary.trace_off()
                # --- End Profiler Trace ---

            # --- Optional: Stop Profiler ---
            # if epoch == 1 or epoch == 5:
            #     tf.profiler.experimental.stop()
            # --- End Profiler Stop ---

            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation with careful memory management
            val_dice = self.validate(val_ds)
            
            # Force garbage collection again
            collect_garbage()
                
            # Update history
            epoch_time = time.time() - start_time
            epoch_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            self.history['train_loss'].append(epoch_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )
            
            # Logging
            print(f"Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {self.history['learning_rate'][-1]:.8e}")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_loss:.6f},{val_dice:.6f},{self.history['learning_rate'][-1]:.8e},{epoch_time:.2f}\n")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_supervised_enhanced')
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
            
            # Plot progress every 5 epochs with memory cleanup before and after
            if (epoch + 1) % 5 == 0:
                collect_garbage()
                self.plot_progress()
                collect_garbage()
        
        # Final checkpoint
        self.save_checkpoint('final')
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        self.plot_progress()
        
        return self.history
    
    def plot_progress(self):
        """Plot training progress with memory-efficient approach"""
        if len(self.history['train_loss']) < 2:
            return

        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Use a single figure to save memory
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot Dice score
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_dice'])
        plt.title(f'Validation Dice (Best: {max(self.history["val_dice"]):.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(self.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/progress_{timestamp}.png')
        plt.savefig(f'{plots_dir}/latest_progress.png')  # Overwrite for latest
        plt.close()  # Close to free memory

    def _combined_loss(self, y_true, y_pred):
        """Combined loss function using dice loss and weighted BCE loss."""
        # --- Cast inputs based on policy ---
        policy = tf.keras.mixed_precision.global_policy()
        y_true = tf.cast(y_true, policy.compute_dtype)
        y_pred = tf.cast(y_pred, policy.compute_dtype) # Logits

        # --- Calculate Dice Loss (using float32 internally for stability) ---
        dice_loss = self._dice_loss(y_true, y_pred) # Pass original y_true and logits

        # --- Calculate Weighted BCE Loss (using float32 internally) ---
        y_true_f32 = tf.cast(y_true, tf.float32)
        y_pred_f32 = tf.cast(y_pred, tf.float32) # Logits as float32

        # For binary segmentation with one-hot encoding, get the foreground class
        if tf.shape(y_true_f32)[-1] > 1:
            y_true_f32 = y_true_f32[..., 1:2] # Shape [B, H, W, 1]
        
        if tf.shape(y_pred_f32)[-1] > 1:
            # y_pred contains logits for each class. Channel 0 for background, Channel 1 for foreground.
            # We need the logits for the foreground class for BCE.
            y_pred_f32 = y_pred_f32[..., 1:2] # CORRECTED: Use channel 1 for foreground logits
            # If y_pred_f32 was already (B, H, W, 1), it means logits for foreground.
            
        # Apply sigmoid to predictions for BCE (in float32)
        y_pred_sigmoid_f32 = tf.sigmoid(y_pred_f32) # Shape [B, H, W, 1]
        
        epsilon = 1e-5
        pos_weight = 5.0
        
        # Calculate BCE loss in float32
        bce = tf.keras.losses.binary_crossentropy(
            y_true_f32, 
            y_pred_sigmoid_f32, 
            from_logits=False # Sigmoid already applied
        ) # Output shape is typically [B, H, W]
        
        # Apply positive weight
        weight_map = tf.where(tf.equal(y_true_f32, 1.0), pos_weight, 1.0) # Shape [B, H, W, 1]
        
        # --- Explicitly reshape weight_map to remove the last dimension ---
        current_shape = tf.shape(weight_map)
        # Ensure it has 4 dimensions before reshaping to 3
        if len(weight_map.shape) == 4:
             weight_map = tf.reshape(weight_map, [current_shape[0], current_shape[1], current_shape[2]]) # Reshape to [B, H, W]
        # If weight_map is already [B, H, W], do nothing

        weighted_bce = bce * weight_map # Now shapes should match: [B, H, W] * [B, H, W]
        bce_loss_f32 = tf.reduce_mean(weighted_bce)
        
        # --- Combine losses (both should be float32 at this point) ---
        combined_f32 = 0.8 * tf.cast(dice_loss, tf.float32) + 0.2 * bce_loss_f32
        
        # --- Cast final combined loss back to policy dtype ---
        return tf.cast(combined_f32, policy.compute_dtype)

# Custom learning rate scheduler for warmup followed by target schedule
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, initial_learning_rate, target_learning_rate, warmup_steps):
        """
        Args:
          initial_learning_rate: Low learning rate to start training with
          target_learning_rate: Target learning rate or schedule after warmup
          warmup_steps: Number of steps to perform warmup for
        """
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.target_learning_rate = target_learning_rate
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        """
        Calculates learning rate during warmup and after warmup.
        
        Args:
          step: Current training step
          
        Returns:
          Learning rate at this step
        """
        step = tf.cast(step, tf.float32)
        
        # Calculate warmup rate - linear ramp from initial to target
        warmup_percent = tf.math.minimum(1.0, step / self.warmup_steps)
        
        if isinstance(self.target_learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            # If target is a schedule, get its value at current step
            target_lr = self.target_learning_rate(step)
        else:
            # If target is a constant value
            target_lr = self.target_learning_rate
            
        # Linear warmup
        warmup_lr = self.initial_learning_rate + warmup_percent * (target_lr - self.initial_learning_rate)
        
        # Return warmup_lr during warmup, otherwise return target_lr
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: target_lr
        )
        
    def get_config(self):
        """Returns configuration of the learning rate schedule."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "target_learning_rate": self.target_learning_rate,
            "warmup_steps": self.warmup_steps,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--output_dir', type=Path, required=False,
                      help='Path to output directory',
                      default='/scratch/lustre/home/mdah0000/smm/v14/supervised_results') # Default to corrected path
    parser.add_argument('--experiment_name', type=str, default='supervised_enhanced_v2', # Updated experiment name
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=8, # Default batch size (can be overridden)
                      help='Batch size for training (per replica)')
    parser.add_argument('--num_epochs', type=int, default=100, # Increased epochs
                      help='Number of epochs to train')
    parser.add_argument('--img_size', type=int, default=256, # Reduced image size
                      help='Image size (pixels) for x and y dimensions')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with more verbose output')
    args = parser.parse_args()

    # Use our custom GPU setup instead of importing from main
    custom_setup_gpu()
    
    # Create config with memory-efficient settings
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size # Use batch size from args
    config.img_size_x = args.img_size # Use img_size from args
    config.img_size_y = args.img_size # Use img_size from args
    config.n_filters = 32    # Keep filters at 32
    config.num_classes = 2
    config.early_stopping_patience = 15
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='supervised',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    # Use all 225 labeled images for supervised training
    data_paths = prepare_data_paths(args.data_dir, num_labeled=225, num_validation=56) 
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer
    print("Creating memory-optimized trainer...")
    trainer = EnhancedSupervisedTrainer(config, args.output_dir)
    trainer.data_pipeline = data_pipeline
    
    # Set debug flag if requested
    if args.debug:
        trainer.debug = True
    
    # Create experiment directory (using output_dir from args)
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    trainer.train(data_paths)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

# ========================================================================
# PYTHON EXECUTION: This section runs the generated Python script.
# ========================================================================
echo "Starting enhanced supervised training with aligned parameters..."
$PYTHON_CMD $WORK_DIR/run_supervised_enhanced.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --experiment_name "supervised_advanced_final" \
    --batch_size 8 \
    --num_epochs 100 \
    --img_size 256

echo "Training completed"