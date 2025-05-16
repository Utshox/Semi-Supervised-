#!/bin/bash
# SLURM batch job script for MixMatch SSL with optimization for HPC

#SBATCH -p gpu                      # Use gpu partition 
#SBATCH --gres=gpu:2                # Request 2 GPUs (increased from 1)
#SBATCH -n 8                        # Request 8 CPU cores (increased from 4)
#SBATCH --mem=64G                   # Request 64GB RAM (increased from 32GB)
#SBATCH --time=10:00:00             # Set time limit to 12 hours (increased from 8)
#SBATCH -o mixmatch-fixed-%j.out    # Output file name with job ID
#SBATCH -e mixmatch-fixed-%j.err    # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"

echo "========================================================"
echo "Running Fixed MixMatch Learning - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory: $WORK_DIR"
echo "Data directory: $DATA_DIR"
echo "========================================================"

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# Enhanced memory management settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce log verbosity
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2  # One thread per GPU

# Memory optimization for TensorFlow - reduced from 0.85 to 0.7 to prevent OOM
export TF_MEMORY_ALLOCATION=0.7  # Allow TensorFlow to use 70% of GPU memory
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"  # Enable XLA JIT compilation

# Install the correct TensorFlow version if needed
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create cache directory if it doesn't exist
mkdir -p preprocessed_cache
echo "Created preprocessed_cache directory for saving processed data"

# Create a wrapper script to modify and improve MixMatch training
cat > run_mixmatch_fixed.py << 'EOF'
import sys
import os

import tensorflow as tf
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import gc
warnings.filterwarnings('ignore')

# Import your modules
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import MixMatchTrainer
from main import prepare_data_paths

# Fixed setup_gpu function that doesn't conflict with virtual devices
def setup_gpu():
    """Setup GPU for training without conflicts between memory growth and virtual devices"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU.")
        return False
    
    print(f"Found {len(physical_devices)} GPU(s)")
    
    # Don't set memory growth, just limit memory fraction
    if os.environ.get('TF_MEMORY_ALLOCATION'):
        mem_limit = float(os.environ.get('TF_MEMORY_ALLOCATION'))
        for i, device in enumerate(physical_devices):
            try:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(10 * 1024 * mem_limit))])
                print(f"Memory limit set to {mem_limit*100}% for {device}")
            except Exception as e:
                print(f"Error setting memory limit for {device}: {e}")
    
    # Enable mixed precision for better performance and lower memory usage
    if tf.__version__ >= "2.4.0":
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision policy set to mixed_float16")
        except Exception as e:
            print(f"Error setting mixed precision: {e}")
    
    return True

# Memory optimization
def optimize_memory():
    """Configure TensorFlow for optimal memory usage."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU.")
        return
    
    print(f"Found {len(physical_devices)} GPU(s)")
    
    for device in physical_devices:
        try:
            # Allow memory growth - prevents allocating all memory at once
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except:
            print(f"Error setting memory growth for {device}")
    
    # Set memory limit if needed (use 80% of available memory per GPU)
    if os.environ.get('TF_MEMORY_ALLOCATION'):
        mem_limit = float(os.environ.get('TF_MEMORY_ALLOCATION'))
        for i, device in enumerate(physical_devices):
            try:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * mem_limit)])
                print(f"Memory limit set to {mem_limit*100}% for {device}")
            except:
                print(f"Error setting memory limit for {device}")
    
    # Enable mixed precision for better performance and lower memory usage
    if tf.__version__ >= "2.4.0":
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set to mixed_float16")

# Modified initialization to setup teacher variables
def patched_init(self, config):
    """Custom initialization to properly setup variables for teacher update"""
    # Call the original __init__
    self.__original_init__(config)
    
    # Initialize teacher variables after models are created
    # This ensures that teacher weights are properly tracked by TF
    self.teacher_vars = self.teacher_model.trainable_variables
    self.student_vars = self.student_model.trainable_variables
    
    print("Initialized teacher and student variables for TF-compatible updates")

# Graph-compatible teacher update method
def graph_compatible_update_teacher(self):
    """Update teacher model weights using TF operations compatible with graph execution.
    This avoids calling get_weights() and set_weights() in graph mode.
    """
    # EMA decay rate
    alpha = tf.constant(self.config.ema_decay, dtype=tf.float32)
    
    # Update each teacher variable using TF operations
    for teacher_var, student_var in zip(self.teacher_vars, self.student_vars):
        new_teacher_val = alpha * teacher_var + (1.0 - alpha) * student_var
        teacher_var.assign(new_teacher_val)
    
    return

# Override train method in MixMatchTrainer to add progress bar and fix issues
def patched_train(self, data_paths):
    """Main training loop with progress bar and memory optimization"""
    print("\nStarting MixMatch training...")
    
    # Create datasets with proper validation
    print("\nCreating and validating datasets...")
    try:
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        unlabeled_ds = self.data_pipeline.dataloader.create_unlabeled_dataset(
            data_paths['unlabeled']['images'],
            batch_size=self.config.batch_size
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Validate datasets
        print("\nValidating training dataset...")
        for i, batch in enumerate(train_ds.take(1)):
            print(f"Training batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
            
        print("\nValidating unlabeled dataset...")
        for i, batch in enumerate(unlabeled_ds.take(1)):
            print(f"Unlabeled batch shapes - Images: {batch.shape}")
            
        print("\nValidating validation dataset...")
        for i, batch in enumerate(val_ds.take(1)):
            print(f"Validation batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
    except Exception as e:
        raise ValueError(f"Error setting up datasets: {e}")
    
    best_dice = 0
    patience = self.config.early_stopping_patience
    patience_counter = 0
    
    labeled_steps = len(list(train_ds))
    unlabeled_steps = len(list(unlabeled_ds))
    steps_per_epoch = min(labeled_steps, unlabeled_steps)
    print(f"Steps per epoch: {steps_per_epoch} (limited by smaller of labeled/unlabeled sets)")
    
    # Initial teacher model update before training
    # Call this outside the graph using the non-graph version to initialize properly
    self.update_teacher_non_graph()
    
    for epoch in range(self.config.num_epochs):
        # Run garbage collection at start of each epoch to free memory
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass
        
        start_time = time.time()
        
        # Training
        epoch_losses = []
        supervised_losses = []
        consistency_losses = []
        
        # Create iterators
        train_iter = iter(train_ds)
        unlabeled_iter = iter(unlabeled_ds)
        
        # Training steps with progress bar
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Training")
        
        for step in progress_bar:
            try:
                labeled_batch = next(train_iter)
                unlabeled_batch = next(unlabeled_iter)
                
                metrics = self.train_step(labeled_batch, unlabeled_batch)
                epoch_losses.append(float(metrics['total_loss']))
                supervised_losses.append(float(metrics['supervised_loss']))
                consistency_losses.append(float(metrics['consistency_loss']))
                
                # Update progress bar with real-time metrics
                progress_bar.set_postfix({
                    "loss": f"{float(metrics['total_loss']):.4f}", 
                    "sup_loss": f"{float(metrics['supervised_loss']):.4f}",
                    "cons_loss": f"{float(metrics['consistency_loss']):.4f}"
                })
                
                # Periodically clear memory to prevent OOM
                if step % 20 == 0:
                    gc.collect()
                    tf.keras.backend.clear_session()
            except StopIteration:
                print("Reached end of dataset before completing epoch")
                break
            except tf.errors.ResourceExhaustedError:
                print("\nWARNING: Out of memory during training step. Skipping step...")
                gc.collect()
                tf.keras.backend.clear_session()
                continue
            except Exception as e:
                print(f"\nError during training step: {e}")
                continue
        
        # Clear memory before validation
        gc.collect()
        
        # Validation
        try:
            val_dice, teacher_dice = self.validate(val_ds)
        except tf.errors.ResourceExhaustedError:
            print("\nWARNING: Out of memory during validation. Using last best values...")
            val_dice = best_dice
            teacher_dice = 0
        
        # Update history
        self.history['train_loss'].append(np.mean(epoch_losses))
        self.history['val_dice'].append(val_dice)
        self.history['teacher_dice'].append(teacher_dice)
        self.history['supervised_loss'].append(np.mean(supervised_losses))
        self.history['consistency_loss'].append(np.mean(consistency_losses))
        self.history['learning_rate'].append(float(self.lr_schedule(self.optimizer.iterations)))
        
        # Logging
        print(f"Time: {time.time() - start_time:.2f}s | Loss: {np.mean(epoch_losses):.4f} | "
              f"Val Dice: {val_dice:.4f} | Teacher Dice: {teacher_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            try:
                self.save_checkpoint('best_mixmatch_model')
                print(f"✓ New best model saved! Dice: {best_dice:.4f}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break
        
        # Plot progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            try:
                self.plot_progress()
            except Exception as e:
                print(f"Error plotting progress: {e}")
    
    print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
    try:
        self.plot_progress()  # Final plot
    except Exception as e:
        print(f"Error plotting final progress: {e}")
    return self.history

# Non-graph version of update_teacher for initialization and debugging
def update_teacher_non_graph(self):
    """Update the teacher model using EMA of student weights.
    This version is used outside of TF graph execution (e.g., initialization).
    """
    # Implementation using EMA (Exponential Moving Average)
    alpha = self.config.ema_decay  # EMA decay rate
    
    # Get teacher and student model weights
    student_weights = self.student_model.get_weights()
    teacher_weights = self.teacher_model.get_weights()
    
    # Update teacher weights using EMA
    updated_weights = []
    for s_w, t_w in zip(student_weights, teacher_weights):
        updated_weights.append(alpha * t_w + (1 - alpha) * s_w)
    
    # Set the updated weights to the teacher model
    self.teacher_model.set_weights(updated_weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--experiment_name', type=str, default='mixmatch_ssl_fixed',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    args = parser.parse_args()

    # Setup GPU with memory limits, but without memory growth
    # Only use our fixed setup_gpu function, not both
    gpu_available = setup_gpu()
    
    config = StableSSLConfig()
    
    # Set dimensions and hyperparameters
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = args.batch_size
    config.num_epochs = 50  # Adjust as needed
    config.early_stopping_patience = 15  # More patience for semi-supervised
    config.ema_decay = 0.999  # EMA decay rate for teacher update
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='mixmatch',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=63)
    
    # Create trainer with patched initialization
    trainer = MixMatchTrainer(config)
    
    # Save original init method and replace with our patched version
    trainer.__original_init__ = trainer.__init__
    trainer.__init__ = patched_init.__get__(trainer, MixMatchTrainer)
    
    # Re-initialize to apply our patches
    trainer.__init__(config)
    
    # Add the graph-compatible update_teacher method
    trainer.update_teacher = graph_compatible_update_teacher.__get__(trainer, MixMatchTrainer)
    
    # Add non-graph version for initialization
    trainer.update_teacher_non_graph = update_teacher_non_graph.__get__(trainer, MixMatchTrainer)
    
    # Apply patched train method
    trainer.train = patched_train.__get__(trainer, MixMatchTrainer)

    # Print experiment info
    print("\nExperiment Configuration:")
    print(f"Training Type: mixmatch (fixed)")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Learning rate: {trainer.lr_schedule.get_config()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    print(f"EMA decay rate: {config.ema_decay}")
    
    # Create experiment directory
    exp_dir = experiment_config.get_experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / 'config.txt', 'w') as f:
        f.write(f"Training Type: mixmatch (fixed)\n")
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Timestamp: {experiment_config.timestamp}\n")
        f.write(f"Learning Rate Config: {trainer.lr_schedule.get_config()}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Image Size: {config.img_size_x}x{config.img_size_y}\n")
        f.write(f"EMA Decay Rate: {config.ema_decay}\n")

    # Train
    history = trainer.train(data_paths)
    
    # Save final results
    results_dir = experiment_config.get_experiment_dir()
    np.save(results_dir / 'training_history.npy', history)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
EOF

# Run with a batch size of 4 and memory optimizations
echo "Running MixMatch (Fixed) with batch size 4 and enhanced memory optimization"
python3 $WORK_DIR/run_mixmatch_fixed.py \
  --experiment_name mixmatch_ssl_fixed \
  --data_dir $DATA_DIR \
  --batch_size 4

echo "========================================================"
echo "MixMatch Learning (Fixed) completed - $(date)"
echo "========================================================"
echo "Results are located in: $WORK_DIR/mixmatch_results"
echo "========================================================="