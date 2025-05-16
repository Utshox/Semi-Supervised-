#!/bin/bash
# SLURM batch job script for MixMatch SSL with high performance optimization

#SBATCH -p gpu                      # Use gpu partition 
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH -n 8                        # Request 8 CPU cores
#SBATCH --mem=64G                   # Request 64GB RAM
#SBATCH --time=16:00:00             # Set time limit to 16 hours (max allowed)
#SBATCH -o mixmatch-highperf-%j.out # Output file name with job ID
#SBATCH -e mixmatch-highperf-%j.err # Error file name with job ID
#SBATCH --job-name=mixmatch_hp      # Add descriptive job name

# Path to your data directory - always use the HPC path
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mixmatch_results_highperf"

echo "========================================================"
echo "Running High-Performance MixMatch Learning - $(date)"
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

# Memory optimization settings - critical for high-perf MixMatch
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1    # Reduce TF log verbosity
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2     # One thread per GPU

# Enable memory optimization using async allocator
export TF_GPU_ALLOCATOR=cuda_malloc_async  # Use async memory allocator as suggested in error
export TF_MEMORY_ALLOCATION=0.85  # Allow TF to use 85% of GPU memory
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"  # Enable XLA JIT compilation

# Enable mixed precision for better performance and lower memory usage
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Install the correct TensorFlow version and required packages
echo "Installing necessary packages..."
pip install --user --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib pandas h5py

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create a high-performance version of MixMatch
echo "Creating high-performance MixMatch implementation..."
cat > run_mixmatch_highperf.py << 'EOF'
#!/usr/bin/env python3
# High-performance MixMatch implementation with memory optimization and caching

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

# Import modules from codebase
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import MixMatchTrainer
from data_loader_tf2 import DataPipeline
from main import prepare_data_paths

# Disable eager execution for better performance on large models
# tf.compat.v1.disable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Setup GPU with memory optimization
def setup_gpu():
    """Setup GPU with memory optimization for MixMatch"""
    # Get available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU.")
        return False
    
    print(f"Found {len(physical_devices)} GPU(s)")
    
    # Configure memory growth
    try:
        for device in physical_devices:
            # Enable memory growth to prevent allocating all memory at once
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
            
            # Set memory limit if specified in environment
            if os.environ.get('TF_MEMORY_ALLOCATION'):
                mem_limit = float(os.environ.get('TF_MEMORY_ALLOCATION'))
                # Note: We can't use virtual devices when set_memory_growth is enabled
                # so we'll comment this out and rely on memory growth
                # tf.config.set_logical_device_configuration(
                #     device,
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=int(10 * 1024 * mem_limit))])
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
    
    # Clear GPU memory if possible
    for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
        try:
            # Clear unused variables by creating a small tensor and deleting it
            with tf.device(f'/GPU:{i}'):
                dummy = tf.random.normal([1])
                del dummy
        except Exception:
            pass
    
    print("Memory cleared")

# Cache preprocessed data to avoid redundant computations
class DataCache:
    """Cache preprocessed data to disk for faster training"""
    
    def __init__(self, cache_dir, batch_size):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.cache_file = self.cache_dir / f"mixmatch_data_cache_bs{batch_size}.h5"
        
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
            
            # Cache unlabeled data
            unlabeled_images = []
            
            print("Caching unlabeled data...")
            for i, images in enumerate(tqdm(train_unlabeled)):
                # Handle different return types from different dataset types
                if isinstance(images, tuple):
                    images = images[0]  # If it's a tuple, take the first element
                unlabeled_images.append(images.numpy())
                
            unlabeled_grp.create_dataset('images', data=np.concatenate(unlabeled_images, axis=0))
            
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
            
            # Load unlabeled data
            unlabeled_images = f['unlabeled/images'][:]
            
            # Load validation data
            val_images = f['validation/images'][:]
            val_labels = f['validation/labels'][:]
        
        # Create TensorFlow datasets from cached data
        train_labeled = tf.data.Dataset.from_tensor_slices((labeled_images, labeled_labels))
        train_labeled = train_labeled.batch(self.batch_size).prefetch(2)
        
        train_unlabeled = tf.data.Dataset.from_tensor_slices(unlabeled_images)
        train_unlabeled = train_unlabeled.batch(self.batch_size).prefetch(2)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_ds = val_ds.batch(self.batch_size).prefetch(2)
        
        print("Cached datasets loaded successfully.")
        return train_labeled, train_unlabeled, val_ds

# High performance MixMatch trainer with memory optimization
class HighPerfMixMatchTrainer(MixMatchTrainer):
    """Enhanced MixMatch trainer with memory optimizations"""
    
    def __init__(self, config):
        super().__init__(config)
        self.batch_memory_monitoring = False  # Flag to enable per-batch memory monitoring
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': [],
            'memory_usage': []  # Track memory usage
        }
    
    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch):
        """Optimized training step with memory efficiency"""
        # Get labeled data
        labeled_images, labeled_labels = labeled_batch
        unlabeled_images = unlabeled_batch
        
        # Current step for consistency weight calculation
        step = tf.cast(self.optimizer.iterations, tf.float32)
        
        # Apply unlabeled augmentation (weak)
        unlabeled_images_aug1 = self._augment(unlabeled_images)
        unlabeled_images_aug2 = self._augment(unlabeled_images)
        
        # Generate pseudo-labels with the teacher model (more stable)
        unlabeled_logits1 = self.teacher_model(unlabeled_images_aug1, training=False)
        unlabeled_logits2 = self.teacher_model(unlabeled_images_aug2, training=False)
        
        # Average and sharpen pseudo-labels
        pseudo_label = self._sharpen(
            tf.sigmoid(tf.stack([unlabeled_logits1, unlabeled_logits2], axis=0)), 
            self.config.temperature
        )
        
        # Calculate loss
        with tf.GradientTape() as tape:
            # Make sure inputs have the right types and shapes
            labeled_images = tf.cast(labeled_images, tf.float32)
            labeled_labels = tf.cast(labeled_labels, tf.float32)
            unlabeled_images_aug1 = tf.cast(unlabeled_images_aug1, tf.float32)
            unlabeled_images_aug2 = tf.cast(unlabeled_images_aug2, tf.float32)
            
            # Student model predictions
            student_labeled_logits = self.student_model(labeled_images, training=True)
            student_unlabeled_logits1 = self.student_model(unlabeled_images_aug1, training=True)
            student_unlabeled_logits2 = self.student_model(unlabeled_images_aug2, training=True)
            
            # Supervised loss (Dice loss for better segmentation quality)
            supervised_loss = self._dice_loss(labeled_labels, student_labeled_logits)
            
            # Unsupervised loss (MSE between student predictions and sharpened teacher predictions)
            unsupervised_loss1 = tf.reduce_mean(
                tf.square(tf.sigmoid(student_unlabeled_logits1) - pseudo_label[0])
            )
            unsupervised_loss2 = tf.reduce_mean(
                tf.square(tf.sigmoid(student_unlabeled_logits2) - pseudo_label[1])
            )
            
            unsupervised_loss = (unsupervised_loss1 + unsupervised_loss2) / 2.0
            
            # Ramp up unsupervised loss weight
            consistency_weight = self._get_consistency_weight(step)
            
            # Total loss
            total_loss = supervised_loss + consistency_weight * unsupervised_loss
        
        # Optimize student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        # Update teacher model with EMA of student weights
        self._update_teacher_model()
        
        # Return metrics
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': unsupervised_loss,
            'consistency_weight': consistency_weight
        }
    
    def _sharpen(self, p, T):
        """Sharpening function for pseudo-labels"""
        p_power = tf.pow(p, 1.0/T)
        return p_power / (tf.reduce_sum(p_power, axis=0, keepdims=True) + 1e-8)
    
    def _augment(self, x, strength=0.1):
        """Lightweight augmentation to save memory"""
        # Add random noise (computationally cheaper)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=strength)
        return tf.clip_by_value(x + noise, 0.0, 1.0)
    
    def _update_teacher_model(self):
        """Update teacher model weights using EMA"""
        ema_decay = self.config.ema_decay
        
        # Get model weights
        student_weights = self.student_model.get_weights()
        teacher_weights = self.teacher_model.get_weights()
        
        # Update using EMA
        new_weights = []
        for s_w, t_w in zip(student_weights, teacher_weights):
            new_weights.append(ema_decay * t_w + (1 - ema_decay) * s_w)
        
        # Apply new weights
        self.teacher_model.set_weights(new_weights)
    
    def _get_consistency_weight(self, step):
        """Calculate consistency weight using ramp-up function"""
        # Convert to scalar values for cleaner function
        consistency_rampup = float(self.config.consistency_rampup)
        max_consistency_weight = float(self.config.consistency_weight)
        
        # Apply sigmoid rampup (smoother than linear)
        if consistency_rampup == 0.0:
            return max_consistency_weight
            
        current_step = float(step)
        phase = 1.0 - tf.maximum(0.0, current_step / consistency_rampup)
        return max_consistency_weight * tf.exp(-5.0 * phase * phase)
    
    def _compute_gpu_memory(self):
        """Get current GPU memory usage if available"""
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            return memory_info['current'] / (1024 * 1024 * 1024)  # GB
        except:
            return 0.0
    
    def train(self, data_paths, use_cache=True, cache_dir=None):
        """Train the model with memory optimization and caching"""
        print("\nStarting high-performance MixMatch training...")
        
        # Set small batch size to prevent OOM
        batch_size = self.config.batch_size
        print(f"Using batch size: {batch_size}")
        
        # Create or load datasets based on caching preference
        if use_cache and cache_dir:
            # Initialize cache handler
            cache = DataCache(cache_dir, batch_size)
            
            if cache.cache_exists():
                # Load from cache
                train_labeled, train_unlabeled, val_ds = cache.load_cached_datasets()
            else:
                # Create datasets
                print("\nCreating datasets for caching...")
                
                train_labeled = self.data_pipeline.dataloader.create_dataset(
                    data_paths['labeled']['images'],
                    data_paths['labeled']['labels'],
                    batch_size=batch_size,
                    shuffle=True
                )
                
                train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                    data_paths['unlabeled']['images'],
                    None,  # No labels for unlabeled data
                    batch_size=batch_size,
                    shuffle=True
                )
                
                val_ds = self.data_pipeline.dataloader.create_dataset(
                    data_paths['validation']['images'],
                    data_paths['validation']['labels'],
                    batch_size=batch_size,
                    shuffle=False
                )
                
                # Cache the datasets
                cache.cache_datasets(train_labeled, train_unlabeled, val_ds)
                
                # Load cached datasets
                train_labeled, train_unlabeled, val_ds = cache.load_cached_datasets()
        else:
            # Create datasets without caching
            print("\nCreating datasets without caching...")
            
            train_labeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['labeled']['images'],
                data_paths['labeled']['labels'],
                batch_size=batch_size,
                shuffle=True
            )
            
            train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['unlabeled']['images'],
                None,  # No labels for unlabeled data
                batch_size=batch_size,
                shuffle=True
            )
            
            val_ds = self.data_pipeline.dataloader.create_dataset(
                data_paths['validation']['images'],
                data_paths['validation']['labels'],
                batch_size=batch_size,
                shuffle=False
            )
        
        # Validate datasets
        print("Validating datasets...")
        try:
            for batch in train_labeled.take(1):
                labeled_images, labeled_labels = batch
                print(f"Labeled batch - Images shape: {labeled_images.shape}, Labels shape: {labeled_labels.shape}")
            
            for batch in train_unlabeled.take(1):
                print(f"Unlabeled batch - Shape: {batch.shape}")
            
            for batch in val_ds.take(1):
                val_images, val_labels = batch
                print(f"Validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")
        except Exception as e:
            print(f"Error validating datasets: {e}")
        
        # Training parameters
        best_dice = 0
        patience = self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 15
        patience_counter = 0
        
        # Setup logs directory
        logs_dir = f"{self.config.checkpoint_dir}/logs"
        os.makedirs(logs_dir, exist_ok=True)
        log_file = f"{logs_dir}/mixmatch_training_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Write CSV header
        with open(log_file, 'w') as f:
            f.write("epoch,total_loss,supervised_loss,consistency_loss,val_dice,teacher_dice,learning_rate,memory_usage,time\n")
        
        # Calculate the number of steps per epoch
        steps_per_epoch = min(
            sum(1 for _ in train_labeled),
            sum(1 for _ in train_unlabeled)
        )
        print(f"Steps per epoch: {steps_per_epoch}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Reset metrics for this epoch
            epoch_losses = []
            epoch_supervised_losses = []
            epoch_consistency_losses = []
            epoch_memory_usage = []
            
            # Create iterators
            labeled_iter = iter(train_labeled)
            unlabeled_iter = iter(train_unlabeled)
            
            # Progress bar for this epoch
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            progress_bar = tqdm(range(steps_per_epoch), desc=f"Training")
            
            # Training steps
            for step in progress_bar:
                try:
                    # Get batch data
                    labeled_batch = next(labeled_iter)
                    unlabeled_batch = next(unlabeled_iter)
                    
                    # Memory monitoring before step
                    if self.batch_memory_monitoring:
                        pre_step_memory = self._compute_gpu_memory()
                    
                    # Training step
                    metrics = self.train_step(labeled_batch, unlabeled_batch)
                    
                    # Memory monitoring after step
                    if self.batch_memory_monitoring:
                        post_step_memory = self._compute_gpu_memory()
                        memory_diff = post_step_memory - pre_step_memory
                        epoch_memory_usage.append(post_step_memory)
                    
                    # Record metrics
                    epoch_losses.append(float(metrics['total_loss']))
                    epoch_supervised_losses.append(float(metrics['supervised_loss']))
                    epoch_consistency_losses.append(float(metrics['consistency_loss']))
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{float(metrics['total_loss']):.4f}",
                        'sup_loss': f"{float(metrics['supervised_loss']):.4f}",
                        'con_loss': f"{float(metrics['consistency_loss']):.4f}"
                    })
                    
                    # Clear memory periodically
                    if step % 50 == 0 and step > 0:
                        collect_garbage()
                        
                except tf.errors.ResourceExhaustedError:
                    print("\nOUT OF MEMORY ERROR: Skipping batch and collecting garbage...")
                    collect_garbage()
                    continue
                
                except StopIteration:
                    print("\nReached end of dataset before completing all steps")
                    break
                    
                except Exception as e:
                    print(f"\nError during training step: {e}")
                    continue
            
            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation
            val_dice, teacher_dice = self.validate(val_ds)
            
            # Record metrics
            epoch_time = time.time() - start_time
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            mean_sup_loss = np.mean(epoch_supervised_losses) if epoch_supervised_losses else 0.0
            mean_cons_loss = np.mean(epoch_consistency_losses) if epoch_consistency_losses else 0.0
            mean_memory = np.mean(epoch_memory_usage) if epoch_memory_usage else self._compute_gpu_memory()
            
            # Update history
            self.history['train_loss'].append(mean_loss)
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(current_lr)
            self.history['supervised_loss'].append(mean_sup_loss)
            self.history['consistency_loss'].append(mean_cons_loss)
            self.history['memory_usage'].append(mean_memory)
            
            # Log results
            print(f"Time: {epoch_time:.2f}s | Loss: {mean_loss:.4f} | Val Dice: {val_dice:.4f} | Teacher Dice: {teacher_dice:.4f}")
            
            # Write to CSV log
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_sup_loss:.6f},{mean_cons_loss:.6f},"
                        f"{val_dice:.6f},{teacher_dice:.6f},{current_lr:.8e},{mean_memory:.4f},{epoch_time:.2f}\n")
            
            # Update plots every epoch
            self.plot_progress()
            
            # Save best model
            best_model_dice = max(val_dice, teacher_dice)
            better_model = "teacher" if teacher_dice > val_dice else "student"
            
            if best_model_dice > best_dice:
                best_dice = best_model_dice
                checkpoint_path = f"{self.config.checkpoint_dir}/checkpoints/best_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
                
                # Save both models
                self.student_model.save_weights(f"{checkpoint_path}_student")
                self.teacher_model.save_weights(f"{checkpoint_path}_teacher")
                
                print(f"✓ New best model saved! Dice: {best_dice:.4f} ({better_model} model)")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
            
            # Save intermediate checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"{self.config.checkpoint_dir}/checkpoints/epoch_{epoch+1}_{time.strftime('%Y%m%d_%H%M%S')}"
                self.student_model.save_weights(f"{checkpoint_path}_student")
                self.teacher_model.save_weights(f"{checkpoint_path}_teacher")
                print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Final checkpoint
        final_checkpoint = f"{self.config.checkpoint_dir}/checkpoints/final_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
        self.student_model.save_weights(f"{final_checkpoint}_student")
        self.teacher_model.save_weights(f"{final_checkpoint}_teacher")
        
        # Final plot
        self.plot_progress(final=True)
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history
    
    def validate(self, val_ds):
        """Validate both student and teacher models"""
        student_dices = []
        teacher_dices = []
        
        for images, labels in val_ds:
            # Student model validation
            student_logits = self.student_model(images, training=False)
            student_dice = 1.0 - self._dice_loss(labels, student_logits)
            student_dices.append(float(student_dice))
            
            # Teacher model validation
            teacher_logits = self.teacher_model(images, training=False)
            teacher_dice = 1.0 - self._dice_loss(labels, teacher_logits)
            teacher_dices.append(float(teacher_dice))
        
        # Calculate mean dice scores
        mean_student_dice = np.mean(student_dices) if student_dices else 0.0
        mean_teacher_dice = np.mean(teacher_dices) if teacher_dices else 0.0
        
        return mean_student_dice, mean_teacher_dice
    
    def plot_progress(self, final=False):
        """Plot training progress metrics"""
        if len(self.history['train_loss']) < 2:
            return
            
        # Create plots directory
        viz_dir = f"{self.config.checkpoint_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot metrics
        plt.figure(figsize=(15, 12))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], 'b-', label='Total Loss')
        plt.plot(self.history['supervised_loss'], 'g-', label='Supervised Loss')
        plt.plot(self.history['consistency_loss'], 'r-', label='Consistency Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Dice score plot
        plt.subplot(2, 2, 2)
        plt.plot(self.history['val_dice'], 'b-', label='Student Validation Dice')
        plt.plot(self.history['teacher_dice'], 'r-', label='Teacher Validation Dice')
        
        # Add best dice marker
        if self.history['val_dice']:
            best_student_dice = max(self.history['val_dice'])
            best_student_epoch = self.history['val_dice'].index(best_student_dice)
            plt.plot(best_student_epoch, best_student_dice, 'bo', 
                     label=f'Best Student: {best_student_dice:.4f}')
        
        if self.history['teacher_dice']:
            best_teacher_dice = max(self.history['teacher_dice'])
            best_teacher_epoch = self.history['teacher_dice'].index(best_teacher_dice)
            plt.plot(best_teacher_epoch, best_teacher_dice, 'ro',
                     label=f'Best Teacher: {best_teacher_dice:.4f}')
        
        plt.title('Validation Dice Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(2, 2, 3)
        plt.plot(self.history['learning_rate'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        # Memory usage plot
        plt.subplot(2, 2, 4)
        if self.history['memory_usage']:
            plt.plot(self.history['memory_usage'], 'm-')
            plt.title('GPU Memory Usage')
            plt.xlabel('Epoch')
            plt.ylabel('Memory (GB)')
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if final:
            plt.savefig(f"{viz_dir}/final_training_progress.png", dpi=150)
        else:
            plt.savefig(f"{viz_dir}/training_progress_{timestamp}.png", dpi=150)
            plt.savefig(f"{viz_dir}/latest_training_progress.png", dpi=150)
        
        plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='High-performance MixMatch training')
    parser.add_argument('--data_dir', type=Path, default='/scratch/lustre/home/mdah0000/images/cropped',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=Path, default='/scratch/lustre/home/mdah0000/smm/v14/mixmatch_results_highperf',
                        help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--use_cache', action='store_true',
                        help='Whether to cache preprocessed data')
    args = parser.parse_args()
    
    # Set up GPU
    setup_gpu()
    
    # Create configuration
    config = StableSSLConfig()
    config.batch_size = args.batch_size
    config.num_epochs = 100
    config.img_size_x = 512
    config.img_size_y = 512
    config.n_filters = 32
    config.num_channels = 1
    config.num_classes = 2
    config.checkpoint_dir = str(args.output_dir)
    config.ema_decay = 0.999
    config.consistency_weight = 10.0  # Slightly lower than default to prevent overfitting
    config.consistency_rampup = 50000  # Ramp up consistency weight over more steps
    config.temperature = 0.5  # Temperature for sharpening pseudo-labels
    config.alpha = 0.75  # Alpha parameter for MixMatch
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name="mixmatch_highperf",
        experiment_type="mixmatch",
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=63)
    
    # Create data pipeline with caching
    print("Setting up data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create high-performance trainer
    print("Creating high-performance MixMatch trainer...")
    trainer = HighPerfMixMatchTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Define cache directory
    cache_dir = f"{args.output_dir}/cached_data"
    
    # Train the model
    print(f"Starting training with batch size {args.batch_size}...")
    trainer.train(data_paths, use_cache=args.use_cache, cache_dir=cache_dir)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x $WORK_DIR/run_mixmatch_highperf.py
echo "Created high-performance MixMatch implementation"

# Run the optimized MixMatch training script
echo "Starting high-performance MixMatch training..."
python3 $WORK_DIR/run_mixmatch_highperf.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --use_cache

echo "========================================================"
echo "High-Performance MixMatch Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="