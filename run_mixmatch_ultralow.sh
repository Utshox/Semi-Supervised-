#!/bin/bash
# SLURM batch job script for Ultra Memory-Efficient MixMatch SSL

#SBATCH -p gpu                      # Use gpu partition 
#SBATCH --gres=gpu:1                # Request 1 GPU (using more can increase memory pressure)
#SBATCH -n 4                        # Request 4 CPU cores (reduced from 8)
#SBATCH --mem=32G                   # Request 32GB RAM (reduced from 64G)
#SBATCH --time=16:00:00             # Set time limit to 16 hours (max allowed)
#SBATCH -o mixmatch-ultralow-%j.out # Output file name with job ID
#SBATCH -e mixmatch-ultralow-%j.err # Error file name with job ID
#SBATCH --job-name=mixmatch_ul      # Add descriptive job name

# Path to your data directory - always use the HPC path
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mixmatch_results_ultralow"

echo "========================================================"
echo "Running Ultra Memory-Efficient MixMatch - $(date)"
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

# Memory optimization settings - critical for ultra-low memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1  # Single GPU thread

# Enable memory optimization using async allocator
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_MEMORY_ALLOCATION=0.7  # Use only 70% of GPU memory
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Enable mixed precision for better performance and lower memory usage
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Install the correct TensorFlow version and required packages
echo "Installing necessary packages..."
pip install --user --quiet "tensorflow==2.15.0.post1" tqdm matplotlib pandas h5py

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create an ultra memory-efficient implementation of MixMatch
echo "Creating ultra memory-efficient MixMatch implementation..."
cat > run_mixmatch_ultralow.py << 'EOF'
#!/usr/bin/env python3
# Ultra Memory-Efficient MixMatch implementation

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

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Memory-optimized GPU setup
def setup_gpu():
    """Setup GPU with extreme memory optimization for MixMatch"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU.")
        return False
    
    print(f"Found {len(physical_devices)} GPU(s)")
    
    try:
        for device in physical_devices:
            # Enable memory growth to prevent allocating all memory at once
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
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

# Aggressive garbage collection
def collect_garbage(force_release=False):
    """Force garbage collection and clear TF memory more aggressively"""
    tf.keras.backend.clear_session()
    gc.collect()
    
    if force_release:
        # More aggressive memory clearing
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
        except:
            pass
    
    print("Memory cleared")

# Streamlined dataset caching
class SimpleDataCache:
    """Simplified data caching for minimal memory overhead"""
    
    def __init__(self, cache_dir, img_size=256):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.img_size = img_size
        self.labeled_cache = self.cache_dir / f"mixmatch_labeled_{img_size}.npz"
        self.unlabeled_cache = self.cache_dir / f"mixmatch_unlabeled_{img_size}.npz"
        self.val_cache = self.cache_dir / f"mixmatch_val_{img_size}.npz"
    
    def cache_exists(self):
        return (
            self.labeled_cache.exists() and 
            self.unlabeled_cache.exists() and 
            self.val_cache.exists()
        )
    
    def prepare_dataset(self, data_paths, dp):
        """Prepare and cache datasets with downsampling for memory efficiency"""
        # Create datasets at full resolution
        print(f"Creating original datasets...")
        
        # Create labeled dataset
        labeled_ds = dp.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=1,  # Process one image at a time
            shuffle=False  # No need to shuffle for caching
        )
        
        # Create unlabeled dataset
        unlabeled_ds = dp.dataloader.create_dataset(
            data_paths['unlabeled']['images'],
            None,
            batch_size=1,
            shuffle=False
        )
        
        # Create validation dataset
        val_ds = dp.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=1,
            shuffle=False
        )
        
        # Downsample and cache labeled data
        print(f"Downsampling and caching labeled data to {self.img_size}x{self.img_size}...")
        labeled_images = []
        labeled_labels = []
        
        for image, label in tqdm(labeled_ds):
            # Resize image and label
            image_resized = tf.image.resize(image, [self.img_size, self.img_size])
            label_resized = tf.image.resize(label, [self.img_size, self.img_size], method='nearest')
            
            labeled_images.append(image_resized.numpy())
            labeled_labels.append(label_resized.numpy())
        
        # Save labeled data
        np.savez_compressed(
            self.labeled_cache,
            images=np.concatenate(labeled_images, axis=0),
            labels=np.concatenate(labeled_labels, axis=0)
        )
        
        # Free memory
        del labeled_images, labeled_labels
        collect_garbage(True)
        
        # Downsample and cache unlabeled data
        print(f"Downsampling and caching unlabeled data to {self.img_size}x{self.img_size}...")
        unlabeled_images = []
        
        for image in tqdm(unlabeled_ds):
            # Resize image
            image_resized = tf.image.resize(image, [self.img_size, self.img_size])
            unlabeled_images.append(image_resized.numpy())
        
        # Save unlabeled data
        np.savez_compressed(
            self.unlabeled_cache,
            images=np.concatenate(unlabeled_images, axis=0)
        )
        
        # Free memory
        del unlabeled_images
        collect_garbage(True)
        
        # Downsample and cache validation data
        print(f"Downsampling and caching validation data to {self.img_size}x{self.img_size}...")
        val_images = []
        val_labels = []
        
        for image, label in tqdm(val_ds):
            # Resize image and label
            image_resized = tf.image.resize(image, [self.img_size, self.img_size])
            label_resized = tf.image.resize(label, [self.img_size, self.img_size], method='nearest')
            
            val_images.append(image_resized.numpy())
            val_labels.append(label_resized.numpy())
        
        # Save validation data
        np.savez_compressed(
            self.val_cache,
            images=np.concatenate(val_images, axis=0),
            labels=np.concatenate(val_labels, axis=0)
        )
        
        # Free memory
        del val_images, val_labels
        collect_garbage(True)
        
        print("Dataset caching complete.")
    
    def load_datasets(self, batch_size):
        """Load cached datasets with batching"""
        if not self.cache_exists():
            raise FileNotFoundError("Cache files not found. Call prepare_dataset first.")
        
        # Load labeled data
        print("Loading labeled data from cache...")
        labeled_data = np.load(self.labeled_cache)
        labeled_images = labeled_data['images']
        labeled_labels = labeled_data['labels']
        
        # Load unlabeled data
        print("Loading unlabeled data from cache...")
        unlabeled_data = np.load(self.unlabeled_cache)
        unlabeled_images = unlabeled_data['images']
        
        # Load validation data
        print("Loading validation data from cache...")
        val_data = np.load(self.val_cache)
        val_images = val_data['images']
        val_labels = val_data['labels']
        
        # Create TensorFlow datasets
        train_labeled = tf.data.Dataset.from_tensor_slices((labeled_images, labeled_labels))
        train_labeled = train_labeled.batch(batch_size).prefetch(2)
        
        train_unlabeled = tf.data.Dataset.from_tensor_slices(unlabeled_images)
        train_unlabeled = train_unlabeled.batch(batch_size).prefetch(2)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_ds = val_ds.batch(batch_size).prefetch(2)
        
        print(f"Datasets loaded from cache with batch size {batch_size}")
        return train_labeled, train_unlabeled, val_ds

# Ultra memory-efficient MixMatch trainer
class UltraLowMemoryMixMatchTrainer(MixMatchTrainer):
    """MixMatch trainer with extreme memory optimizations"""
    
    def __init__(self, config):
        super().__init__(config)
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': []
        }
        
        # Print memory-optimization status
        print("Using Ultra Low Memory MixMatch Trainer")
        print(f"Image Size: {config.img_size_x}x{config.img_size_y}")
        print(f"Initial Filters: {config.n_filters}")
        print(f"Batch Size: {config.batch_size}")
    
    def _setup_model(self):
        """Setup model with minimal memory footprint"""
        # Use custom architecture or modify existing one for lower memory
        from models_tf2 import PancreasSeg
        
        # Create student model with reduced parameters
        self.student_model = PancreasSeg(self.config)
        
        # Create teacher model
        self.teacher_model = PancreasSeg(self.config)
        
        # Initialize models with dummy input
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy weights from student to teacher
        self.teacher_model.set_weights(self.student_model.get_weights())
        
        # Print model info
        student_params = self.student_model.count_params()
        print(f"Model parameters: {student_params/1000000:.2f}M ({student_params*4/1024/1024:.2f} MB)")
    
    def _setup_training_params(self):
        """Setup training parameters with memory-efficient settings"""
        # Optimizer with minimal memory overhead
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Fixed learning rate for simplicity
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08
        )
        
        # Simple step decay schedule
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [50 * 180, 75 * 180],  # Steps at which to change the learning rate
            [1e-4, 5e-5, 1e-5]     # Learning rates to use
        )
        
        # EMA decay rate for teacher model
        self.ema_decay = 0.999
        
        # Consistency weight
        self.consistency_weight = 10.0
        self.consistency_rampup = 10000
        
        print(f"Using fixed Adam optimizer with learning rate: {1e-4}")
        print(f"EMA decay: {self.ema_decay}")
        print(f"Consistency weight: {self.consistency_weight}")
    
    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch):
        """Memory-optimized training step"""
        # Get labeled data
        labeled_images, labeled_labels = labeled_batch
        unlabeled_images = unlabeled_batch
        
        # Current step for consistency weight calculation
        step = tf.cast(self.optimizer.iterations, tf.float32)
        
        # Simple augmentation (memory-efficient)
        noise = tf.random.normal(tf.shape(unlabeled_images), mean=0.0, stddev=0.1)
        unlabeled_images_aug1 = tf.clip_by_value(unlabeled_images + noise, 0.0, 1.0)
        
        noise = tf.random.normal(tf.shape(unlabeled_images), mean=0.0, stddev=0.1)
        unlabeled_images_aug2 = tf.clip_by_value(unlabeled_images + noise, 0.0, 1.0)
        
        # Get pseudo-labels from teacher model
        teacher_logits1 = self.teacher_model(unlabeled_images_aug1, training=False)
        teacher_logits2 = self.teacher_model(unlabeled_images_aug2, training=False)
        
        # Average pseudo-labels
        pseudo_labels = tf.sigmoid(tf.reduce_mean([teacher_logits1, teacher_logits2], axis=0))
        
        # Calculate loss
        with tf.GradientTape() as tape:
            # Forward pass through student model
            student_labeled_logits = self.student_model(labeled_images, training=True)
            student_unlabeled_logits1 = self.student_model(unlabeled_images_aug1, training=True)
            student_unlabeled_logits2 = self.student_model(unlabeled_images_aug2, training=True)
            
            # Supervised loss (dice loss)
            supervised_loss = self._dice_loss(labeled_labels, student_labeled_logits)
            
            # Unsupervised loss (MSE)
            # No sharpening to save memory
            unsupervised_loss1 = tf.reduce_mean(tf.square(tf.sigmoid(student_unlabeled_logits1) - pseudo_labels))
            unsupervised_loss2 = tf.reduce_mean(tf.square(tf.sigmoid(student_unlabeled_logits2) - pseudo_labels))
            
            unsupervised_loss = (unsupervised_loss1 + unsupervised_loss2) / 2.0
            
            # Calculate consistency weight
            # Simple linear ramp-up to save memory
            rampup_length = self.consistency_rampup
            rampup = tf.minimum(1.0, step / rampup_length)
            consistency_weight = self.consistency_weight * rampup
            
            # Total loss
            total_loss = supervised_loss + consistency_weight * unsupervised_loss
        
        # Get gradients
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        # Update teacher model with EMA of student weights
        self._update_teacher_model()
        
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'unsupervised_loss': unsupervised_loss,
            'consistency_weight': consistency_weight
        }
    
    def _update_teacher_model(self):
        """Update teacher model weights with EMA of student weights"""
        # Get model weights
        student_weights = self.student_model.get_weights()
        teacher_weights = self.teacher_model.get_weights()
        
        # Update using EMA
        updated_weights = []
        for s_w, t_w in zip(student_weights, teacher_weights):
            updated_weights.append(self.ema_decay * t_w + (1 - self.ema_decay) * s_w)
        
        # Apply updated weights to teacher model
        self.teacher_model.set_weights(updated_weights)
    
    def train(self, data_paths, use_cache=True, cache_dir=None):
        """Train with memory-efficient operations"""
        print("\nStarting ultra memory-efficient MixMatch training...")
        
        # Set small batch size
        batch_size = self.config.batch_size
        
        # Prepare datasets with caching
        if use_cache and cache_dir:
            # Initialize cache handler
            cache = SimpleDataCache(cache_dir, img_size=self.config.img_size_x)
            
            if not cache.cache_exists():
                # Create datasets with downsampling
                print("\nPreparing datasets with downsampling...")
                cache.prepare_dataset(data_paths, self.data_pipeline)
            
            # Load datasets from cache
            train_labeled, train_unlabeled, val_ds = cache.load_datasets(batch_size)
        else:
            # Create datasets directly
            print("\nCreating datasets without caching...")
            
            train_labeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['labeled']['images'],
                data_paths['labeled']['labels'],
                batch_size=batch_size,
                shuffle=True
            )
            
            train_unlabeled = self.data_pipeline.dataloader.create_dataset(
                data_paths['unlabeled']['images'],
                None,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_ds = self.data_pipeline.dataloader.create_dataset(
                data_paths['validation']['images'],
                data_paths['validation']['labels'],
                batch_size=batch_size,
                shuffle=False
            )
        
        # Training parameters
        best_dice = 0
        patience = 15
        patience_counter = 0
        
        # Create logs directory
        logs_dir = f"{self.config.checkpoint_dir}/logs"
        os.makedirs(logs_dir, exist_ok=True)
        log_file = f"{logs_dir}/mixmatch_training_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Write CSV header
        with open(log_file, 'w') as f:
            f.write("epoch,total_loss,supervised_loss,consistency_loss,val_dice,teacher_dice,learning_rate\n")
        
        # Determine steps per epoch
        steps_per_epoch = min(
            sum(1 for _ in train_labeled),
            sum(1 for _ in train_unlabeled)
        )
        print(f"Steps per epoch: {steps_per_epoch}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Reset metrics
            epoch_losses = []
            epoch_supervised_losses = []
            epoch_consistency_losses = []
            
            # Create iterators
            labeled_iter = iter(train_labeled)
            unlabeled_iter = iter(train_unlabeled)
            
            # Progress bar
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            progress_bar = tqdm(range(steps_per_epoch), desc="Training")
            
            # Training steps
            for step in progress_bar:
                try:
                    # Get batch data
                    labeled_batch = next(labeled_iter)
                    unlabeled_batch = next(unlabeled_iter)
                    
                    # Training step
                    metrics = self.train_step(labeled_batch, unlabeled_batch)
                    
                    # Record metrics
                    epoch_losses.append(float(metrics['total_loss']))
                    epoch_supervised_losses.append(float(metrics['supervised_loss']))
                    epoch_consistency_losses.append(float(metrics['unsupervised_loss']))
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{float(metrics['total_loss']):.4f}",
                        'sup_loss': f"{float(metrics['supervised_loss']):.4f}"
                    })
                    
                    # Clear memory periodically
                    if step % 25 == 0 and step > 0:
                        collect_garbage()
                    
                except tf.errors.ResourceExhaustedError:
                    print("\nOUT OF MEMORY ERROR: Skipping batch and collecting garbage...")
                    collect_garbage(True)
                    continue
                
                except StopIteration:
                    print("\nReached end of dataset")
                    break
                
                except Exception as e:
                    print(f"\nError during training step: {e}")
                    continue
            
            # Force garbage collection
            collect_garbage(True)
            
            # Validation
            val_dice = self.validate(val_ds)
            teacher_dice = self.validate_teacher(val_ds)
            
            # Calculate metrics
            epoch_time = time.time() - start_time
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            mean_sup_loss = np.mean(epoch_supervised_losses) if epoch_supervised_losses else 0.0
            mean_cons_loss = np.mean(epoch_consistency_losses) if epoch_consistency_losses else 0.0
            
            # Get current learning rate
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            
            # Update history
            self.history['train_loss'].append(mean_loss)
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(current_lr)
            self.history['supervised_loss'].append(mean_sup_loss)
            self.history['consistency_loss'].append(mean_cons_loss)
            
            # Log results
            print(f"Time: {epoch_time:.2f}s | Loss: {mean_loss:.4f} | Val Dice: {val_dice:.4f} | Teacher Dice: {teacher_dice:.4f}")
            
            # Write to CSV log
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_sup_loss:.6f},{mean_cons_loss:.6f},"
                        f"{val_dice:.6f},{teacher_dice:.6f},{current_lr:.8e}\n")
            
            # Update plots every epoch
            self.plot_progress()
            
            # Save best model
            best_model_dice = max(val_dice, teacher_dice)
            better_model = "teacher" if teacher_dice > val_dice else "student"
            
            if best_model_dice > best_dice:
                best_dice = best_model_dice
                
                # Define checkpoint path
                checkpoint_dir = f"{self.config.checkpoint_dir}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/best_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
                
                # Save models
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
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = f"{self.config.checkpoint_dir}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}_{time.strftime('%Y%m%d_%H%M%S')}"
                
                self.student_model.save_weights(f"{checkpoint_path}_student")
                self.teacher_model.save_weights(f"{checkpoint_path}_teacher")
                print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Final checkpoint
        checkpoint_dir = f"{self.config.checkpoint_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_checkpoint = f"{checkpoint_dir}/final_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.student_model.save_weights(f"{final_checkpoint}_student")
        self.teacher_model.save_weights(f"{final_checkpoint}_teacher")
        
        # Final plot
        self.plot_progress(final=True)
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history
    
    def validate(self, val_ds):
        """Validate student model"""
        dice_scores = []
        
        for images, labels in val_ds:
            # Forward pass
            logits = self.student_model(images, training=False)
            
            # Calculate dice score
            dice = 1.0 - self._dice_loss(labels, logits)
            dice_scores.append(float(dice))
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def validate_teacher(self, val_ds):
        """Validate teacher model"""
        dice_scores = []
        
        for images, labels in val_ds:
            # Forward pass
            logits = self.teacher_model(images, training=False)
            
            # Calculate dice score
            dice = 1.0 - self._dice_loss(labels, logits)
            dice_scores.append(float(dice))
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def plot_progress(self, final=False):
        """Plot training progress, minimizing memory usage"""
        if len(self.history['train_loss']) < 2:
            return
        
        # Create directory
        viz_dir = f"{self.config.checkpoint_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot metrics
        plt.figure(figsize=(15, 10))
        
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
        
        # Add best dice markers
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
        
        plt.tight_layout()
        
        # Save the figure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if final:
            plt.savefig(f"{viz_dir}/final_training_progress.png", dpi=100)
        else:
            plt.savefig(f"{viz_dir}/latest_training_progress.png", dpi=100)
            plt.savefig(f"{viz_dir}/training_progress_{timestamp}.png", dpi=100)
        
        plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Ultra memory-efficient MixMatch training')
    parser.add_argument('--data_dir', type=Path, default='/scratch/lustre/home/mdah0000/images/cropped',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=Path, default='/scratch/lustre/home/mdah0000/smm/v14/mixmatch_results_ultralow',
                       help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size for training (lower = less memory)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--num_labeled', type=int, default=20,
                       help='Number of labeled training examples')
    parser.add_argument('--use_cache', action='store_true',
                       help='Whether to cache preprocessed data')
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Create configuration
    config = StableSSLConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.img_size_x = args.img_size
    config.img_size_y = args.img_size
    config.n_filters = 16  # Reduced from 32
    config.num_channels = 1
    config.num_classes = 2
    config.checkpoint_dir = str(args.output_dir)
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name="mixmatch_ultralow",
        experiment_type="mixmatch",
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=args.num_labeled, num_validation=63)
    
    # Create data pipeline
    print("Setting up data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer
    print("Creating ultra memory-efficient MixMatch trainer...")
    trainer = UltraLowMemoryMixMatchTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Define cache directory
    cache_dir = f"{args.output_dir}/cached_data"
    
    # Train the model
    print(f"Starting training with batch size {args.batch_size} and image size {args.img_size}...")
    trainer.train(data_paths, use_cache=args.use_cache, cache_dir=cache_dir)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x $WORK_DIR/run_mixmatch_ultralow.py
echo "Created ultra memory-efficient MixMatch implementation"

# Run the optimized MixMatch training script
echo "Starting ultra memory-efficient MixMatch training..."
python3 $WORK_DIR/run_mixmatch_ultralow.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --img_size 256 \
    --num_labeled 20 \
    --use_cache

echo "========================================================"
echo "Ultra Memory-Efficient MixMatch Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="