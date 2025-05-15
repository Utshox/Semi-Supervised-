#!/bin/bash
# SLURM batch job script for supervised learning with HIGH PERFORMANCE optimization

#SBATCH -p gpu                     # Use gpu partition 
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 8                       # Request 8 CPU cores for better data loading
#SBATCH --mem=64G                  # Request 64GB RAM
#SBATCH --time=06:00:00            # Set time limit to 6 hours
#SBATCH -o supervised-hp-%j.out    # Output file name with job ID
#SBATCH -e supervised-hp-%j.err    # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"

echo "========================================================"
echo "Running HIGH PERFORMANCE Supervised Learning - $(date)"
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

# TensorFlow GPU environment settings for maximum performance
export TF_FORCE_GPU_ALLOW_GROWTH=false  # Use all GPU memory for better performance
export TF_GPU_THREAD_MODE=gpu_private   # Ensure each GPU has its own thread
export TF_GPU_THREAD_COUNT=2            # Use 2 threads per GPU
export TF_CPP_MIN_LOG_LEVEL=1           # Reduce log verbosity
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install the correct TensorFlow version if needed
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib psutil

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create cache directory if it doesn't exist
mkdir -p preprocessed_cache
echo "Created preprocessed_cache directory for saving processed data"

# ========================================================================
# CREATING PYTHON FILE: This section creates the Python script file that
# will be executed. The Python code is embedded directly in this SLURM script.
# ========================================================================
echo "Creating Python script file: run_supervised_highperf.py"

cat > run_supervised_highperf.py << 'EOF'
import sys
import os
import psutil
import numpy as np

# Add the directory containing your module to Python's path - FIXED PATH
sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

import tensorflow as tf
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your modules
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import SupervisedTrainer
from data_loader_tf2 import DataPipeline, PancreasDataLoader
from main import prepare_data_paths, setup_gpu

# Configure TensorFlow for maximum performance
def configure_tensorflow():
    """Configure TensorFlow for high-performance training"""
    # Enable mixed precision for faster training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Compute dtype: {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}")
    
    # Set up better GPU performance settings
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    if gpu_devices:
        # Set memory growth and set to use BFloat16
        for device in gpu_devices:
            # Using full memory for maximum performance
            try:
                tf.config.experimental.set_memory_growth(device, False)
            except:
                pass

        # Disable XLA JIT compilation which causes errors with UpSampling2D layers
        tf.config.optimizer.set_jit(False)  # Disable XLA
        print("XLA JIT compilation disabled due to compatibility with UpSampling2D")
        
        return True
    else:
        print("No GPU devices found.")
        return False

# Inherit directly from PancreasDataLoader instead of creating a function
# This ensures all methods are properly inherited
class OptimizedPancreasDataLoader(PancreasDataLoader):
    """Enhanced dataloader with optimized loading and 3D to 2D conversion"""
    def __init__(self, config):
        super().__init__(config)
        print(f"Initialized OptimizedPancreasDataLoader with img_size: {config.img_size_x}x{config.img_size_y}")
    
    def _load_volume(self, file_path):
        """Load a volume from a file path and convert to 2D middle slice"""
        try:
            # Convert any tensor type to string
            if isinstance(file_path, tf.Tensor):
                file_path = file_path.numpy()
            
            # Convert bytes to string if necessary
            if isinstance(file_path, bytes):
                file_path = file_path.decode('utf-8')
            
            # Ensure we have a proper string path
            file_path = str(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"WARNING: File does not exist: {file_path}")
                return np.zeros((self.config.img_size_x, self.config.img_size_y), dtype=np.float32)
            
            # Load data from numpy file
            data = np.load(file_path)
            
            # Get the center slice from the volume (assuming 3D data)
            # This converts 3D volume to 2D slice for compatibility with the model
            if len(data.shape) == 3:
                middle_slice_idx = data.shape[1] // 2
                data_2d = data[:, middle_slice_idx, :]
            else:
                print(f"Unexpected data shape: {data.shape}, using as is")
                data_2d = data
            
            # Ensure correct size for the model
            data_2d_resized = tf.image.resize(
                tf.expand_dims(data_2d, -1),  # Add channel dimension
                [self.config.img_size_x, self.config.img_size_y]
            )
            
            return data_2d_resized.numpy().astype(np.float32)
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Return a dummy 2D image in case of error
            return np.zeros((self.config.img_size_x, self.config.img_size_y), dtype=np.float32)
    
    def _normalize_image(self, image):
        """Normalize image data - handling unknown rank tensors safely"""
        # First ensure we're working with a tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Add channel dimension if needed
        if len(tf.shape(image)) < 3:
            image = tf.expand_dims(image, -1)
            
        # Safe normalization that works regardless of shape
        image_min = tf.reduce_min(image)
        image_max = tf.reduce_max(image)
        return tf.cast((image - image_min) / (tf.maximum(image_max - image_min, 1e-7)), tf.float32)
        
    def _normalize_mask(self, mask):
        """Normalize mask data for segmentation - handling unknown rank tensors safely"""
        # First ensure we're working with a tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        
        # Add channel dimension if needed
        if len(tf.shape(mask)) < 3:
            mask = tf.expand_dims(mask, -1)
            
        # Binarize mask: 1 for pancreas, 0 for background
        return tf.cast(tf.greater(mask, 0.5), tf.float32)
            
    def create_dataset(self, image_paths, label_paths=None, batch_size=8, 
                       shuffle=False, augment=False, cache=True):
        """Creates a TF dataset with optimized pipeline"""
        # Print the first few paths to debug
        print("First few image paths:", image_paths[:3])
        if label_paths is not None:
            print("First few label paths:", label_paths[:3])
        
        if label_paths is not None:
            # Create a dataset with both images and labels
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
            
            # For troubleshooting path types
            for path_pair in dataset.take(1):
                img_path, mask_path = path_pair
                print(f"Image path type: {type(img_path)}, Label path type: {type(mask_path)}")
            
            # Use parallel loading - with explicit output shapes to avoid shape inference issues
            dataset = dataset.map(
                lambda x, y: (
                    tf.py_function(self._load_volume, [x], tf.float32),
                    tf.py_function(self._load_volume, [y], tf.float32)
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Add augmentation if needed
            if augment:
                dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Normalize data, reshape, and add channel dimension
            dataset = dataset.map(
                lambda x, y: (
                    self._normalize_image(x),
                    self._normalize_mask(y)
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Debug info
            for i, (img, mask) in enumerate(dataset.take(1)):
                print(f"Dataset item shapes - Image: {img.shape}, Mask: {mask.shape}")
                
        else:
            # Create a dataset with only images
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            
            # Load and process images
            dataset = dataset.map(
                lambda x: self._normalize_image(tf.py_function(self._load_volume, [x], tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Cache the dataset in memory if requested
        if cache:
            dataset = dataset.cache()
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch, prefetch, and enable parallelism
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create optimized data pipeline
def create_optimized_dataloader(config):
    """Create an optimized dataloader with prefetch and caching"""
    return OptimizedPancreasDataLoader(config)

# Override train method with optimizations
def patched_train(self, data_paths):
    """Optimized training loop"""
    print("\nStarting supervised training with HIGH PERFORMANCE settings...")
    
    # Create datasets with optimized loading
    train_ds = self.data_pipeline.dataloader.create_dataset(
        data_paths['labeled']['images'],
        data_paths['labeled']['labels'],
        batch_size=self.config.batch_size,
        shuffle=True,
        cache=True
    )
    
    val_ds = self.data_pipeline.dataloader.create_dataset(
        data_paths['validation']['images'],
        data_paths['validation']['labels'],
        batch_size=self.config.batch_size,
        shuffle=False,
        cache=True
    )
    
    best_dice = 0
    patience = self.config.early_stopping_patience
    patience_counter = 0
    
    steps_per_epoch = len(list(train_ds))
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # DO NOT use XLA compilation for train_step - it doesn't support UpSampling2D gradients
    # self.train_step = tf.function(self.train_step, jit_compile=True)
    
    # Times for benchmarking
    epoch_times = []
    
    for epoch in range(self.config.num_epochs):
        start_time = time.time()
        
        # Training with progress bar
        epoch_losses = []
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        
        # Start system monitoring
        cpu_percents = []
        mem_percent = psutil.virtual_memory().percent
        
        progress_bar = tqdm(train_ds, total=steps_per_epoch, desc=f"Training")
        for batch in progress_bar:
            # Monitor CPU usage periodically
            if len(epoch_losses) % 5 == 0:
                cpu_percents.append(psutil.cpu_percent(interval=0.1))
            
            images, labels = batch
            
            # Print shapes of first batch for debugging
            if epoch == 0 and len(epoch_losses) == 0:
                print(f"Training batch shapes - Images: {images.shape}, Labels: {labels.shape}")
                
            loss, _ = self.train_step(images, labels)
            epoch_losses.append(float(loss))
            progress_bar.set_postfix({"loss": f"{float(loss):.4f}"})
        
        # Do validation
        val_dice = self.validate(val_ds)
            
        # Update history
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        self.history['train_loss'].append(float(sum(epoch_losses) / len(epoch_losses)))
        self.history['val_dice'].append(val_dice)
        self.history['learning_rate'].append(
            float(self.lr_schedule(self.optimizer.iterations))
        )
        
        # Logging
        print(f"Time: {epoch_time:.2f}s | Loss: {self.history['train_loss'][-1]:.4f} | Val Dice: {val_dice:.4f}")
        print(f"System usage - CPU: {sum(cpu_percents)/max(len(cpu_percents),1):.1f}% | Memory: {mem_percent:.1f}%")
        print(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f}s | Batch size: {self.config.batch_size}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            self.save_checkpoint('best_supervised_model')
            print(f"✓ New best model saved! Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break
        
        # Plot progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            self.plot_progress()
    
    print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
    self.plot_progress()  # Final plot
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f} seconds")
    print(f"Fastest epoch: {min(epoch_times):.2f} seconds")
    print(f"Slowest epoch: {max(epoch_times):.2f} seconds")
    print(f"Total training time: {sum(epoch_times)/60:.2f} minutes")
    
    return self.history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--experiment_name', type=str, default='supervised_highperf',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Number of epochs to train')
    args = parser.parse_args()

    # Setup with high-performance configuration
    gpu_available = configure_tensorflow()
    
    # Create config with performance settings
    config = StableSSLConfig()
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='supervised',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=225, num_validation=56)
    
    # Create optimized data pipeline
    print("Creating optimized data pipeline...")
    data_pipeline = DataPipeline(config)
    data_pipeline.dataloader = create_optimized_dataloader(config)
    
    # Create trainer
    print("Creating supervised trainer...")
    trainer = SupervisedTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Apply patch to use optimized training loop
    trainer.train = patched_train.__get__(trainer, SupervisedTrainer)

    # Print experiment info
    print("\nExperiment Configuration:")
    print(f"Training Type: supervised (high-performance)")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Learning rate: {trainer.lr_schedule.get_config()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    print(f"Using mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    print(f"XLA compilation enabled: {tf.config.optimizer.get_jit()}")
    
    # Create experiment directory
    exp_dir = experiment_config.get_experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / 'config.txt', 'w') as f:
        f.write(f"Training Type: supervised (high-performance)\n")
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Timestamp: {experiment_config.timestamp}\n")
        f.write(f"Learning Rate Config: {trainer.lr_schedule.get_config()}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Image Size: {config.img_size_x}x{config.img_size_y}\n")
        f.write(f"Mixed Precision: {tf.keras.mixed_precision.global_policy().name}\n")
        f.write(f"XLA Compilation: {tf.config.optimizer.get_jit()}\n")

    # Train
    print("\nStarting high-performance training...")
    history = trainer.train(data_paths)
    
    # Save final results
    results_dir = experiment_config.get_experiment_dir()
    np.save(results_dir / 'training_history.npy', history)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
EOF

# Make the Python file executable
chmod +x run_supervised_highperf.py
echo "Python script created: run_supervised_highperf.py"

# ========================================================================
# EXECUTING PYTHON FILE: Now we run the script we just created
# ========================================================================
echo "Executing Python script with high-performance optimizations"
python3 $WORK_DIR/run_supervised_highperf.py \
  --experiment_name supervised_highperf \
  --data_dir $DATA_DIR \
  --batch_size 4 \
  --num_epochs 30

echo "========================================================"
echo "High-Performance Supervised Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $WORK_DIR/supervised_highperf_*"
echo "========================================================"