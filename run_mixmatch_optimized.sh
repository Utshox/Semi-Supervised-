#!/bin/bash
# SLURM batch job script for MixMatch SSL with optimization for HPC

#SBATCH -p gpu                      # Use gpu partition 
#SBATCH --gres=gpu:1                # Request 1 GPU (conserving GPU hours)
#SBATCH -n 4                        # Request 4 CPU cores
#SBATCH --mem=32G                   # Request 32GB RAM
#SBATCH --time=08:00:00             # Set time limit to 8 hours
#SBATCH -o mixmatch-opt-%j.out      # Output file name with job ID
#SBATCH -e mixmatch-opt-%j.err      # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"

echo "========================================================"
echo "Running Optimized MixMatch Learning - $(date)"
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

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce log verbosity
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

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
cat > run_mixmatch_with_logging.py << 'EOF'
import sys
import os

# Add the directory containing your module to Python's path
sys.path.append('/stud3/2023/mdah0000/smm/Semi-Supervised-')

import tensorflow as tf
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import your modules
from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import MixMatchTrainer
from main import prepare_data_paths, setup_gpu

# Override train method in MixMatchTrainer to add progress bar
def patched_train(self, data_paths):
    """Main training loop with progress bar"""
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
    
    for epoch in range(self.config.num_epochs):
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
            except StopIteration:
                print("Reached end of dataset before completing epoch")
                break
        
        # Validation
        val_dice, teacher_dice = self.validate(val_ds)
        
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
            self.save_checkpoint('best_mixmatch_model')
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
    return self.history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--experiment_name', type=str, default='mixmatch_ssl',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    args = parser.parse_args()

    # Setup
    gpu_available = setup_gpu()
    config = StableSSLConfig()
    
    # Set dimensions and hyperparameters
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = args.batch_size
    config.num_epochs = 50  # Adjust as needed
    config.early_stopping_patience = 15  # More patience for semi-supervised
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='mixmatch',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=63)
    
    # Create trainer
    trainer = MixMatchTrainer(config)
    
    # Apply patch to add progress bar
    trainer.train = patched_train.__get__(trainer, MixMatchTrainer)

    # Print experiment info
    print("\nExperiment Configuration:")
    print(f"Training Type: mixmatch")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Learning rate: {trainer.lr_schedule.get_config()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    
    # Create experiment directory
    exp_dir = experiment_config.get_experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / 'config.txt', 'w') as f:
        f.write(f"Training Type: mixmatch\n")
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Timestamp: {experiment_config.timestamp}\n")
        f.write(f"Learning Rate Config: {trainer.lr_schedule.get_config()}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Image Size: {config.img_size_x}x{config.img_size_y}\n")

    # Train
    history = trainer.train(data_paths)
    
    # Save final results
    results_dir = experiment_config.get_experiment_dir()
    np.save(results_dir / 'training_history.npy', history)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
EOF

# Run with optimal batch size for memory
echo "Running MixMatch with batch size 2 for memory optimization and enhanced logging"
python3 $WORK_DIR/run_mixmatch_with_logging.py \
  --experiment_name mixmatch_ssl \
  --data_dir $DATA_DIR \
  --batch_size 2

echo "========================================================"
echo "MixMatch Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $WORK_DIR/mixmatch_results"
echo "========================================================"