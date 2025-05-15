#!/bin/bash
# SLURM batch job script for MixMatch semi-supervised learning with enhancements

#SBATCH -p gpu                     # Use gpu partition 
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 8                       # Request 8 CPU cores for better data loading
#SBATCH --mem=64G                  # Request 64GB RAM
#SBATCH --time=08:00:00            # Time limit to 12 hours
#SBATCH -o mixmatch-enh-%j.out     # Output file name with job ID
#SBATCH -e mixmatch-enh-%j.err     # Error file name with job ID
#SBATCH --mail-type=END,FAIL       # Send email when job ends or fails
#SBATCH --job-name=pancreas_mixm   # Add descriptive job name

# Path to your data directory - using consistent paths as supervised model
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
OUTPUT_DIR="$WORK_DIR/mixmatch_results"

echo "========================================================"
echo "Running Enhanced MixMatch Learning - $(date)"
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

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Memory optimization settings - using async allocator for better memory handling
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_GPU_HOST_MEM_LIMIT_IN_MB=4096  # Limit host memory to 4GB

# Install the correct TensorFlow version if needed
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1" tqdm matplotlib psutil

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create a customized version of main.py to support our enhanced MixMatch
cat > $WORK_DIR/run_mixmatch_enhanced.py << 'EOF'
import tensorflow as tf
from pathlib import Path
import argparse
import time
from datetime import datetime
import numpy as np
import gc

# Import your modules - use the existing path
import sys
sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import MixMatchTrainer
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from main import prepare_data_paths, setup_gpu

print("TensorFlow version:", tf.__version__)

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

class EnhancedMixMatchTrainer(MixMatchTrainer):
    """Enhanced version of MixMatch trainer with memory optimization"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._setup_model()  # Override parent's model setup
        self._setup_training_params()
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
    def _setup_model(self):
        """Setup the U-Net model with memory-efficient parameters"""
        print("Setting up enhanced U-Net model for MixMatch...")
        
        # Create a more memory efficient model by reducing filters
        self.config.n_filters = 32  # Match supervised settings
        
        # Create the model directly using PancreasSeg
        self.model = PancreasSeg(self.config)
        
        # Initialize model with dummy input
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        _ = self.model(dummy_input)
        
        print(f"Model created with input shape: ({self.config.img_size_x}, {self.config.img_size_y}, {self.config.num_channels})")
        print(f"Initial filters: {self.config.n_filters}")
        
        # Print model summary to understand the memory footprint
        total_params = self.model.count_params()
        print(f"Total model parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
        
    def _setup_training_params(self):
        """Setup training parameters with improved learning rate scheduler"""
        # Define a LearningRateSchedule class
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
        
        # Calculate total steps for entire training
        total_steps = self.config.num_epochs * (225 // self.config.batch_size + 1)
        
        # Create learning rate schedule with optimized values matching supervised
        max_lr = 5e-4
        min_lr = 1e-5
        
        # Create an instance of our custom learning rate scheduler
        self.lr_schedule = OneCycleLR(
            max_lr=max_lr,
            min_lr=min_lr,
            total_steps=total_steps
        )
        
        # Set MixMatch specific parameters
        self.alpha = 0.75         # Beta distribution parameter for MixUp
        self.temperature = 0.5    # Sharpening temperature for pseudo-labels
        self.lambda_u = 100.0     # Weight for unsupervised loss
        self.rampup_length = 30000  # Steps to ramp up lambda_u
        
        print(f"Learning rate schedule: One-cycle from {min_lr} to {max_lr} and back to {min_lr}")
        print(f"Total training steps: {total_steps}")
        print(f"MixMatch alpha: {self.alpha}")
        print(f"MixMatch temperature: {self.temperature}")
        print(f"MixMatch lambda_u: {self.lambda_u}")
        
        # Setup optimizer with proper learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08
        )
    
    def sharpen(self, p, T=0.5):
        """Sharpen the probability distribution"""
        # p has shape [batch, h, w, n_classes]
        p_power = tf.pow(p, 1.0 / T)
        return p_power / tf.reduce_sum(p_power, axis=-1, keepdims=True)
    
    def mixup(self, x1, x2, y1, y2, alpha=0.75):
        """Perform mixup data augmentation"""
        # Sample lambda from beta distribution
        batch_size = tf.shape(x1)[0]
        
        # Sample mixing parameter from beta distribution
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        l = beta_dist.sample(batch_size)
        l = tf.maximum(l, 1-l)  # Ensure l is not too small
        l = tf.reshape(l, [batch_size, 1, 1, 1])  # Shape for broadcasting
        
        # Mixup the data
        mixed_x = l * x1 + (1 - l) * x2
        
        # If y is one-hot, reshape l for y
        l_y = tf.reshape(l, [batch_size, 1, 1, 1])
        mixed_y = l_y * y1 + (1 - l_y) * y2
        
        return mixed_x, mixed_y
    
    def get_unsupervised_weight(self, step):
        """Get ramped unsupervised weight"""
        if not hasattr(self, 'rampup_length'):
            return self.lambda_u
            
        # Ramp up the unsupervised weight
        rampup_length = self.rampup_length
        rampup = tf.minimum(1.0, tf.cast(step, tf.float32) / rampup_length)
        return self.lambda_u * rampup
    
    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images):
        """Execute one training step with MixMatch"""
        # Current step for scheduling
        step = self.optimizer.iterations
        batch_size = tf.shape(labeled_images)[0]
        
        # Generate pseudo-labels for unlabeled data
        # Apply two different augmentations
        unlabeled_images_aug1 = unlabeled_images + tf.random.normal(
            shape=tf.shape(unlabeled_images), 
            mean=0.0, 
            stddev=0.03
        )
        
        unlabeled_images_aug2 = unlabeled_images + tf.random.normal(
            shape=tf.shape(unlabeled_images), 
            mean=0.0, 
            stddev=0.03
        )
        
        # Get predictions for both augmentations
        with tf.GradientTape() as tape:
            # Forward pass for labeled data
            labeled_logits = self.model(labeled_images, training=True)
            
            # Get predictions for unlabeled augmentations
            unlabeled_logits1 = self.model(unlabeled_images_aug1, training=True)
            unlabeled_logits2 = self.model(unlabeled_images_aug2, training=True)
            
            # Convert logits to probabilities for unlabeled data
            # For binary segmentation, apply sigmoid
            unlabeled_prob1 = tf.sigmoid(unlabeled_logits1)
            unlabeled_prob2 = tf.sigmoid(unlabeled_logits2)
            
            # Average predictions as pseudo-labels and sharpen
            pseudo_labels = (unlabeled_prob1 + unlabeled_prob2) / 2.0
            pseudo_labels = tf.where(
                pseudo_labels > 0.5, 
                tf.ones_like(pseudo_labels), 
                tf.zeros_like(pseudo_labels)
            )
            
            # MixUp augmentation for labeled and unlabeled data
            # Swap unlabeled examples within batch
            unlabeled_indices = tf.random.shuffle(tf.range(tf.shape(unlabeled_images)[0]))
            unlabeled_images_shuffled = tf.gather(unlabeled_images, unlabeled_indices)
            pseudo_labels_shuffled = tf.gather(pseudo_labels, unlabeled_indices)
            
            # Swap labeled examples within batch
            labeled_indices = tf.random.shuffle(tf.range(tf.shape(labeled_images)[0]))
            labeled_images_shuffled = tf.gather(labeled_images, labeled_indices)
            labeled_labels_shuffled = tf.gather(labeled_labels, labeled_indices)
            
            # Mixup labeled samples with each other
            mixed_labeled_images, mixed_labeled_labels = self.mixup(
                labeled_images, labeled_images_shuffled,
                labeled_labels, labeled_labels_shuffled,
                self.alpha
            )
            
            # Mixup unlabeled samples with each other
            mixed_unlabeled_images, mixed_pseudo_labels = self.mixup(
                unlabeled_images, unlabeled_images_shuffled,
                pseudo_labels, pseudo_labels_shuffled,
                self.alpha
            )
            
            # Forward pass for mixed labeled data
            mixed_labeled_logits = self.model(mixed_labeled_images, training=True)
            
            # Forward pass for mixed unlabeled data
            mixed_unlabeled_logits = self.model(mixed_unlabeled_images, training=True)
            
            # Supervised loss with mixed labeled data
            supervised_loss = self._dice_loss(mixed_labeled_labels, mixed_labeled_logits)
            
            # Unsupervised loss with mixed unlabeled data and pseudo-labels
            mixed_unlabeled_probs = tf.sigmoid(mixed_unlabeled_logits)
            unsupervised_loss = tf.reduce_mean(
                tf.square(mixed_unlabeled_probs - mixed_pseudo_labels)
            )
            
            # Get current unsupervised weight
            unsup_weight = self.get_unsupervised_weight(tf.cast(step, tf.float32))
            
            # Combined loss
            total_loss = supervised_loss + unsup_weight * unsupervised_loss
        
        # Get gradients and update model
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate dice score for monitoring (just for labeled data)
        dice_score = 1 - self._dice_loss(labeled_labels, labeled_logits)
        
        return total_loss, dice_score, supervised_loss, unsupervised_loss
    
    def validate(self, val_dataset):
        """Run validation on the model"""
        val_dice_scores = []
        
        for images, labels in val_dataset:
            # Get predictions
            logits = self.model(images, training=False)
            
            # Calculate dice score
            dice_score = 1 - self._dice_loss(labels, logits)
            val_dice_scores.append(float(dice_score))
        
        # Return mean dice score
        if val_dice_scores:
            mean_dice = float(sum(val_dice_scores) / len(val_dice_scores))
            
            # Print information about the best dice score in this validation batch
            best_dice = max(val_dice_scores)
            print(f"Mean validation Dice: {mean_dice:.4f}, Best batch Dice: {best_dice:.4f}")
            
            return mean_dice
        else:
            print("WARNING: No validation batches processed!")
            return 0.0
    
    def train(self, data_paths):
        """Train with both labeled and unlabeled data using MixMatch"""
        print("\nStarting Enhanced MixMatch training...")
        
        # Use a smaller batch size to avoid OOM
        small_batch_size = 2
        print(f"Using batch size: {small_batch_size}")
        self.config.batch_size = small_batch_size
        
        # Recalculate total steps with correct batch size
        total_steps = self.config.num_epochs * (225 // self.config.batch_size + 1)
        
        # Create learning rate schedule with recalculated steps
        max_lr = 5e-4
        min_lr = 1e-5
        
        # Replace learning rate schedule with correct steps
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
        
        # Update learning rate schedule
        self.lr_schedule = OneCycleLR(
            max_lr=max_lr,
            min_lr=min_lr,
            total_steps=total_steps
        )
        
        # Update optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08
        )
        
        print(f"Learning rate schedule recalculated for batch size {self.config.batch_size}")
        print(f"Total training steps: {total_steps}")
        
        # Create datasets
        train_labeled = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=small_batch_size,
            shuffle=True
        )
        
        train_unlabeled = self.data_pipeline.dataloader.create_dataset(
            data_paths['unlabeled']['images'],
            None,  # No labels for unlabeled data
            batch_size=small_batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=small_batch_size,
            shuffle=False
        )
        
        # Setup training parameters
        best_dice = 0
        patience = 15  # Early stopping patience
        patience_counter = 0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create experiment log file
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f'mixmatch_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        
        with open(log_file, 'w') as f:
            f.write('epoch,loss,supervised_loss,unsupervised_loss,val_dice,learning_rate,time\n')
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            epoch_losses = []
            epoch_sup_losses = []
            epoch_unsup_losses = []
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Zip labeled and unlabeled datasets
            # Need to handle case where they have different sizes
            labeled_iter = iter(train_labeled)
            unlabeled_iter = iter(train_unlabeled)
            
            num_labeled_batches = sum(1 for _ in train_labeled)
            
            # Process data batches
            for i in range(num_labeled_batches):
                try:
                    labeled_batch = next(labeled_iter)
                    labeled_images, labeled_labels = labeled_batch
                except StopIteration:
                    # Reset labeled iterator if it runs out first
                    labeled_iter = iter(train_labeled)
                    labeled_batch = next(labeled_iter)
                    labeled_images, labeled_labels = labeled_batch
                
                try:
                    unlabeled_batch = next(unlabeled_iter)
                    unlabeled_images = unlabeled_batch[0]  # Only images for unlabeled
                except StopIteration:
                    # Reset unlabeled iterator if it runs out first
                    unlabeled_iter = iter(train_unlabeled)
                    unlabeled_batch = next(unlabeled_iter)
                    unlabeled_images = unlabeled_batch[0]
                
                # Training step
                loss, dice, sup_loss, unsup_loss = self.train_step(
                    labeled_images, labeled_labels, unlabeled_images)
                
                # Record metrics
                epoch_losses.append(float(loss))
                epoch_sup_losses.append(float(sup_loss))
                epoch_unsup_losses.append(float(unsup_loss))
                
                # Progress printing every 10 batches
                if (i+1) % 10 == 0:
                    print(f"Batch {i+1}/{num_labeled_batches}, "
                          f"Loss: {float(loss):.4f}, "
                          f"Dice: {float(dice):.4f}, "
                          f"Sup Loss: {float(sup_loss):.4f}, "
                          f"Unsup Loss: {float(unsup_loss):.4f}")
            
            # Force garbage collection after each epoch
            collect_garbage()
            
            # Validation
            val_dice = self.validate(val_ds)
            
            # Force garbage collection again
            collect_garbage()
            
            # Update history
            epoch_time = time.time() - start_time
            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            mean_sup_loss = sum(epoch_sup_losses) / len(epoch_sup_losses) if epoch_sup_losses else 0
            mean_unsup_loss = sum(epoch_unsup_losses) / len(epoch_unsup_losses) if epoch_unsup_losses else 0
            
            self.history['train_loss'].append(mean_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )
            
            # Logging
            print(f"Time: {epoch_time:.2f}s | Loss: {mean_loss:.4f} | "
                  f"Sup Loss: {mean_sup_loss:.4f} | Unsup Loss: {mean_unsup_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | LR: {self.history['learning_rate'][-1]:.8e}")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{mean_loss:.6f},{mean_sup_loss:.6f},"
                        f"{mean_unsup_loss:.6f},{val_dice:.6f},"
                        f"{self.history['learning_rate'][-1]:.8e},{epoch_time:.2f}\n")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                model_path = checkpoint_dir / f"best_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
                self.model.save_weights(str(model_path))
                print(f"✓ New best model saved! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
            
        # Save final model
        final_model_path = checkpoint_dir / f"final_mixmatch_{time.strftime('%Y%m%d_%H%M%S')}"
        self.model.save_weights(str(final_model_path))
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--output_dir', type=Path, required=False,
                      help='Path to output directory',
                      default='/scratch/lustre/home/mdah0000/smm/v14/mixmatch_results')
    parser.add_argument('--experiment_name', type=str, default='mixmatch_enhanced',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training (will be reduced to 2 internally)')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--alpha', type=float, default=0.75,
                      help='MixMatch alpha parameter (Beta distribution)')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature for sharpening pseudo-labels')
    parser.add_argument('--lambda_u', type=float, default=100.0,
                      help='Weight for unsupervised loss')
    parser.add_argument('--rampup_length', type=int, default=30000,
                      help='Steps to ramp up lambda_u')
    parser.add_argument('--num_labeled', type=int, default=225,
                      help='Number of labeled training images')
    parser.add_argument('--num_validation', type=int, default=56,
                      help='Number of validation images')
    parser.add_argument('--n_filters', type=int, default=32,
                      help='Number of filters in UNet')
    parser.add_argument('--img_size', type=int, default=256,
                      help='Image size (both dimensions)')
    args = parser.parse_args()

    # Setup GPU
    setup_gpu()
    
    # Create config with memory-efficient settings
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.img_size_x = args.img_size
    config.img_size_y = args.img_size
    config.n_filters = args.n_filters
    config.num_classes = 2
    config.output_dir = args.output_dir
    config.checkpoint_dir = f"{args.output_dir}/checkpoints"
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='mixmatch',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths with specified splits
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=args.num_labeled, num_validation=args.num_validation)
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer
    print("Creating memory-optimized MixMatch trainer...")
    trainer = EnhancedMixMatchTrainer(config)
    trainer.data_pipeline = data_pipeline
    trainer.alpha = args.alpha
    trainer.temperature = args.temperature
    trainer.lambda_u = args.lambda_u
    trainer.rampup_length = args.rampup_length
    
    # Create experiment directory
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    trainer.train(data_paths)
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
EOF

# Run the enhanced MixMatch script
echo "Starting enhanced MixMatch training with optimized parameters..."
$PYTHON_CMD $WORK_DIR/run_mixmatch_enhanced.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --experiment_name "mixmatch_enhanced" \
    --alpha 0.75 \
    --temperature 0.5 \
    --lambda_u 100.0 \
    --rampup_length 30000 \
    --batch_size 2 \
    --num_epochs 100 \
    --n_filters 32 \
    --img_size 256 \
    --num_labeled 225 \
    --num_validation 56

echo "========================================================"
echo "MixMatch Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================"