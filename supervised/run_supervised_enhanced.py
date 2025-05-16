import sys
import os
import numpy as np
import gc

# Add the directory containing your module to Python's path
sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

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
from main import prepare_data_paths, setup_gpu

print("TensorFlow version:", tf.__version__)

class EnhancedSupervisedTrainer(SupervisedTrainer):
    """Enhanced version of supervised trainer with improved learning rate scheduling"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._setup_model()
        self._setup_training_params()
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
    def _setup_model(self):
        """Setup the U-Net model with enhanced parameters"""
        print("Setting up enhanced U-Net model...")
        self.model = self._create_unet(
            input_size=(self.config.img_size_x, self.config.img_size_y, self.config.num_channels),
            n_filters=self.config.n_filters,
            n_classes=1
        )
    
    def _combined_loss(self, y_true, y_pred):
        """Combined loss function (BCE + Dice + Focal) for better segmentation results"""
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle multi-channel labels
        if y_true.shape[-1] > 1:
            y_true = y_true[..., 1:2]  # Take the foreground class
        
        # Get logits and probabilities
        logits = y_pred
        y_pred = tf.nn.sigmoid(y_pred)
        
        # Binary cross-entropy loss (with label smoothing for regularization)
        epsilon = 1e-6
        # Apply label smoothing - helps prevent overconfidence
        y_true_smooth = y_true * 0.9 + 0.05
        
        bce = tf.keras.losses.binary_crossentropy(
            y_true_smooth, y_pred, from_logits=False
        )
        bce = tf.reduce_mean(bce)
        
        # Dice loss with softened targets for better gradient flow
        # Calculate per-sample intersection (True Positives)
        intersection_per_sample = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        
        # Calculate per-sample sum of pixels in true and predicted masks
        sum_true_pixels_per_sample = tf.reduce_sum(y_true, axis=[1, 2, 3])
        sum_pred_pixels_per_sample = tf.reduce_sum(y_pred, axis=[1, 2, 3])
        
        # Denominator for Dice coefficient: sum of pixels in true mask + sum of pixels in predicted mask
        denominator_per_sample = sum_true_pixels_per_sample + sum_pred_pixels_per_sample
        # Numerator for Dice coefficient: 2 * intersection
        numerator_per_sample = 2. * intersection_per_sample
        
        # Calculate Dice coefficient per sample, ensuring stability for 0/0 cases
        # (numerator + epsilon) / (denominator + epsilon) handles 0/0 as 1.0
        dice_coeff_per_sample = (numerator_per_sample + epsilon) / (denominator_per_sample + epsilon)
        
        # Average Dice coefficient over the batch
        mean_dice_coeff = tf.reduce_mean(dice_coeff_per_sample)
        
        # Dice loss is 1 - mean Dice coefficient
        dice_loss = 1.0 - mean_dice_coeff
        
        # Focal loss component - helps focus on hard examples
        gamma = 2.0
        focal_weight = (1 - y_pred) ** gamma * y_true + y_pred ** gamma * (1 - y_true)
        focal_loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=False
        )
        focal_loss = tf.reduce_mean(focal_weight * focal_loss)
        
        # Combined loss with weighted components
        return 0.4 * bce + 0.4 * dice_loss + 0.2 * focal_loss
    
    def _dice_metric(self, y_true, y_pred):
        """Dice coefficient metric for evaluation"""
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle multi-channel labels
        if y_true.shape[-1] > 1:
            y_true = y_true[..., 1:2]  # Take the foreground class
        
        # Apply sigmoid to get probabilities
        y_pred = tf.nn.sigmoid(y_pred)
        
        # Threshold predictions for binary segmentation evaluation
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        # Calculate dice coefficient
        epsilon = 1e-6
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        
        # Return dice coefficient
        return tf.reduce_mean((2. * intersection + epsilon) / (union + epsilon))
    
    def _setup_training_params(self):
        """Setup training parameters with improved learning rate scheduler"""
        # Define a better learning rate scheduler
        def one_cycle_lr(max_lr, min_lr, total_steps, step):
            """One-cycle learning rate scheduler with step parameter"""
            half_steps = total_steps // 2
            # Convert to proper TensorFlow types
            step = tf.cast(step, tf.float32)
            half_steps = tf.cast(half_steps, tf.float32)
            total_steps = tf.cast(total_steps, tf.float32)
            max_lr = tf.cast(max_lr, tf.float32)
            min_lr = tf.cast(min_lr, tf.float32)
            
            # First half: linear increase
            if step <= half_steps:
                return min_lr + (max_lr - min_lr) * (step / half_steps)
            # Second half: cosine decay
            else:
                decay_steps = step - half_steps
                cosine_decay = 0.5 * (1 + tf.cos(
                    tf.constant(np.pi) * decay_steps / (total_steps - half_steps)))
                return min_lr + (max_lr - min_lr) * cosine_decay
        
        # Calculate total steps for entire training
        total_steps = self.config.num_epochs * (225 // self.config.batch_size + 1)
        
        # Create learning rate schedule with LOWER learning rates
        # Using much lower learning rates for medical segmentation
        self.lr_schedule = lambda step: one_cycle_lr(
            max_lr=5e-5,  # Reduced from 1e-4 to 5e-5
            min_lr=1e-6,  # Reduced from 1e-5 to 1e-6
            total_steps=total_steps,
            step=step
        )
        
        # Setup optimizer with weight decay for regularization
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-6,  # Start with very low learning rate
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Setup loss function
        self.loss_fn = self._combined_loss
        
        print(f"Learning rate: One-cycle from 1e-6 to 5e-5 and back to 1e-6")
    
    def train_step(self, images, labels):
        """Custom training step with learning rate scheduling and gradient clipping"""
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        
        # Update learning rate based on current step
        lr = self.lr_schedule(self.optimizer.iterations)
        self.optimizer.learning_rate.assign(lr)
        
        # Get gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate dice score for monitoring
        dice_score = self._dice_metric(labels, predictions)
        
        return loss, dice_score, lr
    
    def validate(self, val_ds):
        """Validate the model on the validation dataset"""
        dice_scores = []
        
        for batch in val_ds:
            images, labels = batch
            predictions = self.model(images, training=False)
            dice_score = self._dice_metric(labels, predictions)
            dice_scores.append(float(dice_score))
        
        return float(sum(dice_scores) / len(dice_scores)) if dice_scores else 0.0
    
    def train(self, data_paths):
        """Enhanced training loop with progress tracking"""
        print("\nStarting enhanced supervised training...")
        
        # Create datasets
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True,
            augment=True
        )
    
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        best_dice = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            epoch_losses = []
            epoch_lr = []
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            progress_bar = tqdm(train_ds, desc="Training")
            
            for batch in progress_bar:
                images, labels = batch
                loss, _, lr = self.train_step(images, labels)
                epoch_losses.append(float(loss))
                epoch_lr.append(float(lr))
                progress_bar.set_postfix({"loss": f"{float(loss):.4f}"})
                
            # Call garbage collection to free memory
            gc.collect()
            
            # Validation
            val_dice = self.validate(val_ds)
                
            # Update history
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(float(sum(epoch_losses) / len(epoch_losses)))
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(float(sum(epoch_lr) / len(epoch_lr)))
            
            # Logging
            print(f"Time: {epoch_time:.2f}s | Loss: {self.history['train_loss'][-1]:.4f} | Val Dice: {val_dice:.4f} | LR: {self.history['learning_rate'][-1]:.8e}")
            
            # Save checkpoint every epoch
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = Path(f"{self.config.checkpoint_dir}/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint(f'best_supervised_enhanced_{timestamp}')
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
    
    def plot_progress(self):
        """Plot training progress"""
        if len(self.history['train_loss']) < 2:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot Dice score
        ax2.plot(self.history['val_dice'])
        ax2.set_title('Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(self.history['learning_rate'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        plt.tight_layout()
        output_dir = Path("/scratch/lustre/home/mdah0000/smm/v14/supervised_results_v4_memory_fix/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/supervised_enhanced_progress.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      help='Path to data directory', 
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--experiment_name', type=str, default='supervised_enhanced',
                      help='Name for this experiment')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, required=False,
                      default='/scratch/lustre/home/mdah0000/smm/v14/supervised_results_v4_memory_fix',
                      help='Directory to save results')
    args = parser.parse_args()

    # Set up GPU
    setup_gpu()
    
    # Create config
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.checkpoint_dir = args.output_dir
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='supervised',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, num_labeled=225, num_validation=56)
    
    # Create data pipeline
    print("Creating data pipeline...")
    data_pipeline = DataPipeline(config)
    
    # Create trainer
    print("Creating enhanced supervised trainer...")
    trainer = EnhancedSupervisedTrainer(config)
    trainer.data_pipeline = data_pipeline
    
    # Print experiment info
    print("\nExperiment Configuration:")
    print(f"Training Type: supervised (enhanced)")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    print(f"Output directory: {config.checkpoint_dir}")
    
    # Create experiment directory
    exp_dir = Path(config.checkpoint_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    trainer.train(data_paths)
    
if __name__ == "__main__":
    main()