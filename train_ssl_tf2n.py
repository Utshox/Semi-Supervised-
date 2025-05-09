import tensorflow as tf
import numpy as np
from pathlib import Path
import time
import logging
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from models_tf2 import *
from data_loader_tf2 import DataPipeline, PancreasDataLoader

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# GPU setup with direct placement fallback
def setup_gpu():
    """Setup GPU for training with TF 2.18.0 compatibility and direct device placement"""
    logger.info("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    
    # Try standard GPU setup first
    if gpus:
        try:
            # Use all available GPUs and allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Log GPU information
            logger.info(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                logger.info(f"  {gpu.device_type}: {gpu.name}")
                
            # Set mixed precision policy - using mixed_float16 for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info(f"Using {policy.name} precision")
            
            return True
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    
    # Fallback to checking if nvidia-smi works
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected through nvidia-smi")
            
            # Try a simple GPU operation to confirm usability
            try:
                with tf.device('/device:GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                    c = tf.matmul(a, b)
                    _ = c.numpy()  # Force execution
                    logger.info("GPU is accessible via direct device placement!")
                    
                    # Set mixed precision policy for better performance
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info(f"Using {policy.name} precision with direct GPU access")
                    return True
            except Exception as e:
                logger.error(f"Error using GPU via direct placement: {e}")
        else:
            logger.warning("No GPU detected via nvidia-smi")
    except Exception:
        pass
    
    logger.warning("No GPU found. Using CPU.")
    return False

# Create a TensorFlow distributed strategy that works with or without GPUs
def create_strategy():
    """Create a TensorFlow distributed strategy that works even with detection issues"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            return tf.distribute.MirroredStrategy()
        except:
            logger.warning("Could not create MirroredStrategy, falling back to default strategy")
            return tf.distribute.get_strategy()
    
    # Try direct GPU access
    try:
        with tf.device('/device:GPU:0'):
            a = tf.constant(1)
            _ = a.numpy()
        return tf.distribute.OneDeviceStrategy(device="/device:GPU:0")
    except:
        logger.warning("Using default strategy with CPU")
        return tf.distribute.get_strategy()

def prepare_data_paths(data_dir, num_labeled=60, num_validation=63):
    print("Finding data pairs...")
    all_image_paths = []
    all_label_paths = []
    
    for folder in sorted(data_dir.glob('pancreas_*')):
        folder_no_nii = str(folder).replace('.nii', '')
        img_path = Path(folder_no_nii) / 'img_cropped.npy'
        mask_path = Path(folder_no_nii) / 'mask_cropped.npy'
        
        if img_path.exists() and mask_path.exists():
            all_image_paths.append(str(img_path))
            all_label_paths.append(str(mask_path))

    # Create splits
    train_images = all_image_paths[:num_labeled]
    train_labels = all_label_paths[:num_labeled]
    val_images = all_image_paths[-num_validation:]
    val_labels = all_label_paths[-num_validation:]
    unlabeled_images = all_image_paths[num_labeled:-num_validation]

    return {
        'labeled': {
            'images': train_images,
            'labels': train_labels,
        },
        'unlabeled': {
            'images': unlabeled_images
        },
        'validation': {
            'images': val_images,
            'labels': val_labels,
        }
    }


def masked_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy that ignores None values"""
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Apply sigmoid if needed (for logits)
    y_pred = tf.nn.sigmoid(y_pred)
    
    # Calculate binary crossentropy
    bce = -(y_true * tf.math.log(y_pred + 1e-7) + 
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    
    # Return mean of non-zero elements
    return tf.reduce_mean(bce)





#### 14jun 

class StableDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]

        y_pred = tf.nn.sigmoid(y_pred)
        
        reduction_axes = [1, 2]
        intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_axes)
        sum_true = tf.reduce_sum(y_true, axis=reduction_axes)
        sum_pred = tf.reduce_sum(y_pred, axis=reduction_axes)

        dice = (2. * intersection + self.smooth) / (sum_true + sum_pred + self.smooth)
        dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
        dice = tf.where(tf.math.is_inf(dice), tf.ones_like(dice), dice)
        
        return 1.0 - tf.reduce_mean(dice)


class StableSSLTrainer:
    def __init__(self, config, data_pipeline=None, labeled_data_size=2):
        print("Initializing StableSSLTrainer...")
        self.config = config
        self.labeled_data_size = labeled_data_size
        
        # Initialize data pipeline
        self.data_pipeline = data_pipeline if data_pipeline else DataPipeline(config)
        
        # Initialize models
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy initial weights
        self.teacher_model.set_weights(self.student_model.get_weights())
        
        # Setup training parameters
        self._setup_training_params()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': []
        }
        
        # Add patience attribute for early stopping
        self.patience = 20
        
        print("Initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('ssl_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def _setup_training_params(self):
        """Setup training parameters with more dynamic learning"""
        # Lower initial learning rate with warmup
        initial_lr = 1e-4
        max_lr = 1e-4
        
        # CosineDecayRestarts schedule - fix by converting to float value
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=500,  # Adjust based on your dataset size
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0,
        )
        
        # Create optimizer with initial float value instead of scheduled value
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(initial_lr))
        
        # Loss function with larger smooth factor
        self.supervised_loss_fn = StableDiceLoss(smooth=10.0)
        
        # Consistency weight parameters
        self.initial_consistency_weight = 0.0
        self.max_consistency_weight = 1.0
        self.rampup_length = 4000

    def train_step(self, batch):
        if len(batch) == 2:
            images, labels = batch
            unlabeled_images = None
        else:
            images, labels, unlabeled_images = batch

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
    
        # Z-score normalization
        mean = tf.reduce_mean(images)
        std = tf.math.reduce_std(images) + 1e-6
        images = (images - mean) / std

        with tf.GradientTape() as tape:
            # Get predictions
            student_logits = self.student_model(images, training=True)
            
            if unlabeled_images is not None:
              # Calculate consistency loss only if unlabeled images are present
              teacher_logits = self.teacher_model(unlabeled_images, training=False)

            # Calculate supervised loss
            supervised_loss = self.supervised_loss_fn(labels, student_logits)

            # Calculate consistency weight with cosine rampup
            current_step = tf.cast(self.optimizer.iterations, tf.float32)
            rampup_ratio = tf.minimum(1.0, current_step / self.rampup_length)
            consistency_weight = self.max_consistency_weight * rampup_ratio  # Changed

            # MSE on probabilities
            if unlabeled_images is not None:
              teacher_probs = tf.nn.sigmoid(teacher_logits)
              student_probs = tf.nn.sigmoid(student_logits)
              consistency_loss = tf.reduce_mean(tf.square(teacher_probs - student_probs))
            else:
              consistency_loss = 0.0 # No consistency loss if no unlabeled images

            # Total loss with gradient scaling
            total_loss = supervised_loss + consistency_weight * consistency_loss

            # Scale loss for numerical stability
            scaled_loss = total_loss * 0.1

        # Compute and apply gradients with gradient accumulation
        gradients = tape.gradient(scaled_loss, self.student_model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

        # Update student model
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model
        self._update_teacher()

        # Print debug info occasionally
        if self.optimizer.iterations % 100 == 0:
            tf.print("\nStep:", self.optimizer.iterations)
            tf.print("Learning rate:", self.optimizer.learning_rate)
            tf.print("Consistency weight:", consistency_weight)
            tf.print("Supervised loss:", supervised_loss)
            tf.print("Consistency loss:", consistency_loss)

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss
        }

    def _update_teacher(self):
        """Update teacher model with dynamic EMA"""
        # Ramp up EMA decay from 0.95 to 0.999
        current_step = tf.cast(self.optimizer.iterations, tf.float32)
        warmup_steps = 1000.0
        current_decay = 0.95 + 0.049 * tf.minimum(1.0, current_step / warmup_steps)
        
        # Debug print every 100 steps
        if current_step % 100 == 0:
            tf.print("\nCurrent EMA decay:", current_decay)
         # ADD THESE PRINT STATEMENTS
        tf.print("Student weights before update:", tf.reduce_mean(self.student_model.trainable_variables[0]))
        tf.print("Teacher weights before update:", tf.reduce_mean(self.teacher_model.trainable_variables[0]))


        for student_var, teacher_var in zip(
                self.student_model.trainable_variables,
                self.teacher_model.trainable_variables):
            delta = current_decay * (teacher_var - student_var)
            teacher_var.assign_sub(delta)
            
        # Verify update
        if current_step % 100 == 0:
            student_weights = self.student_model.get_weights()[0]
            teacher_weights = self.teacher_model.get_weights()[0]
            tf.print("Weight diff:", tf.reduce_mean(tf.abs(student_weights - teacher_weights)))

    def _compute_dice(self, y_true, y_pred):
        """Compute Dice score"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]
        
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        sum_true = tf.reduce_sum(y_true, axis=[1, 2])
        sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
        dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
        
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Run validation"""
        dice_scores = []
        teacher_dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            
            student_logits = self.student_model(images, training=False)
            teacher_logits = self.teacher_model(images, training=False)
            
            student_dice = self._compute_dice(labels, student_logits)
            teacher_dice = self._compute_dice(labels, teacher_logits)
            
            if not tf.math.is_nan(student_dice) and not tf.math.is_nan(teacher_dice):
                dice_scores.append(float(student_dice))
                teacher_dice_scores.append(float(teacher_dice))
        
        if dice_scores and teacher_dice_scores:
            return np.mean(dice_scores), np.mean(teacher_dice_scores)
        return 0.0, 0.0

    def train(self, data_paths):
        """Main training loop"""
        print("\nStarting training...")
        
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        best_dice = 0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Training
            epoch_losses = []
            supervised_losses = []
            consistency_losses = []

            for batch in train_ds:
                metrics = self.train_step(batch)
                epoch_losses.append(float(metrics['total_loss']))
                supervised_losses.append(float(metrics['supervised_loss']))
                consistency_losses.append(float(metrics['consistency_loss']))

            # Validation
            val_dice, teacher_dice = self.validate(val_ds)

            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['supervised_loss'].append(np.mean(supervised_losses))
            self.history['consistency_loss'].append(np.mean(consistency_losses))
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )

            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - epoch_start:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"Teacher Dice: {teacher_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_model')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                print("\nEarly stopping triggered!")
                break

            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()

        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        try:
            student_path = self.checkpoint_dir / f'{name}_student'
            teacher_path = self.checkpoint_dir / f'{name}_teacher'
            
            self.student_model.save_weights(str(student_path))
            self.teacher_model.save_weights(str(teacher_path))
            print(f"Saved checkpoint: {name}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def plot_progress(self):
        """Plot training progress"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Total Loss')
        ax.plot(self.history['supervised_loss'], label='Supervised Loss')
        ax.plot(self.history['consistency_loss'], label='Consistency Loss')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot Dice scores
        ax = axes[0, 1]
        ax.plot(self.history['val_dice'], label='Student Dice')
        ax.plot(self.history['teacher_dice'], label='Teacher Dice')
        ax.set_title('Validation Dice Scores')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.legend()
        
        # Plot learning rate
        ax = axes[1, 0]
        ax.plot(self.history['learning_rate'])
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'training_progress_{len(self.history["train_loss"])}.png')
        plt.close()

    def plot_final_results(self):
        """Create final training summary plots"""
        self.plot_progress()  # Save final progress plot
        
        # Save history to CSV
        import pandas as pd
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir / 'training_history.csv'}")
#########################
#loss
         
class StableConsistencyLoss(tf.keras.losses.Loss):
    """Stable Consistency Loss with temperature scaling"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def call(self, teacher_logits, student_logits):
        # Apply temperature scaling
        teacher_probs = tf.nn.sigmoid(teacher_logits / self.temperature)
        student_probs = tf.nn.sigmoid(student_logits / self.temperature)
        
        # Calculate MSE loss
        return tf.reduce_mean(tf.square(teacher_probs - student_probs))

class CombinedSSLLoss(tf.keras.losses.Loss):
    """Combined loss for semi-supervised learning"""
    def __init__(self, dice_weight=1.0, consistency_weight=1.0):
        super().__init__()
        self.dice_loss = StableDiceLoss()
        self.consistency_loss = StableConsistencyLoss()
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
    
    def call(self, y_true, y_pred, teacher_pred=None):
        losses = {'dice_loss': self.dice_loss(y_true, y_pred)}
        
        if teacher_pred is not None:
            losses['consistency_loss'] = self.consistency_loss(teacher_pred, y_pred)
            total_loss = (self.dice_weight * losses['dice_loss'] + 
                         self.consistency_weight * losses['consistency_loss'])
        else:
            total_loss = losses['dice_loss']
        
        return total_loss
#########################
class ImprovedSSLTrainer:
    def __init__(self, config, labeled_data_size=2):
        print("Initializing ImprovedSSLTrainer...")
        self.config = config
        self.labeled_data_size = labeled_data_size
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # Initialize models
        print("Creating models...")
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config)
        
        # Initialize with dummy input to create weights
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy initial weights
        print("Copying initial weights...")
        self.teacher_model.set_weights(self.student_model.get_weights())
        
        # Training parameters
        initial_lr = 0.0001
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr, 
            first_decay_steps=100,
            t_mul=2.0,
            m_mul=0.95,
            alpha=0.2
        )

        self.patience = 20
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # Loss functions
        self.supervised_loss_fn = CombinedLoss(alpha=0.5, beta=0.5)
        self.consistency_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Modify EMA decay
        self.initial_ema_decay = 0.95  # Start with lower EMA decay
        self.final_ema_decay = 0.999   # End with high EMA decay
        # Create directories
        self.setup_directories()
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'consistency_loss': [],
            'learning_rate': []
        }
        print("Initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    @tf.function
    def train_step(self, batch, training=True):
        images, labels = batch
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        # Add Gaussian noise for consistency regularization
        if training:
            noised_images = images + tf.random.normal(tf.shape(images), 0, 0.1)
            noised_images = tf.clip_by_value(noised_images, 0, 1)
        else:
            noised_images = images
        
        with tf.GradientTape() as tape:
            # Get predictions
            student_logits = self.student_model(noised_images, training=True)
            teacher_logits = self.teacher_model(images, training=False)
            
            # Calculate supervised loss with class weights
            supervised_loss = self.supervised_loss_fn(labels, student_logits)
            
            # Calculate consistency loss with ramp-up
            consistency_loss = self.consistency_loss_fn(
                tf.nn.sigmoid(teacher_logits),
                tf.nn.sigmoid(student_logits)
            )
            
            # Dynamic consistency weight
            current_step = tf.cast(self.optimizer.iterations, tf.float32)
            rampup_length = 1000.0  # Number of steps for ramp-up
            rampup = tf.minimum(1.0, current_step / rampup_length)
            consistency_weight = 20.0 * rampup  # Increased from 0.1
            
            # Total loss
            total_loss = supervised_loss + consistency_weight * consistency_loss
        
        if training:
            # Gradient clipping
            gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]  # Increased from 1.0
            self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
            
            # Dynamic EMA decay
            current_ema = self.initial_ema_decay + (self.final_ema_decay - self.initial_ema_decay) * rampup
            self.update_teacher(current_ema)
        
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'student_logits': student_logits,
            'teacher_logits': teacher_logits
        }

    def update_teacher(self, current_ema):
        """Update teacher model weights using current EMA decay."""
        for student_weights, teacher_weights in zip(
                self.student_model.trainable_variables,
                self.teacher_model.trainable_variables):
            teacher_weights.assign(
                current_ema * teacher_weights + 
                (1 - current_ema) * student_weights
            )
    def compute_dice(self, y_true, y_pred):
        """Compute Dice score."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Run validation."""
        dice_scores = []
        teacher_dice_scores = []
        
        for batch in val_dataset:
            metrics = self.train_step(batch, training=False)
            
            # Compute Dice scores
            student_dice = self.compute_dice(batch[1], metrics['student_logits'])
            teacher_dice = self.compute_dice(batch[1], metrics['teacher_logits'])
            
            dice_scores.append(student_dice)
            teacher_dice_scores.append(teacher_dice)
        
        return np.mean(dice_scores), np.mean(teacher_dice_scores)

    def train(self, data_paths):
        """Main training loop"""
        self.print_debug_info(data_paths) 
        print("Setting up datasets...")
        # Setup datasets
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        print("Starting training loop...")
        best_dice = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            start_time = time.time()
            
            # Training
            epoch_losses = []
            for batch in train_ds:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['total_loss'])
            
            # Validation
            val_dice, teacher_dice = self.validate(val_ds)
            
            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(self.lr_schedule(self.optimizer.iterations))
            
            # Logging
            print(f"Time taken: {time.time() - start_time:.2f}s")
            print(f"Training Loss: {np.mean(epoch_losses):.4f}")
            print(f"Validation Dice: {val_dice:.4f}")
            print(f"Teacher Dice: {teacher_dice:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_model')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")

    def generate_report_visualizations(trainer, val_dataset, save_dir):
        """Generate all visualizations for the report"""
        visualizer = PerformanceVisualizer(save_dir)
        
        # Plot training history
        visualizer.plot_training_history(trainer.history)
        
        # Plot sample predictions
        for batch in val_dataset.take(1):  # Take one batch
            images, labels = batch
            student_preds = trainer.student_model(images, training=False)
            teacher_preds = trainer.teacher_model(images, training=False)
            visualizer.plot_sample_predictions(
                images.numpy(),
                labels.numpy(),
                student_preds.numpy(),
                teacher_preds.numpy()
            )
        
        # Plot performance range
        visualizer.plot_performance_range()
        
        # Save performance summary
        summary = visualizer.save_performance_summary(trainer.history)
        
        print(f"\nPerformance Summary:")
        print(summary.to_string(index=False))
        print(f"\nVisualizations saved in: {save_dir}")
        
        return save_dir
    def save_checkpoint(self, name, timestamp=None):
        """Save model checkpoint and training history"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save models
        checkpoint_dir = self.checkpoint_dir / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        student_path = checkpoint_dir / f'{name}_student'
        teacher_path = checkpoint_dir / f'{name}_teacher'
        self.student_model.save_weights(str(student_path))
        self.teacher_model.save_weights(str(teacher_path))
        
        # Save training history
        history_path = checkpoint_dir / f'{name}_history.npy'
        np.save(history_path, self.history)
        
        # Save training metrics summary
        metrics = {
            'best_dice': max(self.history['val_dice']),
            'final_teacher_dice': self.history['teacher_dice'][-1],
            'num_epochs': len(self.history['train_loss']),
            'timestamp': timestamp
        }
        
        metrics_path = checkpoint_dir / f'{name}_metrics.npy'
        np.save(metrics_path, metrics)
        
        print(f"Saved checkpoint and history to {checkpoint_dir}")

    def plot_progress(self, timestamp=None):
        """Plot training progress with timestamp"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        plot_dir = self.plot_dir / timestamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        axes[0, 0].plot(self.history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Plot Dice scores
        axes[0, 1].plot(self.history['val_dice'], label='Student')
        axes[0, 1].plot(self.history['teacher_dice'], label='Teacher')
        axes[0, 1].set_title('Validation Dice Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        
        # Plot learning rate
        axes[1, 0].plot(self.history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        
        # Save plot with timestamp
        plot_path = plot_dir / f'training_progress.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, filename):
        results = {
            'history': self.history,
            'best_dice': max(self.history['val_dice']),
            'final_teacher_dice': self.history['teacher_dice'][-1],
            'num_epochs': len(self.history['train_loss'])
        }
        np.save(filename, results)

    def print_debug_info(self, data_paths):
        """Print debug information about the data"""
        print("\nDebug Information:")
        print("Labeled images:", len(data_paths['labeled']['images']))
        print("Labeled labels:", len(data_paths['labeled']['labels']))
        print("Validation images:", len(data_paths['validation']['images']))
        print("Validation labels:", len(data_paths['validation']['labels']))
        
        # Print sample path
        print("\nSample paths:")
        print("Image path:", data_paths['labeled']['images'][0])
        print("Label path:", data_paths['labeled']['labels'][0])
        
        # Try loading a sample
        try:
            sample_img = np.load(data_paths['labeled']['images'][0])
            sample_label = np.load(data_paths['labeled']['labels'][0])
            print("\nSample shapes:")
            print("Image shape:", sample_img.shape)
            print("Label shape:", sample_label.shape)
            print("Image dtype:", sample_img.dtype)
            print("Label dtype:", sample_label.dtype)
            print("Image range:", np.min(sample_img), "-", np.max(sample_img))
            print("Label range:", np.min(sample_label), "-", np.max(sample_label))
        except Exception as e:
            print("Error loading sample:", e)
######################################
class HighPerformanceSSLTrainer(ImprovedSSLTrainer):
    def __init__(self, config, labeled_data_size=2):
        super().__init__(config, labeled_data_size)
        
        # Use exact hyperparameters from successful run
        initial_lr = 5e-3  # Changed back to original value
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr, 
            first_decay_steps=100,  # Increased from 50
            t_mul=2.0,
            m_mul=0.95,  # Increased from 0.9
            alpha=0.2    # Increased from 0.1
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # EMA decay parameters
        self.initial_ema_decay = 0.95
        self.final_ema_decay = 0.999
        
        # Add seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Modify batch size
        config.batch_size = 16  # Set explicit batch size

    def update_teacher(self, rampup_value):
        """Update teacher model weights using EMA with ramp-up."""
        # Calculate current ema decay
        current_ema = self.initial_ema_decay + (self.final_ema_decay - self.initial_ema_decay) * rampup_value
        
        for student_weights, teacher_weights in zip(
                self.student_model.trainable_variables,
                self.teacher_model.trainable_variables):
            teacher_weights.assign(
                current_ema * teacher_weights + 
                (1 - current_ema) * student_weights
            )

    @tf.function
    def train_step(self, batch, training=True):
        images, labels = batch
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        # Add Gaussian noise for consistency regularization
        if training:
            noised_images = images + tf.random.normal(tf.shape(images), 0, 0.1)
            noised_images = tf.clip_by_value(noised_images, 0, 1)
        else:
            noised_images = images
        
        with tf.GradientTape() as tape:
            # Student predictions with noise
            student_logits = self.student_model(noised_images, training=True)
            
            # Teacher predictions without noise
            teacher_logits = self.teacher_model(images, training=False)
            
            # Calculate supervised loss
            supervised_loss = self.supervised_loss_fn(labels, student_logits)
            
            # Calculate consistency loss
            consistency_loss = self.consistency_loss_fn(
                tf.nn.sigmoid(teacher_logits),
                tf.nn.sigmoid(student_logits)
            )
            
            # Dynamic ramp-up
            current_step = tf.cast(self.optimizer.iterations, tf.float32)
            rampup_length = 1000.0  # Number of steps for ramp-up
            rampup_value = tf.minimum(1.0, current_step / rampup_length)
            consistency_weight = 20.0 * rampup_value  # Higher max weight
            
            # Total loss
            total_loss = supervised_loss + consistency_weight * consistency_loss
        
        if training:
            # Gradient clipping
            gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]  # Increased from 1.0
            self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
            
            # Update teacher model with current rampup value
            self.update_teacher(rampup_value)
        
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'student_logits': student_logits,
            'teacher_logits': teacher_logits
        }

        
        for batch in val_dataset:
            metrics = self.train_step(batch, training=False)
            
            # Compute Dice scores
            student_dice = self.compute_dice(batch[1], metrics['student_logits'])
            teacher_dice = self.compute_dice(batch[1], metrics['teacher_logits'])
            
            dice_scores.append(student_dice)
            teacher_dice_scores.append(teacher_dice)
        
        return np.mean(dice_scores), np.mean(teacher_dice_scores)
        visualization_path = generate_report_figures(trainer, val_dataset, 'mean_teacher_results')
        print(f"Generated report figures in: {visualization_path}")
    def compute_dice(self, y_true, y_pred):
        """Compute Dice score with sigmoid activation."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.nn.sigmoid(y_pred)  # Apply sigmoid to logits
        
        # Take pancreas channel
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
        if len(y_pred.shape) > 3:
            y_pred = y_pred[..., -1]
            
        # Threshold predictions
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        # Calculate Dice
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)
    def generate_report_figures(trainer, val_dataset, save_dir='report_figures'):
        """Generate comprehensive figures for the report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 1. Training History
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.history['train_loss'], label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(trainer.history['val_dice'], label='Student Dice')
        plt.plot(trainer.history['teacher_dice'], label='Teacher Dice')
        plt.title('Validation Dice Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample Predictions
        for batch_idx, (images, labels) in enumerate(val_dataset.take(1)):
            student_preds = trainer.student_model(images, training=False)
            teacher_preds = trainer.teacher_model(images, training=False)
            
            # Plot 4 examples
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            for i in range(4):
                # Original
                axes[i, 0].imshow(images[i, ..., 0], cmap='gray')
                axes[i, 0].set_title('Input Image')
                
                # Ground Truth
                axes[i, 1].imshow(labels[i, ..., -1], cmap='RdYlBu')
                axes[i, 1].set_title('Ground Truth')
                
                # Student Prediction
                axes[i, 2].imshow(tf.nn.sigmoid(student_preds[i, ..., -1]), cmap='RdYlBu')
                axes[i, 2].set_title(f'Student Pred (Dice={trainer.compute_dice(labels[i:i+1], student_preds[i:i+1]):.3f})')
                
                # Teacher Prediction
                axes[i, 3].imshow(tf.nn.sigmoid(teacher_preds[i, ..., -1]), cmap='RdYlBu')
                axes[i, 3].set_title(f'Teacher Pred (Dice={trainer.compute_dice(labels[i:i+1], teacher_preds[i:i+1]):.3f})')
                
                # Turn off axes
                for ax in axes[i]:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'sample_predictions_batch_{batch_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance Summary Table
        performance_data = {
            'Metric': ['Best Validation Dice', 'Final Teacher Dice', 'Final Training Loss'],
            'Value': [
                max(trainer.history['val_dice']),
                trainer.history['teacher_dice'][-1],
                trainer.history['train_loss'][-1]
            ]
        }
        
        # Save as CSV
        pd.DataFrame(performance_data).to_csv(save_dir / 'performance_summary.csv', index=False)
        
        return save_dir    
############################################
class MixMatchTrainer(ImprovedSSLTrainer):
    def __init__(self, config, data_pipeline=None, labeled_data_size=2):
        # Call parent constructor but override data_pipeline
        super().__init__(config, labeled_data_size)
        
        # Override data_pipeline if provided
        if data_pipeline is not None:
            self.data_pipeline = data_pipeline
            
        self.T = 0.5  # Sharpening temperature
        self.K = 2    # Number of augmentations
        self.alpha = 0.75  # Beta distribution parameter
        
        # Initialize history correctly
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'consistency_loss': [],
            'learning_rate': [],
            'supervised_loss': []  # Added this
        }
        
        # Reduce batch size for MixMatch
        self.batch_size = config.batch_size // 2  # Halve the batch size

    @tf.function(reduce_retracing=True)
    def train_step(self, labeled_batch, unlabeled_batch):
        if labeled_batch[0] is None or unlabeled_batch is None:
            return {
                'total_loss': 0.0,
                'supervised_loss': 0.0,
                'consistency_loss': 0.0
            }

        images_l, labels_l = labeled_batch
        images_u = unlabeled_batch
        
        # Ensure proper types
        images_l = tf.cast(images_l, tf.float32)
        labels_l = tf.cast(labels_l, tf.float32)
        images_u = tf.cast(images_u, tf.float32)

        with tf.GradientTape() as tape:
            # Generate pseudo-labels with fewer augmentations
            teacher_pred = self.teacher_model(images_u, training=False)
            pseudo_labels = self.sharpen(teacher_pred, self.T)

            # MixUp with smaller batch
            shuffle = tf.random.shuffle(tf.range(tf.shape(images_l)[0]))
            images_l_mix, labels_l_mix = self.mixup(
                images_l, tf.gather(images_l, shuffle),
                labels_l, tf.gather(labels_l, shuffle),
                self.alpha
            )

            # Student predictions
            logits_l = self.student_model(images_l_mix, training=True)
            logits_u = self.student_model(images_u, training=True)

            # Calculate losses
            supervised_loss = self.supervised_loss_fn(labels_l_mix, logits_l)
            consistency_loss = self.consistency_loss_fn(pseudo_labels, logits_u)
            
            # Ramp up consistency weight
            consistency_weight = 10.0 * tf.minimum(1.0, tf.cast(self.optimizer.iterations / 500, tf.float32))
            
            # Total loss
            total_loss = supervised_loss + consistency_weight * consistency_loss

        # Optimize
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher
        self.update_teacher()

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss
        }

    def train(self, data_paths):
        print("\nInitializing MixMatch training...")
        
        # Setup datasets with smaller batch size
        train_labeled_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.batch_size,
            shuffle=True
        )
        
        train_unlabeled_ds = self.data_pipeline.dataloader.create_unlabeled_dataset(
            data_paths['unlabeled']['images'],
            batch_size=self.batch_size
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.batch_size,
            shuffle=False
        )

        # Training loop
        best_dice = 0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            start_time = time.time()
            
            # Training
            epoch_losses = []
            supervised_losses = []
            consistency_losses = []
            
            for labeled_batch in train_labeled_ds:
                for unlabeled_batch in train_unlabeled_ds.take(1):  # Take only one unlabeled batch
                    metrics = self.train_step(labeled_batch, unlabeled_batch)
                    epoch_losses.append(metrics['total_loss'])
                    supervised_losses.append(metrics['supervised_loss'])
                    consistency_losses.append(metrics['consistency_loss'])
            
            # Validation
            val_dice, teacher_dice = self.validate(val_ds)
            
            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['supervised_loss'].append(np.mean(supervised_losses))
            self.history['consistency_loss'].append(np.mean(consistency_losses))
            
            # Logging
            print(f"Time taken: {time.time() - start_time:.2f}s")
            print(f"Training Loss: {np.mean(epoch_losses):.4f}")
            print(f"Validation Dice: {val_dice:.4f}")
            print(f"Teacher Dice: {teacher_dice:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mixmatch_model')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 20:
                print("Early stopping triggered")
                break
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")

############################################

    def __init__(self, config, labeled_data_size=2):
        super().__init__(config, labeled_data_size)
        self.T = 0.5  # Sharpening temperature
        self.K = 2    # Number of augmentations
        self.alpha = 0.75  # Beta distribution parameter
        
    def sharpen(self, x, T):
        """Sharpening function for MixMatch"""
        x = tf.nn.softmax(x / T, axis=-1)
        return x

    def mixup(self, x1, x2, y1, y2, alpha=0.75):
        """Mixup data augmentation"""
        beta = tf.random.beta([tf.shape(x1)[0]], alpha, alpha)
        beta = tf.maximum(beta, 1-beta)
        x = beta * x1 + (1-beta) * x2
        y = beta * y1 + (1-beta) * y2
        return x, y

    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch):
        images_l, labels_l = labeled_batch
        images_u = unlabeled_batch
        batch_size = tf.shape(images_l)[0]

        with tf.GradientTape() as tape:
            # Generate pseudo-labels for unlabeled data
            logits_u = []
            for _ in range(self.K):
                logits_u.append(self.teacher_model(images_u, training=False))
            logits_u = tf.reduce_mean(tf.stack(logits_u, axis=0), axis=0)
            pseudo_labels = self.sharpen(logits_u, self.T)

            # MixUp
            shuffle = tf.random.shuffle(tf.range(batch_size))
            images_l_mix, labels_l_mix = self.mixup(
                images_l, tf.gather(images_l, shuffle),
                labels_l, tf.gather(labels_l, shuffle),
                self.alpha
            )
            
            images_u_mix, pseudo_labels_mix = self.mixup(
                images_u, tf.gather(images_u, shuffle),
                pseudo_labels, tf.gather(pseudo_labels, shuffle),
                self.alpha
            )

            # Forward pass
            logits_l = self.student_model(images_l_mix, training=True)
            logits_u = self.student_model(images_u_mix, training=True)

            # Calculate losses
            supervised_loss = self.supervised_loss_fn(labels_l_mix, logits_l)
            consistency_loss = self.consistency_loss_fn(pseudo_labels_mix, logits_u)
            
            # Total loss
            consistency_weight = 100 * tf.minimum(1.0, tf.cast(self.optimizer.iterations / 100, tf.float32))
            total_loss = supervised_loss + consistency_weight * consistency_loss

        # Calculate gradients
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model
        self.update_teacher()

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'student_logits': logits_l
        }

    def train(self, data_paths):
        """Main training loop for MixMatch"""
        # Setup datasets
        train_labeled_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        train_unlabeled_ds = self.data_pipeline.dataloader.create_unlabeled_dataset(
            data_paths['unlabeled']['images'],
            batch_size=self.config.batch_size
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Convert to iterators
        train_labeled_iter = iter(train_labeled_ds.repeat())
        train_unlabeled_iter = iter(train_unlabeled_ds.repeat())
        
        best_dice = 0
        patience = 10
        patience_counter = 0
        
        steps_per_epoch = len(list(train_unlabeled_ds))
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            start_time = time.time()
            
            # Training
            epoch_losses = []
            for step in range(steps_per_epoch):
                # Get next batches
                labeled_batch = next(train_labeled_iter)
                unlabeled_batch = next(train_unlabeled_iter)
                
                # Training step
                metrics = self.train_step(labeled_batch, unlabeled_batch)
                epoch_losses.append(metrics['total_loss'])
                
                if step % 10 == 0:
                    print(f"Step {step}/{steps_per_epoch}, Loss: {metrics['total_loss']:.4f}")
            
            # Validation
            val_dice, teacher_dice = self.validate(val_ds)
            
            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['learning_rate'].append(self.lr_schedule(self.optimizer.iterations))
            
            # Logging
            print(f"Time taken: {time.time() - start_time:.2f}s")
            print(f"Training Loss: {np.mean(epoch_losses):.4f}")
            print(f"Validation Dice: {val_dice:.4f}")
            print(f"Teacher Dice: {teacher_dice:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mixmatch_model')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")

# Add the SupervisedTrainer class after the existing trainer classes
class SupervisedTrainer:
    def __init__(self, config, data_pipeline=None, labeled_data_size=2):
        print("Initializing SupervisedTrainer...")
        self.config = config
        self.labeled_data_size = labeled_data_size
        
        # Initialize data pipeline
        self.data_pipeline = data_pipeline if data_pipeline else DataPipeline(config)
        
        # Initialize model - single model for supervised learning
        self.model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.model(dummy_input)
        
        # Setup training parameters
        self._setup_training_params()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        # Add patience attribute for early stopping
        self.patience = 20
        
        print("Initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('supervised_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def _setup_training_params(self):
        """Setup training parameters"""
        # Initial learning rate
        initial_lr = 1e-4
        
        # Cosine decay learning rate schedule
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=500,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0,
        )
        
        # Create optimizer with initial float value
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(initial_lr))
        
        # Loss function
        self.loss_fn = StableDiceLoss(smooth=10.0)

    def train_step(self, batch):
        images, labels = batch

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
    
        # Z-score normalization
        mean = tf.reduce_mean(images)
        std = tf.math.reduce_std(images) + 1e-6
        images = (images - mean) / std

        with tf.GradientTape() as tape:
            # Get predictions
            logits = self.model(images, training=True)
            
            # Calculate loss
            loss = self.loss_fn(labels, logits)
            
            # Scale loss for numerical stability
            scaled_loss = loss * 0.1

        # Compute and apply gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

        # Update model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Print debug info occasionally
        if self.optimizer.iterations % 100 == 0:
            tf.print("\nStep:", self.optimizer.iterations)
            tf.print("Learning rate:", self.optimizer.learning_rate)
            tf.print("Loss:", loss)

        return {
            'loss': loss
        }

    def _compute_dice(self, y_true, y_pred):
        """Compute Dice score"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]
        
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        sum_true = tf.reduce_sum(y_true, axis=[1, 2])
        sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
        dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
        
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Run validation"""
        dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            
            logits = self.model(images, training=False)
            
            dice = self._compute_dice(labels, logits)
            
            if not tf.math.is_nan(dice):
                dice_scores.append(float(dice))
        
        if dice_scores:
            return np.mean(dice_scores)
        return 0.0

    def train(self, data_paths):
        """Main training loop"""
        print("\nStarting supervised training...")
        
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        best_dice = 0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Training
            epoch_losses = []

            for batch in train_ds:
                metrics = self.train_step(batch)
                epoch_losses.append(float(metrics['loss']))

            # Validation
            val_dice = self.validate(val_ds)

            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )

            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - epoch_start:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Dice: {val_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_model')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                print("\nEarly stopping triggered!")
                break

            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()

        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        
        # Save final plots
        self.plot_final_results()
        
        return best_dice

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        try:
            model_path = self.checkpoint_dir / f'{name}'
            
            self.model.save_weights(str(model_path))
            print(f"Saved checkpoint: {name}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def plot_progress(self):
        """Plot training progress"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot loss
        ax = axes[0]
        ax.plot(self.history['train_loss'], label='Loss')
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot Dice score
        ax = axes[1]
        ax.plot(self.history['val_dice'], label='Validation Dice')
        ax.set_title('Validation Dice Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'training_progress_{len(self.history["train_loss"])}.png')
        plt.close()

    def plot_final_results(self):
        """Create final training summary plots"""
        self.plot_progress()  # Save final progress plot
        
        # Save history to CSV
        import pandas as pd
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir / 'training_history.csv'}")

