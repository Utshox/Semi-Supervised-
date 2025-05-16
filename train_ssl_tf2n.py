import tensorflow as tf
#import tensorflow_addons as tfa
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
from visualization import ExperimentVisualizer

#mean_teacher
#GPU
def setup_gpu():
    """Setup GPU for training"""
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Use all available GPUs and allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Log GPU information
            print(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu.device_type}: {gpu.name}")
                
            # Set mixed precision policy
            tf.keras.mixed_precision.set_global_policy('float32')  # Changed to float32 for stability
            print("Using float32 precision")
            
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU found. Using CPU.")
        return False

def prepare_data_paths(data_dir, num_labeled=20, num_validation=63):
    """Prepare data paths without verbose logging"""
    def get_case_number(path):
        return int(str(path).split('pancreas_')[-1][:3])

    # Gather paths silently
    all_image_paths = []
    all_label_paths = []
    
    for folder in sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x)):
        folder_no_nii = str(folder).replace('.nii', '')
        img_path = Path(folder_no_nii) / 'img_cropped.npy'
        mask_path = Path(folder_no_nii) / 'mask_cropped.npy'
        
        if img_path.exists() and mask_path.exists():
            all_image_paths.append(str(img_path))
            all_label_paths.append(str(mask_path))

    total_samples = len(all_image_paths)
    print(f"Found {total_samples} valid samples")
    
    if total_samples < (num_labeled + num_validation):
        raise ValueError(f"Not enough valid samples. Found {total_samples}, need at least {num_labeled + num_validation}")

    # Create splits
    train_images = all_image_paths[:num_labeled]
    train_labels = all_label_paths[:num_labeled]
    val_images = all_image_paths[-num_validation:]
    val_labels = all_label_paths[-num_validation:]
    unlabeled_images = all_image_paths[num_labeled:-num_validation]

    print(f"Data split - Labeled: {len(train_images)}, Unlabeled: {len(unlabeled_images)}, Validation: {len(val_images)}")

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

#############################################################
class SupervisedTrainer:
    def __init__(self, config):
        """Initialize supervised trainer"""
        print("Initializing Supervised Trainer...")
        self.config = config

        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # Initialize model
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
        
        print("Supervised trainer initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('supervised_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def _setup_training_params(self):
        """Setup training parameters"""
        initial_lr = 5e-4  # Higher learning rate for supervised
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=200,
            t_mul=1.5,
            m_mul=0.95,
            alpha=0.1
        )
        
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.99
        )
        
        self.loss_fn = StableDiceLoss(smooth=5.0)

    @tf.function
    def train_step(self, images, labels):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(images, training=True)
            
            # Calculate loss
            loss = self.loss_fn(labels, logits)
        
        # Calculate gradients and apply updates
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, logits

    def compute_dice(self, y_true, y_pred):
        """Compute Dice score"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, tf.float32)
        
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Run validation"""
        dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            predictions = self.model(images, training=False)
            dice_score = self.compute_dice(labels, predictions)
            dice_scores.append(float(dice_score))
        
        return np.mean(dice_scores)

    def train(self, data_paths):
        """Main training loop"""
        print("\nStarting supervised training...")
        
        # Create datasets
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
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            epoch_losses = []
            for batch in train_ds:
                images, labels = batch
                loss, _ = self.train_step(images, labels)
                epoch_losses.append(float(loss))
            
            # Validation every 2 epochs
            if epoch % 2 == 0:
                val_dice = self.validate(val_ds)
                
                # Update history
                self.history['val_dice'].append(val_dice)
                
                # Save best model
                if val_dice > best_dice:
                    best_dice = val_dice
                    self.save_checkpoint('best_supervised_model')
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    break
            
            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - start_time:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f}"
                  + (f" | Val Dice: {val_dice:.4f}" if epoch % 2 == 0 else ""))
            
            # Plot progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.checkpoint_dir / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f'{name}.weights.h5'
        self.model.save_weights(str(model_path))
        # Save history
        np.save(checkpoint_dir / f'{name}_history.npy', self.history)
        print(f"Saved checkpoint to {checkpoint_dir}")

    def plot_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_dice'])
        plt.title('Validation Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'training_progress_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()


#############################################################




class StableSSLTrainer:
    def __init__(self, config, labeled_data_size=2):
        print("Initializing StableSSLTrainer...")
        self.config = config
        self.labeled_data_size = labeled_data_size
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # Initialize models
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy initial weights
        self.teacher_model.set_weights(self.student_model.get_weights())
        
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
        # Higher initial learning rate with better decay parameters
        initial_lr = 5e-4
        
        # CosineDecayRestarts schedule with minimum learning rate
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=500,  # Adjust based on your dataset size
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1,  # Setting alpha to 0.1 ensures learning rate never goes below 10% of initial
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
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
        """Main training loop with validation"""
        print("\nStarting training...")
        print("\nValidating data paths...")
        
        # Validate data paths before creating datasets
        for key, paths in data_paths['labeled'].items():
            print(f"\nChecking labeled {key}...")
            for path in paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Labeled {key} path does not exist: {path}")
                try:
                    data = np.load(path)
                    print(f"Successfully loaded {path}, shape: {data.shape}")
                except Exception as e:
                    raise ValueError(f"Error loading {path}: {e}")
        
        print("\nCreating datasets...")
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
        
        # Validate datasets
        try:
            print("\nValidating training dataset...")
            for i, batch in enumerate(train_ds.take(1)):
                print(f"Training batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
                
            print("\nValidating validation dataset...")
            for i, batch in enumerate(val_ds.take(1)):
                print(f"Validation batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
        except Exception as e:
            raise ValueError(f"Error validating datasets: {e}")
        
        print("\nStarting training loop...")
        
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
            if patience_counter >= self.config.early_stopping_patience:
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
        plt.savefig(self.plot_dir / f'training_progress_{time.strftime("%Y%m%d_%H%M%S")}.png')
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

# Add MeanTeacherTrainer class definition here

# In train_ssl_tf2n.py

class MeanTeacherTrainer(tf.keras.Model):
    def __init__(self, student_model, teacher_model, ema_decay,
                 # Parameters for Phased EMA
                 teacher_warmup_epochs=0, 
                 initial_teacher_ema_decay=0.95, 
                 sharpening_temperature=0.5,
                 **kwargs): # Keep **kwargs for other tf.keras.Model arguments
        super().__init__(**kwargs) # Pass **kwargs to the parent constructor
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        self.base_ema_decay = ema_decay # This is the target high decay (e.g., 0.999) after warmup
        self.teacher_ema_warmup_epochs = teacher_warmup_epochs # Epochs in MT phase for faster teacher updates
        self.initial_teacher_ema_decay = initial_teacher_ema_decay # EMA decay during these teacher_ema_warmup_epochs
        self.sharpening_temperature = tf.constant(sharpening_temperature, dtype=tf.float32) # Temperature for sharpening logits
        self.teacher_model.trainable = False # Teacher is not trained by optimizer

        self.consistency_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="consistency_weight")
        self.consistency_loss_fn = tf.keras.losses.MeanSquaredError() # Standard MSE for prob difference
        
        # To track the current epoch within the Mean Teacher training phase (updated by a callback)
        self.mt_phase_current_epoch = tf.Variable(-1, dtype=tf.int32, trainable=False, name="mt_phase_epoch_counter")

    def _update_teacher_model(self):
        current_epoch_mt_phase = self.mt_phase_current_epoch # This is a tf.Variable (int32)
        
        # --- Convert Python ints/floats to TF Tensors for comparison if not already ---
        # self.teacher_ema_warmup_epochs is a Python int from __init__
        # self.initial_teacher_ema_decay is a Python float
        # self.base_ema_decay is a Python float
        
        # Condition for using initial_teacher_ema_decay
        # All parts of the condition must result in TF boolean tensors
        cond1 = tf.greater(tf.cast(self.teacher_ema_warmup_epochs, tf.int32), 0) # True if teacher_ema_warmup_epochs > 0
        cond2 = tf.less(current_epoch_mt_phase, tf.cast(self.teacher_ema_warmup_epochs, tf.int32)) # True if current_epoch < teacher_ema_warmup_epochs
        cond3 = tf.not_equal(current_epoch_mt_phase, tf.constant(-1, dtype=tf.int32)) # True if current_epoch has been set

        # Combine conditions using tf.logical_and
        is_in_teacher_warmup_phase = tf.logical_and(cond1, tf.logical_and(cond2, cond3))

        # Use tf.cond to select the effective_ema_decay
        effective_ema_decay = tf.cond(
            is_in_teacher_warmup_phase,
            lambda: tf.cast(self.initial_teacher_ema_decay, tf.float32), # True branch
            lambda: tf.cast(self.base_ema_decay, tf.float32)             # False branch
        )
        
        # Optional: tf.print for debugging the EMA decay being used (conditionally)
        # This print will execute based on the TF graph's control flow.
        # def print_warmup():
        #     tf.print(f"EMA Update: MT Epoch {current_epoch_mt_phase + 1}, using WARMUP EMA: {effective_ema_decay}", output_stream=sys.stderr)
        #     return tf.constant(True) # Dummy return for tf.cond
        # def print_base():
        #     tf.print(f"EMA Update: MT Epoch {current_epoch_mt_phase + 1}, using BASE EMA: {effective_ema_decay}", output_stream=sys.stderr)
        #     return tf.constant(True)
        # _ = tf.cond(is_in_teacher_warmup_phase, print_warmup, print_base)


        for student_var, teacher_var in zip(self.student_model.trainable_variables, self.teacher_model.trainable_variables):
            # Ensure dtypes match if necessary, though assign usually handles it
            # student_val_casted = tf.cast(student_var, teacher_var.dtype)
            # teacher_var.assign(effective_ema_decay * teacher_var + (1.0 - effective_ema_decay) * student_val_casted)
            teacher_var.assign(effective_ema_decay * teacher_var + (1.0 - effective_ema_decay) * student_var)

    def _calculate_dice(self, y_true, y_pred_logits, smooth=1e-6):
        # y_true shape: (batch, H, W, 1)
        # y_pred_logits shape: (batch, H, W, 1)
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_probs = tf.nn.sigmoid(y_pred_logits) # Convert logits to probabilities
        y_pred_f_binary = tf.cast(y_pred_probs > 0.5, tf.float32) # Binarize predictions

        # Calculate Dice per sample in the batch then average
        intersection = tf.reduce_sum(y_true_f * y_pred_f_binary, axis=[1, 2, 3]) # Sum over H, W, C (Channel is 1)
        sum_true = tf.reduce_sum(y_true_f, axis=[1, 2, 3])
        sum_pred = tf.reduce_sum(y_pred_f_binary, axis=[1, 2, 3])
        
        dice_per_sample = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
        
        return tf.reduce_mean(dice_per_sample) # Average Dice across the batch

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        # self.compiled_loss refers to supervised_loss_fn
        # self.compiled_metrics refers to metrics

    @property
    def metrics(self):
        # This ensures Keras tracks metrics correctly.
        # It starts with metrics passed to compile (like student_dice).
        # If you had metrics you only wanted to update in train_step but not display, you'd manage them differently.
        return super().metrics 

    def train_step(self, data):
        labeled_data, unlabeled_data = data
        labeled_images, true_labels = labeled_data
        unlabeled_student_input, unlabeled_teacher_input = unlabeled_data # These are (student_aug, teacher_aug)

        with tf.GradientTape() as tape:
            # Supervised loss on labeled data
            student_labeled_logits = self.student_model(labeled_images, training=True)
            supervised_loss = self.compiled_loss(
                true_labels, 
                student_labeled_logits, 
                regularization_losses=self.student_model.losses # Include model's internal regularization losses
            )

            # Consistency loss on unlabeled data
            student_unlabeled_logits = self.student_model(unlabeled_student_input, training=True)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_teacher_input, training=False) # Teacher not in training mode
            
            sharpened_teacher_logits = teacher_unlabeled_logits / self.sharpening_temperature 
            sharpened_teacher_probs_for_consistency = tf.nn.sigmoid(sharpened_teacher_logits)

            student_unlabeled_probs = tf.nn.sigmoid(student_unlabeled_logits)
            teacher_unlabeled_probs = tf.nn.sigmoid(teacher_unlabeled_logits) # Teacher output also needs sigmoid
            
            consistency_loss_value = self.consistency_loss_fn(teacher_unlabeled_probs, student_unlabeled_probs)
            
            # Total loss calculation
            total_loss = supervised_loss + self.consistency_weight * consistency_loss_value

        # Compute gradients for student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model using EMA (uses self.mt_phase_current_epoch)
        self._update_teacher_model()

        # Update compiled Keras metrics (e.g., student_dice on labeled data)
        self.compiled_metrics.update_state(true_labels, student_labeled_logits)

        # Prepare logs to be returned: these must be TENSOR objects
        logs = {
            'loss': total_loss, 
            'supervised_loss': supervised_loss, 
            'consistency_loss': consistency_loss_value
        }
        # Add results from compiled metrics (e.g., student_dice.result())
        for metric in self.metrics: # self.metrics includes metrics passed to compile()
            logs[metric.name] = metric.result()
        
        return logs

    # In train_ssl_tf2n.py, class MeanTeacherTrainer
    def test_step(self, data):
        val_images, val_labels = data

        # Student model evaluation
        student_val_logits = self.student_model(val_images, training=False)
        val_student_loss = self.compiled_loss(val_labels, student_val_logits, regularization_losses=self.student_model.losses)
        self.compiled_metrics.update_state(val_labels, student_val_logits)

        # Teacher model evaluation
        teacher_val_logits = self.teacher_model(val_images, training=False)
        teacher_dice_score_batch = self._calculate_dice(val_labels, teacher_val_logits)

        # --- ENHANCED DEBUGGING FOR TEACHER PREDICTIONS ---
        # This will run for every validation batch. It might be very verbose.
        # Consider adding a counter or condition to print less frequently if needed.
        # Example: if self.optimizer.iterations % 10 == 0 (if optimizer is accessible and relevant)
        # Or use a tf.Variable as a counter for debug prints.
        
        # For debugging, let's look at the raw components for the current batch
        y_true_b = tf.cast(val_labels, tf.float32)
        teacher_pred_probs_b = tf.nn.sigmoid(teacher_val_logits)
        teacher_pred_binary_b = tf.cast(teacher_pred_probs_b > 0.5, tf.float32)

        # Sums per sample in the batch
        gt_sums_per_sample = tf.reduce_sum(y_true_b, axis=[1, 2, 3]) # Sum over H, W, C
        teacher_pred_sums_per_sample = tf.reduce_sum(teacher_pred_binary_b, axis=[1, 2, 3])

        # Number of samples in batch where GT is all zero
        num_gt_empty = tf.reduce_sum(tf.cast(tf.equal(gt_sums_per_sample, 0.0), tf.int32))
        # Number of samples in batch where Teacher prediction is all zero
        num_teacher_pred_empty = tf.reduce_sum(tf.cast(tf.equal(teacher_pred_sums_per_sample, 0.0), tf.int32))

        # Number of samples where BOTH GT and Teacher pred are empty
        both_empty_mask = tf.logical_and(tf.equal(gt_sums_per_sample, 0.0), tf.equal(teacher_pred_sums_per_sample, 0.0))
        num_both_empty = tf.reduce_sum(tf.cast(both_empty_mask, tf.int32))

        # Average teacher probability for this batch (can indicate if it's always predicting low values)
        avg_teacher_prob_batch = tf.reduce_mean(teacher_pred_probs_b)

        tf.print(
            "DEBUG ValBatch Teacher - BatchDice:", teacher_dice_score_batch,
            "| GT empty samples:", num_gt_empty, "/", tf.shape(val_labels)[0],
            "| TeacherPred empty samples:", num_teacher_pred_empty, "/", tf.shape(val_labels)[0],
            "| BothEmpty samples:", num_both_empty, "/", tf.shape(val_labels)[0],
            "| AvgTeacherProb:", avg_teacher_prob_batch,
            # Optional: print sums for first sample for more detail
            # "| S0_GT Sum:", gt_sums_per_sample[0] if tf.shape(val_labels)[0] > 0 else -1,
            # "| S0_TeacherPred Sum:", teacher_pred_sums_per_sample[0] if tf.shape(val_labels)[0] > 0 else -1,
            output_stream=sys.stderr,
            summarize=-1 # Print all elements of tensors if they are small
        )
        # --- END ENHANCED DEBUGGING ---
        
        final_logs = {'loss': val_student_loss, 'teacher_dice': teacher_dice_score_batch}
        for metric in self.metrics: 
            final_logs[metric.name] = metric.result() # Keras handles val_ prefix
            
        return final_logs

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
# In train_ssl_tf2n.py, modify/replace the MixMatchTrainer:

class MixMatchTrainer: # Ensure this is the one being used, not an older version
    def __init__(self, config):
        print("Initializing MixMatch Trainer (Revised)...")
        self.config = config
        
        # Ensure proper batch sizes
        self.config.batch_size = min(8, self.config.batch_size) # Let user control via args
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # MixMatch hyperparameters from config or defaults
        self.T = getattr(config, 'mixmatch_T', 0.5)  # Temperature sharpening
        self.K = getattr(config, 'mixmatch_K', 2)    # Number of augmentations for pseudo-label
        self.alpha = getattr(config, 'mixmatch_alpha', 0.75)  # Beta distribution parameter for MixUp
        self.max_consistency_weight = getattr(config, 'mixmatch_consistency_max', 10.0)
        self.rampup_length_steps = getattr(config, 'mixmatch_rampup_steps', 1000) # Rampup in steps

        # EMA parameters for teacher model
        self.initial_ema_decay = getattr(config, 'initial_ema_decay', 0.95)
        self.final_ema_decay = getattr(config, 'ema_decay', 0.999) # Final target EMA decay
        self.ema_warmup_steps = getattr(config, 'ema_warmup_steps', 1000) # Steps for EMA decay to ramp up
        
        # Create output directories
        self.output_dir_base = Path(getattr(config, 'output_dir', 'mixmatch_results_default')) # Get from config
        self.experiment_name = getattr(config, 'experiment_name', f"MixMatch_Exp_{time.strftime('%Y%m%d_%H%M%S')}")
        self.output_dir = self.output_dir_base / self.experiment_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        print("Creating student and teacher models...")
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config) # EMA of student
        
        dummy_input_shape = (1, config.img_size_x, config.img_size_y, config.num_channels)
        self.student_model.build(input_shape=dummy_input_shape)
        self.teacher_model.build(input_shape=dummy_input_shape)
        self.teacher_model.set_weights(self.student_model.get_weights())
        self.teacher_model.trainable = False # Teacher is not trained by optimizer
        
        # Setup optimizer
        initial_lr = getattr(config, 'learning_rate', 1e-4) # Get from config
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr, 
            first_decay_steps=getattr(config, 'lr_decay_steps', 500), # steps, not epochs
            t_mul=2.0, m_mul=0.95, alpha=0.1 # alpha is min_lr_factor
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # Loss functions
        self.supervised_loss_fn = CombinedLoss(config) # Dice + Focal
        self.consistency_loss_fn = tf.keras.losses.MeanSquaredError() # L2 loss for consistency
        
        # History tracking
        self.history = {
            'epoch': [], 'total_loss': [], 'supervised_loss': [], 'consistency_loss': [],
            'val_student_dice': [], 'val_teacher_dice': [], 'learning_rate': []
        }
        print("MixMatch trainer initialization complete!")

    def _sharpen_sigmoid_probs(self, p, T):
        """Sharpen sigmoid probabilities. p is a probability (0 to 1)."""
        if T == 0: # Avoid division by zero, effectively becomes argmax
            return tf.cast(p > 0.5, p.dtype)
        p_sharp = p**(1/T)
        return p_sharp / (p_sharp + (1-p)**(1/T) + 1e-7) # Add epsilon for stability

    def mixup(self, x1, x2, y1, y2, alpha):
        """Performs mixup augmentation. y1, y2 are probabilities."""
        batch_size = tf.shape(x1)[0]
        
        # Beta distribution for mixing coefficient
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lambda_ = beta_dist.sample(batch_size)
        lambda_ = tf.maximum(lambda_, 1.0 - lambda_) # Ensure lambda >= 0.5
        
        lambda_x = tf.reshape(lambda_, [batch_size, 1, 1, 1])
        mixed_x = lambda_x * x1 + (1.0 - lambda_x) * x2
        
        # For labels (probabilities), reshape lambda appropriately
        # y1, y2 have shape [batch, H, W, 1]
        lambda_y = tf.reshape(lambda_, [batch_size, 1, 1, 1]) # Ensure it broadcasts correctly
        mixed_y = lambda_y * y1 + (1.0 - lambda_y) * y2
        
        return mixed_x, mixed_y

    def _update_teacher_ema(self):
        current_step = tf.cast(self.optimizer.iterations, tf.float32)
        
        # Ramp up EMA decay factor
        ramp_ratio = tf.minimum(1.0, current_step / tf.cast(self.ema_warmup_steps, tf.float32))
        current_decay = self.initial_ema_decay + (self.final_ema_decay - self.initial_ema_decay) * ramp_ratio
        
        for student_var, teacher_var in zip(
                self.student_model.trainable_variables,
                self.teacher_model.trainable_variables):
            teacher_var.assign(current_decay * teacher_var + (1.0 - current_decay) * student_var)

    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch_single_view):
        images_l, labels_l = labeled_batch 
        images_u_raw_batch = unlabeled_batch_single_view[0] # This is a BATCH of raw unlabeled images [B, H, W, C]

        # --- Generate Pseudo-Labels for Unlabeled Data ---
        list_of_teacher_probs_on_aug_u = []
        for _ in range(self.K):
            # --- Apply weak augmentation to EACH image in the images_u_raw_batch ---
            # We need to map the augmentation function over the batch dimension.
            current_k_weak_aug_batch = tf.map_fn(
                lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='weak'),
                images_u_raw_batch # Pass the entire batch here
            )
            # current_k_weak_aug_batch will also be [B, H, W, C]
            
            logits_u_k_teacher = self.teacher_model(current_k_weak_aug_batch, training=False)
            probs_u_k_teacher = tf.nn.sigmoid(logits_u_k_teacher)
            list_of_teacher_probs_on_aug_u.append(probs_u_k_teacher)
        
        avg_teacher_probs_u = tf.reduce_mean(tf.stack(list_of_teacher_probs_on_aug_u, axis=0), axis=0)
        pseudo_labels_target_probs = self._sharpen_sigmoid_probs(avg_teacher_probs_u, self.T)

        # --- Augmentations for Student Model Input (applied to BATCHES) ---
        # images_l is already a batch [B_l, H, W, C]
        # images_u_raw_batch is also a batch [B_u, H, W, C]

        images_l_strong_aug_batch = tf.map_fn(
            lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='strong'),
            images_l # Pass the labeled batch
        )
        images_u_strong_aug_batch = tf.map_fn(
            lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='strong'),
            images_u_raw_batch # Pass the unlabeled batch
        )

        # --- MixUp ---
        # MixUp for supervised part:
        bs_l = tf.shape(images_l_strong_aug_batch)[0]
        shuffle_l_indices = tf.random.shuffle(tf.range(bs_l))
        images_l_mixed, labels_l_mixed = self.mixup(
            images_l_strong_aug_batch, tf.gather(images_l_strong_aug_batch, shuffle_l_indices),
            labels_l, tf.gather(labels_l, shuffle_l_indices), 
            self.alpha
        )

        # MixUp for unsupervised part:
        bs_u = tf.shape(images_u_strong_aug_batch)[0]
        shuffle_u_indices = tf.random.shuffle(tf.range(bs_u))
        images_u_mixed, pseudo_labels_target_probs_mixed = self.mixup(
            images_u_strong_aug_batch, tf.gather(images_u_strong_aug_batch, shuffle_u_indices),
            pseudo_labels_target_probs, tf.gather(pseudo_labels_target_probs, shuffle_u_indices),
            self.alpha
        )
            
        with tf.GradientTape() as tape:
            logits_l_student_on_mixed = self.student_model(images_l_mixed, training=True)
            logits_u_student_on_mixed = self.student_model(images_u_mixed, training=True)

            supervised_loss = self.supervised_loss_fn(labels_l_mixed, logits_l_student_on_mixed)
            
            student_probs_u_on_mixed = tf.nn.sigmoid(logits_u_student_on_mixed)
            consistency_loss = self.consistency_loss_fn(pseudo_labels_target_probs_mixed, student_probs_u_on_mixed)
            
            current_step_float = tf.cast(self.optimizer.iterations, tf.float32)
            consistency_weight_val = self.max_consistency_weight * tf.minimum(1.0, current_step_float / tf.cast(self.rampup_length_steps, tf.float32))
            
            total_loss = supervised_loss + consistency_weight_val * consistency_loss

        trainable_vars = self.student_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else tf.zeros_like(v) for g, v in zip(gradients, trainable_vars)]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self._update_teacher_ema()

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'consistency_weight': consistency_weight_val
        }
    
    def _compute_dice_metric(self, y_true, y_pred_logits):
        """Computes Dice score for validation. y_pred_logits are model outputs."""
        y_true = tf.cast(y_true, tf.float32)
        # Sigmoid and threshold for Dice calculation
        y_pred_probs = tf.nn.sigmoid(y_pred_logits)
        y_pred_binary = tf.cast(y_pred_probs > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred_binary, axis=[1, 2, 3]) # Sum over H, W, C
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_binary, axis=[1, 2, 3])
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        student_dice_scores = []
        teacher_dice_scores = []
        
        for images, labels in val_dataset:
            student_logits = self.student_model(images, training=False)
            teacher_logits = self.teacher_model(images, training=False) # Teacher is EMA
            
            student_dice = self._compute_dice_metric(labels, student_logits)
            teacher_dice = self._compute_dice_metric(labels, teacher_logits)
            
            if not tf.math.is_nan(student_dice): student_dice_scores.append(student_dice.numpy())
            if not tf.math.is_nan(teacher_dice): teacher_dice_scores.append(teacher_dice.numpy())
        
        avg_student_dice = np.mean(student_dice_scores) if student_dice_scores else 0.0
        avg_teacher_dice = np.mean(teacher_dice_scores) if teacher_dice_scores else 0.0
        return avg_student_dice, avg_teacher_dice

    def train(self, data_paths, num_epochs, steps_per_epoch=None): # Add steps_per_epoch
        print("\nStarting MixMatch training...")
        
        train_ds_l = self.data_pipeline.build_labeled_dataset(
            data_paths['labeled']['images'], data_paths['labeled']['labels'],
            self.config.batch_size, is_training=True
        ).repeat().prefetch(tf.data.AUTOTUNE)

        train_ds_u = self.data_pipeline.build_unlabeled_dataset_for_mixmatch(
            data_paths['unlabeled']['images'], self.config.batch_size, is_training=True
        ).repeat().prefetch(tf.data.AUTOTUNE) # Repeat unlabeled data

        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'], data_paths['validation']['labels'],
            self.config.batch_size
        ).prefetch(tf.data.AUTOTUNE)
        
        if steps_per_epoch is None:
            # Try to determine from labeled dataset
            cardinality_l = tf.data.experimental.cardinality(train_ds_l).numpy()
            if cardinality_l > 0 and cardinality_l != tf.data.experimental.UNKNOWN_CARDINALITY:
                steps_per_epoch = cardinality_l
                print(f"Determined steps_per_epoch from labeled dataset: {steps_per_epoch}")
            else:
                # Fallback if cardinality is unknown (e.g., due to interleave)
                # Estimate based on number of labeled image files and typical slices
                # num_labeled_files = len(data_paths['labeled']['images'])
                # avg_slices_per_file = 30 # Rough estimate
                # steps_per_epoch = (num_labeled_files * avg_slices_per_file) // self.config.batch_size
                steps_per_epoch = 100 # Default if still unknown
                print(f"Warning: Labeled dataset cardinality unknown/zero. Using default steps_per_epoch: {steps_per_epoch}")
        else:
            print(f"Using provided steps_per_epoch: {steps_per_epoch}")


        train_iter_l = iter(train_ds_l)
        train_iter_u = iter(train_ds_u)
        
        best_student_dice = 0.0
        patience_counter = 0
        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 20)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            epoch_total_losses = []
            epoch_sup_losses = []
            epoch_cons_losses = []
            
            # Reset labeled iterator if it runs out within an epoch (if steps_per_epoch > #labeled_batches)
            # This ensures we always have labeled data for each step.
            # Note: If steps_per_epoch is less than actual labeled batches, some labeled data won't be seen.
            
            # Progress bar for steps within an epoch
            progress_bar = tf.keras.utils.Progbar(steps_per_epoch,unit_name="step") # Corrected import for Progbar
            
            for step in range(steps_per_epoch):
                try:
                    labeled_batch = next(train_iter_l)
                except tf.errors.OutOfRangeError: # Or StopIteration
                    train_iter_l = iter(train_ds_l) # Reinitialize iterator
                    labeled_batch = next(train_iter_l)
                
                unlabeled_batch_single_view = next(train_iter_u) # Unlabeled is repeated, so no OutOfRange expected
                
                metrics = self.train_step(labeled_batch, unlabeled_batch_single_view)
                
                epoch_total_losses.append(metrics['total_loss'].numpy())
                epoch_sup_losses.append(metrics['supervised_loss'].numpy())
                epoch_cons_losses.append(metrics['consistency_loss'].numpy())
                progress_bar.update(step + 1, values=[("total_loss", metrics['total_loss']), ("sup_loss", metrics['supervised_loss']), ("cons_loss", metrics['consistency_loss'])])

            # --- End of Epoch ---
            avg_total_loss = np.mean(epoch_total_losses)
            avg_sup_loss = np.mean(epoch_sup_losses)
            avg_cons_loss = np.mean(epoch_cons_losses)
            current_lr = self.lr_schedule(self.optimizer.iterations).numpy() # Get current LR

            # Validation
            val_student_dice, val_teacher_dice = self.validate(val_ds)
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['total_loss'].append(avg_total_loss)
            self.history['supervised_loss'].append(avg_sup_loss)
            self.history['consistency_loss'].append(avg_cons_loss)
            self.history['val_student_dice'].append(val_student_dice)
            self.history['val_teacher_dice'].append(val_teacher_dice)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time_taken = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time_taken:.2f}s - LR: {current_lr:.3e}")
            print(f"  Losses: Total={avg_total_loss:.4f}, Sup={avg_sup_loss:.4f}, Cons={avg_cons_loss:.4f}")
            print(f"  Val Dice: Student={val_student_dice:.4f}, Teacher={val_teacher_dice:.4f}")

            # Save best model based on student validation dice
            if val_student_dice > best_student_dice:
                best_student_dice = val_student_dice
                self.save_checkpoint('best_mixmatch_model')
                print(f"  ✓ New best student validation Dice: {best_student_dice:.4f}. Checkpoint saved.")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1: # Plot every 5 epochs or last epoch
                self.plot_progress(epoch + 1)

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in student validation Dice for {early_stopping_patience} epochs.")
                break
        
        print(f"\nTraining completed! Best student validation Dice: {best_student_dice:.4f}")
        # Save final model
        self.save_checkpoint('final_mixmatch_model')
        self.plot_progress(num_epochs, final_plot=True) # Final plot
        
        # Save history to CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'mixmatch_training_history.csv', index=False)
        print(f"Training history saved to {self.output_dir / 'mixmatch_training_history.csv'}")
        
        return self.history

    def save_checkpoint(self, name_prefix):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        student_model_path = self.checkpoint_dir / f'{name_prefix}_student_{timestamp}.weights.h5'
        teacher_model_path = self.checkpoint_dir / f'{name_prefix}_teacher_{timestamp}.weights.h5'
        
        self.student_model.save_weights(str(student_model_path))
        self.teacher_model.save_weights(str(teacher_model_path)) # Save teacher EMA weights
        print(f"Saved checkpoint: Student to {student_model_path}, Teacher to {teacher_model_path}")

    def plot_progress(self, epoch_num, final_plot=False):
        if not self.history['epoch']: return # Nothing to plot

        plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'MixMatch Training Progress - Epoch {epoch_num} - {self.experiment_name}', fontsize=16)

        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['total_loss'], label='Total Loss', color='red', marker='o', linestyle='-')
        ax.plot(self.history['epoch'], self.history['supervised_loss'], label='Supervised Loss', color='blue', marker='x', linestyle='--')
        ax.plot(self.history['epoch'], self.history['consistency_loss'], label='Consistency Loss', color='green', marker='s', linestyle=':')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['val_student_dice'], label='Student Val Dice', color='purple', marker='o')
        ax.plot(self.history['epoch'], self.history['val_teacher_dice'], label='Teacher Val Dice', color='orange', marker='x', linestyle='--')
        ax.set_title('Validation Dice Scores')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.set_ylim(0, 1.05) # Dice score range
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['learning_rate'], label='Learning Rate', color='teal', marker='.')
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR Value')
        ax.set_yscale('log') # Often useful for LR
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Placeholder for consistency weight if you decide to log it directly
        # Or remove this subplot if not logging consistency_weight_val from train_step's output dict
        if 'consistency_weight' in self.history and self.history['consistency_weight']:
             ax = axes[1, 1]
             ax.plot(self.history['epoch'], self.history['consistency_weight'], label='Consistency Weight ($\lambda_u$)', color='brown', marker='^')
             ax.set_title('Consistency Weight')
             ax.set_xlabel('Epoch')
             ax.set_ylabel('Weight Value')
             ax.legend()
             ax.grid(True, linestyle='--', alpha=0.7)
        else:
            fig.delaxes(axes[1,1]) # Remove empty subplot


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
        
        plot_filename = 'final_training_summary.png' if final_plot else f'training_progress_epoch_{epoch_num}.png'
        plt.savefig(self.plot_dir / plot_filename, dpi=150) # Slightly lower DPI for faster saving during training
        plt.close(fig)
        print(f"Saved progress plot: {self.plot_dir / plot_filename}")