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

class MeanTeacherTrainer(tf.keras.Model):
    def __init__(self, student_model, teacher_model, ema_decay,  teacher_ema_warmup_epochs=0,  initial_teacher_ema_decay=0.95, **kwargs):
        super().__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.base_ema_decay = ema_decay
        
        self.teacher_ema_warmup_epochs = teacher_ema_warmup_epochs
        self.initial_teacher_ema_decay = initial_teacher_ema_decay
        # Ensure teacher model is not trainable by the optimizer
        self.teacher_model.trainable = False

        self.consistency_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="consistency_weight")
        self.consistency_loss_fn = tf.keras.losses.MeanSquaredError()

        # To track the current epoch within the Mean Teacher training phase
        self.mt_phase_current_epoch = tf.Variable(-1, dtype=tf.int32, trainable=False, name="mt_phase_epoch_counter")

    def _update_teacher_model(self):
        # self.mt_phase_current_epoch should be updated by a callback
        current_epoch = self.mt_phase_current_epoch 
        
        effective_ema_decay = self.base_ema_decay
        if self.teacher_ema_warmup_epochs > 0 and current_epoch < self.teacher_ema_warmup_epochs and current_epoch != -1 : # current_epoch != -1 ensures it's been set
            effective_ema_decay = self.initial_teacher_ema_decay
            # Optional: tf.print for debugging the EMA decay being used
            # tf.print(f"EMA Update: MT Epoch {current_epoch + 1}, using WARMUP EMA decay: {effective_ema_decay}", output_stream=sys.stderr)
        # else:
            # tf.print(f"EMA Update: MT Epoch {current_epoch + 1}, using BASE EMA decay: {effective_ema_decay}", output_stream=sys.stderr)


        for student_var, teacher_var in zip(self.student_model.trainable_variables, self.teacher_model.trainable_variables):
            teacher_var.assign(self.ema_decay * teacher_var + (1.0 - self.ema_decay) * student_var)

# In train_ssl_tf2n.py

class MeanTeacherTrainer(tf.keras.Model):
    def __init__(self, student_model, teacher_model, ema_decay,
                 # Parameters for Phased EMA
                 teacher_warmup_epochs=0, 
                 initial_teacher_ema_decay=0.95, 
                 **kwargs): # Keep **kwargs for other tf.keras.Model arguments
        super().__init__(**kwargs) # Pass **kwargs to the parent constructor
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        self.base_ema_decay = ema_decay # This is the target high decay (e.g., 0.999) after warmup
        self.teacher_ema_warmup_epochs = teacher_warmup_epochs # Epochs in MT phase for faster teacher updates
        self.initial_teacher_ema_decay = initial_teacher_ema_decay # EMA decay during these teacher_ema_warmup_epochs
        
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

    def compile(self, optimizer, supervised_loss_fn, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=supervised_loss_fn, metrics=metrics, **kwargs)
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

    def test_step(self, data):
        val_images, val_labels = data

        # Student model evaluation
        student_val_logits = self.student_model(val_images, training=False)
        # Calculate student's supervised loss on validation data
        val_student_loss = self.compiled_loss(val_labels, student_val_logits, regularization_losses=self.student_model.losses)

        # Update compiled Keras metrics for the student on validation data
        self.compiled_metrics.update_state(val_labels, student_val_logits)

        # Teacher model evaluation
        teacher_val_logits = self.teacher_model(val_images, training=False)
        # Calculate teacher's Dice score manually for logging
        teacher_dice_score = self._calculate_dice(val_labels, teacher_val_logits) 

        # Prepare logs to be returned
        logs = {
            'loss': val_student_loss, # Overall validation loss is student's supervised loss
            'teacher_dice': teacher_dice_score # Log teacher's dice score
        }
        # Add results from compiled metrics (this will include val_student_dice)
        for metric in self.metrics:
            logs[f"val_{metric.name}"] = metric.result() # Keras automatically prefixes 'val_'
                                                         # but for compiled_metrics, result() here gives current state.
                                                         # Let's rely on Keras to name them correctly.
                                                         # The metric.name for DiceCoefficient is 'student_dice'.
                                                         # Keras will log it as 'val_student_dice'.
            # We can just add the compiled metrics directly; Keras handles val_ prefix.
            logs[metric.name] = metric.result() 

        # To be absolutely clear for logging:
        # Keras will take the 'loss' from here and log it as 'val_loss'.
        # It will take 'student_dice' from compiled_metrics.result() and log it as 'val_student_dice'.
        # So, we just need to add 'teacher_dice'.
        final_logs = {'loss': val_student_loss, 'teacher_dice': teacher_dice_score}
        for metric in self.metrics: # This gets the updated state for 'student_dice'
            final_logs[metric.name] = metric.result()
            
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
# class MixMatchTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.data_pipeline = DataPipeline(config)
        
#         # MixMatch parameters
#         self.T = 0.5 
#         self.K = 2  
#         self.alpha = 0.75
#         self.rampup_length = 1000
        
#         # Models
#         print("Initializing models...")
#         self.student_model = PancreasSeg(config)
#         self.teacher_model = PancreasSeg(config)
        
#         # Initialize models with dummy input
#         print("Initializing with dummy input...")
#         dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
#         _ = self.student_model(dummy_input)
#         _ = self.teacher_model(dummy_input)
#         self.teacher_model.set_weights(self.student_model.get_weights())
        
#         # Optimizer and losses
#         print("Setting up optimizer and losses...")
#         self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#             1e-4, 100, t_mul=2.0, m_mul=0.95, alpha=0.2
#         )
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
#         self.supervised_loss_fn = CombinedLoss(config)
        
#         # History tracking
#         self.history = {
#             'train_loss': [],
#             'val_dice': [],
#             'teacher_dice': [],
#             'learning_rate': [],
#             'supervised_loss': [],
#             'consistency_loss': []
#         }
        
#         # Setup directories
#         print("Creating output directories...")
#         self.output_dir = Path('mixmatch_results')
#         self.checkpoint_dir = self.output_dir / 'checkpoints'
#         self.plot_dir = self.output_dir / 'plots'
        
#         self.output_dir.mkdir(exist_ok=True)
#         self.checkpoint_dir.mkdir(exist_ok=True)
#         self.plot_dir.mkdir(exist_ok=True)
        
#         print("MixMatch trainer initialization complete!")

#     def sharpen(self, x, T):
#         x = tf.nn.softmax(x / T, axis=-1)
#         return x

#     def mixup(self, x1, x2, y1, y2, alpha=0.75):
#         beta = tf.random.beta([tf.shape(x1)[0]], alpha, alpha)
#         beta = tf.maximum(beta, 1-beta)
#         beta = tf.reshape(beta, [-1, 1, 1, 1])
#         x = beta * x1 + (1-beta) * x2
#         beta_label = tf.reshape(beta, [-1, 1, 1, 1])
#         y = beta_label * y1 + (1-beta_label) * y2
#         return x, y

#     @tf.function
#     def train_step(self, labeled_batch, unlabeled_batch):
#         images_l, labels_l = labeled_batch
#         images_u = unlabeled_batch
#         batch_size = tf.shape(images_l)[0]

#         with tf.GradientTape() as tape:
#             # Generate pseudo-labels
#             logits_u = []
#             for _ in range(self.K):
#                 logits_u.append(self.teacher_model(images_u, training=False))
#             logits_u = tf.reduce_mean(tf.stack(logits_u, axis=0), axis=0)
#             pseudo_labels = self.sharpen(logits_u, self.T)

#             # MixUp
#             shuffle = tf.random.shuffle(tf.range(batch_size))
#             images_l_mix, labels_l_mix = self.mixup(
#                 images_l, tf.gather(images_l, shuffle),
#                 labels_l, tf.gather(labels_l, shuffle),
#                 self.alpha
#             )
            
#             images_u_mix, pseudo_labels_mix = self.mixup(
#                 images_u, tf.gather(images_u, shuffle),
#                 pseudo_labels, tf.gather(pseudo_labels, shuffle),
#                 self.alpha
#             )

#             # Forward passes
#             logits_l = self.student_model(images_l_mix, training=True)
#             logits_u = self.student_model(images_u_mix, training=True)

#             # Calculate losses
#             supervised_loss = self.supervised_loss_fn(labels_l_mix, logits_l)
#             consistency_loss = tf.reduce_mean(tf.square(
#                 tf.nn.sigmoid(logits_u) - tf.nn.sigmoid(pseudo_labels_mix)
#             ))
            
#             # Ramp up consistency weight
#             current_step = tf.cast(self.optimizer.iterations, tf.float32)
#             consistency_weight = 10.0 * tf.minimum(1.0, current_step / self.rampup_length)
            
#             total_loss = supervised_loss + consistency_weight * consistency_loss

#         # Optimize
#         gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
#         gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
#         self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

#         # Update teacher model
#         self.update_teacher()

#         return {
#             'total_loss': total_loss,
#             'supervised_loss': supervised_loss,
#             'consistency_loss': consistency_loss
#         }

#     def update_teacher(self, rampup_value):
#         """Update teacher model weights using EMA with ramp-up."""
#         # Calculate current ema decay
#         current_ema = self.initial_ema_decay + (self.final_ema_decay - self.initial_ema_decay) * rampup_value
        
#         for student_weights, teacher_weights in zip(
#                 self.student_model.trainable_variables,
#                 self.teacher_model.trainable_variables):
#             teacher_weights.assign(
#                 current_ema * teacher_weights + 
#                 (1 - current_ema) * student_weights
#             )

#     @tf.function
#     def train_step(self, batch, training=True):
#         images, labels = batch
#         images = tf.cast(images, tf.float32)
#         labels = tf.cast(labels, tf.float32)
        
#         # Add Gaussian noise for consistency regularization
#         if training:
#             noised_images = images + tf.random.normal(tf.shape(images), 0, 0.1)
#             noised_images = tf.clip_by_value(noised_images, 0, 1)
#         else:
#             noised_images = images
        
#         with tf.GradientTape() as tape:
#             # Student predictions with noise
#             student_logits = self.student_model(noised_images, training=True)
            
#             # Teacher predictions without noise
#             teacher_logits = self.teacher_model(images, training=False)
            
#             # Calculate supervised loss
#             supervised_loss = self.supervised_loss_fn(labels, student_logits)
            
#             # Calculate consistency loss
#             consistency_loss = self.consistency_loss_fn(
#                 tf.nn.sigmoid(teacher_logits),
#                 tf.nn.sigmoid(student_logits)
#             )
            
#             # Dynamic ramp-up
#             current_step = tf.cast(self.optimizer.iterations, tf.float32)
#             rampup_length = 1000.0  # Number of steps for ramp-up
#             rampup_value = tf.minimum(1.0, current_step / rampup_length)
#             consistency_weight = 20.0 * rampup_value  # Higher max weight
            
#             # Total loss
#             total_loss = supervised_loss + consistency_weight * consistency_loss
        
#         if training:
#             # Gradient clipping
#             gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
#             gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]  # Increased from 1.0
#             self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
            
#             # Update teacher model with current rampup value
#             self.update_teacher(rampup_value)
        
#         return {
#             'total_loss': total_loss,
#             'supervised_loss': supervised_loss,
#             'consistency_loss': consistency_loss
#         }

        
#         for batch in val_dataset:
#             metrics = self.train_step(batch, training=False)
            
#             # Compute Dice scores
#             student_dice = self.compute_dice(batch[1], metrics['student_logits'])
#             teacher_dice = self.compute_dice(batch[1], metrics['teacher_logits'])
            
#             dice_scores.append(student_dice)
#             teacher_dice_scores.append(teacher_dice)
        
#         return np.mean(dice_scores), np.mean(teacher_dice_scores)
#         visualization_path = generate_report_figures(trainer, val_dataset, 'mean_teacher_results')
#         print(f"Generated report figures in: {visualization_path}")
#     def compute_dice(self, y_true, y_pred):
#         """Compute Dice score with sigmoid activation."""
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.nn.sigmoid(y_pred)  # Apply sigmoid to logits
        
#         # Take pancreas channel
#         if len(y_true.shape) > 3:
#             y_true = y_true[..., -1]
#         if len(y_pred.shape) > 3:
#             y_pred = y_pred[..., -1]
            
#         # Threshold predictions
#         y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
#         # Calculate intersection and union
#         intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
#         union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
#         # Calculate Dice
#         dice = (2. * intersection + 1e-6) / (union + 1e-6)
#         return tf.reduce_mean(dice)
#     def generate_report_figures(trainer, val_dataset, save_dir='report_figures'):
#         """Generate comprehensive figures for the report"""
#         save_dir = Path(save_dir)
#         save_dir.mkdir(exist_ok=True)
        
#         # 1. Training History
#         plt.figure(figsize=(15, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(trainer.history['train_loss'], label='Training Loss')
#         plt.title('Training Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.subplot(1, 2, 2)
#         plt.plot(trainer.history['val_dice'], label='Student Dice')
#         plt.plot(trainer.history['teacher_dice'], label='Teacher Dice')
#         plt.title('Validation Dice Scores')
#         plt.xlabel('Epoch')
#         plt.ylabel('Dice Score')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 2. Sample Predictions
#         for batch_idx, (images, labels) in enumerate(val_dataset.take(1)):
#             student_preds = trainer.student_model(images, training=False)
#             teacher_preds = trainer.teacher_model(images, training=False)
            
#             # Plot 4 examples
#             fig, axes = plt.subplots(4, 4, figsize=(20, 20))
#             for i in range(4):
#                 # Original
#                 axes[i, 0].imshow(images[i, ..., 0], cmap='gray')
#                 axes[i, 0].set_title('Input Image')
                
#                 # Ground Truth
#                 axes[i, 1].imshow(labels[i, ..., -1], cmap='RdYlBu')
#                 axes[i, 1].set_title('Ground Truth')
                
#                 # Student Prediction
#                 axes[i, 2].imshow(tf.nn.sigmoid(student_preds[i, ..., -1]), cmap='RdYlBu')
#                 axes[i, 2].set_title(f'Student Pred (Dice={trainer.compute_dice(labels[i:i+1], student_preds[i:i+1]):.3f})')
                
#                 # Teacher Prediction
#                 axes[i, 3].imshow(tf.nn.sigmoid(teacher_preds[i, ..., -1]), cmap='RdYlBu')
#                 axes[i, 3].set_title(f'Teacher Pred (Dice={trainer.compute_dice(labels[i:i+1], teacher_preds[i:i+1]):.3f})')
                
#                 # Turn off axes
#                 for ax in axes[i]:
#                     ax.axis('off')
            
#             plt.tight_layout()
#             plt.savefig(save_dir / f'sample_predictions_batch_{batch_idx}.png', dpi=300, bbox_inches='tight')
#             plt.close()
        
#         # 3. Performance Summary Table
#         performance_data = {
#             'Metric': ['Best Validation Dice', 'Final Teacher Dice', 'Final Training Loss'],
#             'Value': [
#                 max(trainer.history['val_dice']),
#                 trainer.history['teacher_dice'][-1],
#                 trainer.history['train_loss'][-1]
#             ]
#         }
        
#         # Save as CSV
#         pd.DataFrame(performance_data).to_csv(save_dir / 'performance_summary.csv', index=False)
        
#         return save_dir    20
############################################
class MixMatchTrainer:
    def __init__(self, config):
        print("Initializing MixMatch Trainer...")
        self.config = config
        
        # Ensure proper batch sizes
        self.config.batch_size = min(8, self.config.batch_size)  # Smaller batch size for stability
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # MixMatch hyperparameters
        self.T = 0.5  # Temperature sharpening
        self.K = 2    # Number of augmentations
        self.alpha = 0.75  # Beta distribution parameter
        self.rampup_length = 1000
        
        # Create output directories
        self.output_dir = Path('mixmatch_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Initialize models
        print("Creating models...")
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        self.teacher_model.set_weights(self.student_model.get_weights())
        
        # Setup optimizer
        initial_lr = 1e-4
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr, 
            first_decay_steps=100,
            t_mul=2.0,
            m_mul=0.95,
            alpha=0.2
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # Loss functions
        self.supervised_loss_fn = CombinedLoss(config)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': []
        }
        
        print("MixMatch trainer initialization complete!")

    def sharpen(self, x, T):
        """Sharpens the predictions using temperature scaling"""
        x = tf.cast(x, tf.float32)
        temp = tf.cast(T, tf.float32)
        return tf.nn.softmax(x / temp, axis=-1)

    def mixup(self, x1, x2, y1, y2, alpha=0.75):
        """Performs mixup augmentation using uniform distribution"""
        batch_size = tf.shape(x1)[0]
        x2_size = tf.shape(x2)[0]
        
        # Calculate repeat count
        repeat_count = tf.cast(
            tf.math.ceil(tf.cast(batch_size, tf.float32) / tf.cast(x2_size, tf.float32)),
            tf.int32
        )
        
        # Handle unequal batch sizes
        x2 = tf.repeat(x2, repeats=repeat_count, axis=0)
        x2 = x2[:batch_size]
        
        if y2 is not None:
            y2 = tf.repeat(y2, repeats=repeat_count, axis=0)
            y2 = y2[:batch_size]
        
        # Generate mixing coefficient using uniform distribution
        lambda_ = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
        lambda_ = tf.maximum(lambda_, 1 - lambda_)  # Equivalent to beta distribution
        lambda_x = tf.reshape(lambda_, [batch_size, 1, 1, 1])
        
        # Mix the images
        mixed_x = lambda_x * x1 + (1 - lambda_x) * x2
        
        if y1 is not None and y2 is not None:
            lambda_y = tf.reshape(lambda_, [batch_size, 1, 1, 1])
            mixed_y = lambda_y * y1 + (1 - lambda_y) * y2
            return mixed_x, mixed_y
        
        return mixed_x

    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch):
        """Single training step with error handling"""
        try:
            images_l, labels_l = labeled_batch
            images_u = unlabeled_batch
            batch_size = tf.shape(images_l)[0]

            with tf.GradientTape() as tape:
                # Generate pseudo-labels
                logits_u = self.teacher_model(images_u, training=False)
                pseudo_labels = self.sharpen(logits_u, self.T)

                # MixUp on labeled data
                shuffle_l = tf.random.shuffle(tf.range(batch_size))
                mixed_l = self.mixup(
                    images_l, tf.gather(images_l, shuffle_l),
                    labels_l, tf.gather(labels_l, shuffle_l)
                )
                if isinstance(mixed_l, tuple):
                    images_l_mix, labels_l_mix = mixed_l
                else:
                    images_l_mix, labels_l_mix = mixed_l, None

                # MixUp on unlabeled data
                u_batch_size = tf.shape(images_u)[0]
                shuffle_u = tf.random.shuffle(tf.range(u_batch_size))
                mixed_u = self.mixup(
                    images_u, tf.gather(images_u, shuffle_u),
                    pseudo_labels, tf.gather(pseudo_labels, shuffle_u)
                )
                if isinstance(mixed_u, tuple):
                    images_u_mix, pseudo_labels_mix = mixed_u
                else:
                    images_u_mix, pseudo_labels_mix = mixed_u, None

                # Forward passes
                logits_l = self.student_model(images_l_mix, training=True)
                logits_u = self.student_model(images_u_mix, training=True)

                # Calculate losses
                supervised_loss = self.supervised_loss_fn(labels_l_mix, logits_l)
                consistency_loss = tf.reduce_mean(tf.square(
                    tf.nn.sigmoid(logits_u) - tf.nn.sigmoid(pseudo_labels_mix)
                ))
                
                # Ramp up consistency weight
                current_step = tf.cast(self.optimizer.iterations, tf.float32)
                consistency_weight = 10.0 * tf.minimum(1.0, current_step / self.rampup_length)
                
                total_loss = supervised_loss + consistency_weight * consistency_loss

            # Optimize
            gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

            # Update teacher model
            self.update_teacher()

            return {
                'total_loss': total_loss,
                'supervised_loss': supervised_loss,
                'consistency_loss': consistency_loss
            }
        
        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            print(f"Labeled batch shape: {tf.shape(images_l)}")
            print(f"Unlabeled batch shape: {tf.shape(images_u)}")
            raise

    def compute_dice(self, y_true, y_pred):
        """Computes Dice score"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Performs validation"""
        dice_scores = []
        teacher_dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            
            student_logits = self.student_model(images, training=False)
            teacher_logits = self.teacher_model(images, training=False)
            
            student_dice = self.compute_dice(labels, student_logits)
            teacher_dice = self.compute_dice(labels, teacher_logits)
            
            if not tf.math.is_nan(student_dice) and not tf.math.is_nan(teacher_dice):
                dice_scores.append(float(student_dice))
                teacher_dice_scores.append(float(teacher_dice))
        
        if dice_scores and teacher_dice_scores:
            return np.mean(dice_scores), np.mean(teacher_dice_scores)
        return 0.0, 0.0

    def train(self, data_paths):
        """Main training loop with validation"""
        print("\nStarting MixMatch training...")
        
        # Create datasets
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
        
        best_dice = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            epoch_losses = []
            supervised_losses = []
            consistency_losses = []
            
            # Create iterators
            train_iter = iter(train_ds)
            unlabeled_iter = iter(unlabeled_ds)
            
            # Training steps
            for _ in range(len(list(train_ds))):
                try:
                    labeled_batch = next(train_iter)
                    unlabeled_batch = next(unlabeled_iter)
                    
                    metrics = self.train_step(labeled_batch, unlabeled_batch)
                    epoch_losses.append(float(metrics['total_loss']))
                    supervised_losses.append(float(metrics['supervised_loss']))
                    consistency_losses.append(float(metrics['consistency_loss']))
                except StopIteration:
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
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - start_time:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"Teacher Dice: {teacher_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mixmatch_model')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.checkpoint_dir / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        student_path = checkpoint_dir / f'{name}_student'
        teacher_path = checkpoint_dir / f'{name}_teacher'
        
        self.student_model.save_weights(str(student_path))
        self.teacher_model.save_weights(str(teacher_path))
        
        # Save history
        np.save(checkpoint_dir / f'{name}_history.npy', self.history)
        print(f"Saved checkpoint to {checkpoint_dir}")

    def plot_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_dice'], label='Validation Dice')
        plt.plot(self.history['teacher_dice'], label='Teacher Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(str(self.plot_dir / f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()

