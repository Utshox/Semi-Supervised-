#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import random
import time
import numpy as np
import tensorflow as tf

# Ensure the script can find other modules in the same directory or specified paths
# Assuming 'config.py', 'data_loader_tf2.py', 'models_tf2.py', 'train_ssl_tf2n.py' are accessible
# If they are in a specific project directory, that directory should be in PYTHONPATH
# or added to sys.path here. For this example, let's assume they are in the same dir or
# a directory structure that Python's import system can resolve.
# Example: sys.path.append(str(Path(__file__).resolve().parent))

from config import ExperimentConfig, StableSSLConfig # Assuming StableSSLConfig can be a base
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg # Assuming PancreasSeg is your U-Net model
from train_ssl_tf2n import MeanTeacherTrainer # The new MeanTeacherTrainer

# --- Define Custom Loss and Metric ---
class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6, name='dice_bce_loss'):
        super().__init__(name=name)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, y_true, y_pred_logits):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred_logits, [-1]), tf.float32)

        # Dice Loss
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)

        # BCE Loss
        bce_loss = tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f)
        bce_loss = tf.reduce_mean(bce_loss)

        return self.weight_dice * dice_loss + self.weight_bce * bce_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_bce': self.weight_bce,
            'weight_dice': self.weight_dice,
            'smooth': self.smooth
        })
        return config

# --- Define Custom Callback for Consistency Weight ---
class ConsistencyWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, consistency_weight_var, max_weight, rampup_epochs):
        super().__init__()
        self.consistency_weight_var = consistency_weight_var
        self.max_weight = max_weight
        self.rampup_epochs = rampup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.rampup_epochs == 0: # Handle case for immediate max weight
             new_weight = self.max_weight
        elif epoch < self.rampup_epochs:
            new_weight = self.max_weight * (float(epoch + 1) / float(self.rampup_epochs))
        else:
            new_weight = self.max_weight
        self.consistency_weight_var.assign(new_weight)
        if epoch % 10 == 0 or epoch == 0 : # Print less frequently
             print(f"Epoch {epoch+1}: Consistency weight set to {new_weight:.4f}")

# --- Data Preparation Function ---
def prepare_data_paths(data_dir_str: str, num_labeled_patients: int, num_validation_patients: int, seed: int = 42):
    data_dir = Path(data_dir_str)
    # Assuming patient data is in subdirectories like 'pancreas_001', 'pancreas_002', etc.
    # And each contains 'image.npy' and 'mask.npy'
    all_patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pancreas_")])
    
    if not all_patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {data_dir_str} matching 'pancreas_*'.")

    random.seed(seed)
    random.shuffle(all_patient_dirs)

    if len(all_patient_dirs) < num_labeled_patients + num_validation_patients:
        raise ValueError(
            f"Not enough patient data ({len(all_patient_dirs)}) for the specified splits: "
            f"{num_labeled_patients} labeled, {num_validation_patients} validation."
        )

    validation_patient_dirs = all_patient_dirs[:num_validation_patients]
    remaining_patient_dirs = all_patient_dirs[num_validation_patients:]
    
    labeled_patient_dirs = remaining_patient_dirs[:num_labeled_patients]
    unlabeled_patient_dirs = remaining_patient_dirs[num_labeled_patients:] # All others are unlabeled

    print(f"Total patient dirs: {len(all_patient_dirs)}")
    print(f"Labeled patient dirs: {len(labeled_patient_dirs)}")
    print(f"Unlabeled patient dirs: {len(unlabeled_patient_dirs)}")
    print(f"Validation patient dirs: {len(validation_patient_dirs)}")

    def get_file_paths_from_patient_dirs(patient_dirs_list, is_labeled=True):
        image_paths = []
        label_paths = [] if is_labeled else None
        
        for p_dir in patient_dirs_list:
            img_file = p_dir / "image.npy"
            mask_file = p_dir / "mask.npy"
            
            if img_file.exists() and (not is_labeled or mask_file.exists()):
                image_paths.append(str(img_file))
                if is_labeled:
                    label_paths.append(str(mask_file))
            else:
                print(f"Warning: Missing image.npy or mask.npy (if labeled) in {p_dir}", file=sys.stderr)
        
        if is_labeled:
            return image_paths, label_paths
        return image_paths

    labeled_images, labeled_labels = get_file_paths_from_patient_dirs(labeled_patient_dirs, is_labeled=True)
    unlabeled_images = get_file_paths_from_patient_dirs(unlabeled_patient_dirs, is_labeled=False)
    validation_images, validation_labels = get_file_paths_from_patient_dirs(validation_patient_dirs, is_labeled=True)
    
    if not labeled_images: print("Warning: No labeled image paths were found.", file=sys.stderr)
    if not unlabeled_images: print("Warning: No unlabeled image paths were found.", file=sys.stderr)
    if not validation_images: print("Warning: No validation image paths were found.", file=sys.stderr)

    return {
        'labeled': {'images': labeled_images, 'labels': labeled_labels},
        'unlabeled': {'images': unlabeled_images},
        'validation': {'images': validation_images, 'labels': validation_labels}
    }

# --- Main Training Function ---
def main(args):
    # 1. Setup Configuration
    exp_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='semi-supervised-mean-teacher',
        results_dir=Path(args.output_dir) / args.experiment_name
    )
    # Ensure results directory exists
    exp_config.results_dir.mkdir(parents=True, exist_ok=True)

    # Model/Data Config (using StableSSLConfig as a base)
    # TODO: Consider creating a MeanTeacherConfig if more specific params are needed
    model_data_config = StableSSLConfig(
        img_size_x=args.img_size,
        img_size_y=args.img_size,
        num_channels=1, # Assuming grayscale medical images
        num_classes=1,  # For binary segmentation (pancreas vs background)
        batch_size=args.batch_size 
    )
    # Add Mean Teacher specific params if not in StableSSLConfig
    # For example:
    # model_data_config.ema_decay = args.ema_decay 
    # model_data_config.consistency_weight_max = args.consistency_max
    # model_data_config.consistency_rampup_epochs = args.consistency_rampup

    # 2. GPU Setup (Optional, TensorFlow usually handles this)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s) with memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}", file=sys.stderr)
    else:
        print("No GPU detected by TensorFlow. Running on CPU.", file=sys.stderr)


    # 3. Prepare Data Paths
    print("Preparing data paths...")
    data_paths = prepare_data_paths(
        args.data_dir,
        num_labeled_patients=args.num_labeled,
        num_validation_patients=args.num_validation,
        seed=args.seed
    )

    # 4. Initialize DataPipeline
    print("Initializing DataPipeline...")
    data_pipeline = DataPipeline(config=model_data_config)

    # 5. Create Datasets
    print("Building datasets...")
    labeled_train_dataset = data_pipeline.build_labeled_dataset(
        image_paths=data_paths['labeled']['images'],
        label_paths=data_paths['labeled']['labels'],
        batch_size=args.batch_size,
        is_training=True
    )
    unlabeled_train_dataset = data_pipeline.build_unlabeled_dataset_for_mean_teacher(
        image_paths=data_paths['unlabeled']['images'],
        batch_size=args.batch_size,
        is_training=True
    )
    validation_dataset = data_pipeline.build_validation_dataset(
        image_paths=data_paths['validation']['images'],
        label_paths=data_paths['validation']['labels'],
        batch_size=args.batch_size
    )

    if labeled_train_dataset is None or unlabeled_train_dataset is None or validation_dataset is None:
        print("Error: One or more datasets could not be built. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Combine labeled and unlabeled datasets for training
    # Ensure they have a compatible structure for zipping and that unlabeled_train_dataset
    # yields pairs for student/teacher views as expected by MeanTeacherTrainer
    train_dataset = tf.data.Dataset.zip((labeled_train_dataset, unlabeled_train_dataset))
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    # 6. Instantiate Student and Teacher Models
    print("Instantiating student and teacher models...")
    student_model = PancreasSeg(model_data_config) # Pass model_data_config positionally
    teacher_model = PancreasSeg(model_data_config) # Pass model_data_config positionally

    # Build models by calling them with dummy data (important for weight initialization before trainer compile)
    dummy_input_shape = (1, args.img_size, args.img_size, model_data_config.num_channels)
    dummy_input = tf.zeros(dummy_input_shape)
    _ = student_model(dummy_input)
    _ = teacher_model(dummy_input)
    print(f"Student model built: {student_model.built}, Name: {student_model.name}")
    print(f"Teacher model built: {teacher_model.built}, Name: {teacher_model.name}")


    # 7. Instantiate MeanTeacherTrainer
    print("Instantiating MeanTeacherTrainer...")
    mean_teacher_trainer = MeanTeacherTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        # config=model_data_config, # This line was problematic for MeanTeacherTrainer's __init__ if it didn't expect 'config'
        ema_decay=args.ema_decay
        # Remove consistency_weight_max and consistency_rampup_epochs if not used by MeanTeacherTrainer's __init__
        # These are now handled by the ConsistencyWeightScheduler callback and direct access to trainer.consistency_weight
    )

    # 8. Compile the Trainer
    print("Compiling MeanTeacherTrainer...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    supervised_loss = DiceBCELoss(weight_bce=0.5, weight_dice=0.5) # Example weights
    
    # Custom metrics for the Mean Teacher model
    # Define a custom metric for dice score calculation
    class DiceCoefficient(tf.keras.metrics.Metric):
        def __init__(self, name='dice_coefficient', **kwargs):
            super().__init__(name=name, **kwargs)
            self.total = self.add_weight(name='total', initializer='zeros')
            self.count = self.add_weight(name='count', initializer='zeros')
            self.smooth = 1e-6

        def update_state(self, y_true, y_pred_logits, sample_weight=None):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            # Apply sigmoid to logits
            y_pred = tf.nn.sigmoid(y_pred_logits)
            y_pred = tf.cast(tf.reshape(y_pred > 0.5, [-1]), tf.float32)
            
            intersection = tf.reduce_sum(y_true * y_pred)
            dice = (2. * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)
            
            self.total.assign_add(dice)
            self.count.assign_add(1.0)
            
        def result(self):
            return self.total / self.count
            
        def reset_state(self):
            self.total.assign(0.)
            self.count.assign(0.)
    
    # Create metric instances
    student_dice_metric = DiceCoefficient(name='student_dice')
    
    mean_teacher_trainer.compile(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss,
        metrics=[student_dice_metric] # Use the metric object instead of string name
    )
    print("MeanTeacherTrainer compiled.")

    # 9. Define Callbacks
    print("Setting up callbacks...")
    callbacks = []
    
    # ModelCheckpoint: Save the best student model based on validation student dice
    checkpoint_dir = exp_config.results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = checkpoint_dir / "best_student_model_epoch_{epoch:02d}_val_student_dice_{val_student_dice:.4f}.weights.h5" # Use .weights.h5 for save_weights_only
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_filepath),
        save_weights_only=True, # Save only weights
        monitor='val_student_dice', # Monitor student's dice score on validation set
        mode='max',          # Maximize Dice score
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint_callback)

    # TensorBoard
    log_dir = exp_config.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    callbacks.append(tensorboard_callback)

    # CSVLogger
    csv_log_path = exp_config.results_dir / "training_log.csv"
    csv_logger_callback = tf.keras.callbacks.CSVLogger(str(csv_log_path))
    callbacks.append(csv_logger_callback)

    # EarlyStopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_student_dice',
        patience=args.early_stopping_patience,
        mode='max',
        verbose=1,
        restore_best_weights=True # Restore best weights found during training
    )
    callbacks.append(early_stopping_callback)

    # Consistency Weight Scheduler
    consistency_scheduler_callback = ConsistencyWeightScheduler(
        consistency_weight_var=mean_teacher_trainer.consistency_weight, # Pass the trainer's variable
        max_weight=args.consistency_max,
        rampup_epochs=args.consistency_rampup
    )
    callbacks.append(consistency_scheduler_callback)
    
    # Learning Rate Scheduler (optional, Adam has adaptive LR)
    # Example: ReduceLROnPlateau
    # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_student_dice', factor=0.2, patience=10, mode='max', min_lr=1e-7, verbose=1)
    # callbacks.append(reduce_lr_callback)


    # 10. Start Training
    print(f"Starting training for {args.num_epochs} epochs...")
    start_time = time.time()
    
    history = mean_teacher_trainer.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=args.verbose
    )

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Results, checkpoints, and logs saved in: {exp_config.results_dir}")

    # Optionally, save the final student model explicitly
    final_model_path = exp_config.results_dir / "final_student_model.weights.h5"
    student_model.save_weights(str(final_model_path))
    print(f"Final student model weights saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mean Teacher training for pancreas segmentation.")
    
    # Paths and Experiment Config
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of the preprocessed data (e.g., /path/to/images/preprocessed_v2).")
    parser.add_argument("--output_dir", type=str, default="./experiment_results", help="Directory to save experiment results.")
    parser.add_argument("--experiment_name", type=str, default=f"mean_teacher_{time.strftime('%Y%m%d_%H%M%S')}", help="Name for the experiment.")

    # Data Config
    parser.add_argument("--img_size", type=int, default=256, help="Image size (height and width).")
    parser.add_argument("--num_labeled", type=int, default=30, help="Number of labeled patient volumes for training.")
    parser.add_argument("--num_validation", type=int, default=10, help="Number of patient volumes for validation.") # Adjusted from user's 56, as it might be too large for typical splits

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.") # User mentioned 2, but 4 might be a good default
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Patience for early stopping.")
    
    # Mean Teacher Specific Hyperparameters
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate for teacher model updates.")
    parser.add_argument("--consistency_max", type=float, default=10.0, help="Maximum weight for the consistency loss.") # Typical values are 1.0 to 100.0
    parser.add_argument("--consistency_rampup", type=int, default=30, help="Number of epochs for consistency weight ramp-up.")

    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2], help="Verbosity mode for training (0 = silent, 1 = progress bar, 2 = one line per epoch).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    main(args)
