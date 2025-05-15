#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from config import ExperimentConfig, StableSSLConfig
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from train_ssl_tf2n import MeanTeacherTrainer

# --- Define Custom Loss and Metric ---
class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6, name='dice_bce_loss'):
        super().__init__(name=name)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, y_true, y_pred_logits):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_logits_f = tf.cast(tf.reshape(y_pred_logits, [-1]), tf.float32)

        y_pred_probs_f_for_dice = tf.nn.sigmoid(y_pred_logits_f)
        intersection = tf.reduce_sum(y_true_f * y_pred_probs_f_for_dice)
        dice_numerator = 2. * intersection + self.smooth
        dice_denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_probs_f_for_dice) + self.smooth
        dice_loss = 1 - (dice_numerator / dice_denominator)

        bce_loss_val = self.bce_fn(y_true_f, y_pred_logits_f)
        
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss_val

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_bce': self.weight_bce,
            'weight_dice': self.weight_dice,
            'smooth': self.smooth
        })
        return config

class ConsistencyWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, consistency_weight_var, max_weight, rampup_epochs):
        super().__init__()
        self.consistency_weight_var = consistency_weight_var
        self.max_weight = max_weight
        self.rampup_epochs = rampup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.rampup_epochs == 0:
             new_weight = self.max_weight
        elif epoch < self.rampup_epochs:
            new_weight = self.max_weight * (float(epoch + 1) / float(self.rampup_epochs))
        else:
            new_weight = self.max_weight
        self.consistency_weight_var.assign(new_weight)
        if epoch % 10 == 0 or epoch == 0 :
             tf.print(f"Epoch {epoch+1} (MT Phase): Consistency weight set to {new_weight:.4f}")

def prepare_data_paths(data_dir_str: str, num_labeled_patients: int, num_validation_patients: int, seed: int = 42):
    data_dir = Path(data_dir_str)
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
    unlabeled_patient_dirs = remaining_patient_dirs[num_labeled_patients:] 

    tf.print(f"Data Split: Total patient dirs: {len(all_patient_dirs)}, "
             f"Labeled: {len(labeled_patient_dirs)}, "
             f"Unlabeled: {len(unlabeled_patient_dirs)}, "
             f"Validation: {len(validation_patient_dirs)}")

    def get_file_paths_from_patient_dirs(patient_dirs_list, is_labeled=True):
        image_paths = []
        label_paths = [] if is_labeled else None
        for p_dir in patient_dirs_list:
            img_file = p_dir / "image.npy"; mask_file = p_dir / "mask.npy"
            if img_file.exists() and (not is_labeled or mask_file.exists()):
                image_paths.append(str(img_file))
                if is_labeled: label_paths.append(str(mask_file))
            else:
                if not img_file.exists(): tf.print(f"Warning: Missing image.npy in {p_dir}", output_stream=sys.stderr)
                if is_labeled and not mask_file.exists(): tf.print(f"Warning: Missing mask.npy in {p_dir}", output_stream=sys.stderr)
        return (image_paths, label_paths) if is_labeled else image_paths

    labeled_images, labeled_labels = get_file_paths_from_patient_dirs(labeled_patient_dirs, is_labeled=True)
    unlabeled_images = get_file_paths_from_patient_dirs(unlabeled_patient_dirs, is_labeled=False)
    validation_images, validation_labels = get_file_paths_from_patient_dirs(validation_patient_dirs, is_labeled=True)
    
    if not labeled_images: tf.print("CRITICAL WARNING: No Labeled image paths found after filtering.", output_stream=sys.stderr)
    if not unlabeled_images and len(unlabeled_patient_dirs) > 0 : tf.print("WARNING: Unlabeled patient dirs were selected, but no valid image.npy files found in them.", output_stream=sys.stderr)
    elif not unlabeled_images: tf.print("INFO: No Unlabeled image paths found (this might be expected).", output_stream=sys.stderr)
    if not validation_images: tf.print("CRITICAL WARNING: No Validation image paths found after filtering.", output_stream=sys.stderr)

    return {
        'labeled': {'images': labeled_images, 'labels': labeled_labels},
        'unlabeled': {'images': unlabeled_images},
        'validation': {'images': validation_images, 'labels': validation_labels}
    }

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_dice_scores = self.add_weight(name='sum_dice_scores', initializer='zeros')
        self.num_samples_processed = self.add_weight(name='num_samples_processed', initializer='zeros')
        self.smooth = 1e-6

    def update_state(self, y_true, y_pred_logits, sample_weight=None):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_probs = tf.nn.sigmoid(y_pred_logits) 
        y_pred_f_binary = tf.cast(y_pred_probs > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f_binary, axis=[1,2,3]) 
        sum_true = tf.reduce_sum(y_true_f, axis=[1,2,3])
        sum_pred = tf.reduce_sum(y_pred_f_binary, axis=[1,2,3])
        dice_per_sample = (2. * intersection + self.smooth) / (sum_true + sum_pred + self.smooth)
        
        current_batch_dice_sum = tf.reduce_sum(dice_per_sample)
        current_batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        self.sum_dice_scores.assign_add(current_batch_dice_sum)
        self.num_samples_processed.assign_add(current_batch_size)
        
    def result(self):
        return tf.cond(tf.equal(self.num_samples_processed, 0.0),
                       lambda: tf.constant(0.0, dtype=tf.float32),
                       lambda: self.sum_dice_scores / self.num_samples_processed)
            
    def reset_state(self):
        self.sum_dice_scores.assign(0.)
        self.num_samples_processed.assign(0.)

    def get_config(self):
        config = super().get_config(); config.update({'smooth': self.smooth}); return config

def plot_training_summary(history_object, csv_log_path: Path, plot_save_dir: Path, plot_title_prefix: str):
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    data_source = None; history_data = {}

    if history_object and hasattr(history_object, 'history') and history_object.history:
        history_data = history_object.history
        data_source = "Keras History Object"
    elif csv_log_path.exists():
        try:
            df = pd.read_csv(csv_log_path)
            if not df.empty:
                history_data = {col: df[col].tolist() for col in df.columns}
                if 'epoch' in df.columns: history_data['epochs_list'] = df['epoch'].tolist()
                else: history_data['epochs_list'] = list(range(1, len(df) + 1))
                data_source = "CSV Log File"
        except Exception as e: tf.print(f"Error reading CSV log for plotting: {e}", output_stream=sys.stderr)
    
    if not history_data: tf.print("No data found for plotting.", output_stream=sys.stderr); return
    tf.print(f"Generating plots from: {data_source}", output_stream=sys.stderr)

    epochs = history_data.get('epochs_list', None)
    if epochs is None:
        for key in ['loss', 'student_dice', 'val_loss', 'val_student_dice', 'pretrain_student_dice', 'val_pretrain_student_dice']:
            if key in history_data and history_data[key]:
                epochs = list(range(1, len(history_data[key]) + 1)); break
    if epochs is None: tf.print("Could not determine epochs for plotting.", output_stream=sys.stderr); return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1, ax2 = axes[0], axes[1]

    # Losses
    if 'loss' in history_data: ax1.plot(epochs, history_data['loss'], label='Total Train Loss', c='blue')
    if 'supervised_loss' in history_data: ax1.plot(epochs, history_data['supervised_loss'], label='Sup. Loss (Train)', c='orange', ls='--')
    if 'consistency_loss' in history_data: ax1.plot(epochs, history_data['consistency_loss'], label='Cons. Loss (Train)', c='green', ls=':')
    if 'val_loss' in history_data: ax1.plot(epochs, history_data['val_loss'], label='Total Val Loss', c='red')
    ax1.set_title('Losses'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)

    # Dice Scores
    if 'student_dice' in history_data: ax2.plot(epochs, history_data['student_dice'], label='Student Dice (Train)', c='dodgerblue')
    elif 'pretrain_student_dice' in history_data: ax2.plot(epochs, history_data['pretrain_student_dice'], label='Student Dice (PreTrain)', c='dodgerblue') # For pretrain plot
    
    if 'val_student_dice' in history_data: ax2.plot(epochs, history_data['val_student_dice'], label='Student Dice (Val)', c='deepskyblue')
    elif 'val_pretrain_student_dice' in history_data: ax2.plot(epochs, history_data['val_pretrain_student_dice'], label='Student Dice (Val PreTrain)', c='deepskyblue')

    if 'val_teacher_dice' in history_data: ax2.plot(epochs, history_data['val_teacher_dice'], label='Teacher Dice (Val)', c='lightcoral', ls='--')
    
    ax2.set_title('Dice Scores'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice Score'); ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True)

    plt.suptitle(plot_title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = plot_save_dir / f"{plot_title_prefix.replace(' ','_').lower()}_curves.png"
    try: plt.savefig(plot_filename); tf.print(f"Saved plot: {plot_filename}")
    except Exception as e: tf.print(f"Error saving plot: {e}", output_stream=sys.stderr)
    plt.close()

def main(args):
    exp_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='semi-supervised-mean-teacher',
        results_dir=Path(args.output_dir) / args.experiment_name
    )
    exp_config.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_config.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)


    model_data_config = StableSSLConfig(
        img_size_x=args.img_size, img_size_y=args.img_size,
        num_channels=1, num_classes=1, batch_size=args.batch_size 
    )

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]; tf.print(f"Using {len(gpus)} GPU(s).")
        except RuntimeError as e: tf.print(f"GPU Error: {e}", sys.stderr)
    else: tf.print("No GPU. Using CPU.", sys.stderr)

    tf.print("Preparing data paths...")
    data_paths = prepare_data_paths(args.data_dir, args.num_labeled, args.num_validation, args.seed)

    tf.print("Initializing DataPipeline...")
    data_pipeline = DataPipeline(config=model_data_config)

    # --- STUDENT PRE-TRAINING PHASE ---
    if args.student_pretrain_epochs > 0:
        tf.print(f"--- Starting Student Pre-training Phase ({args.student_pretrain_epochs} epochs) ---")
        pretrain_student_model = PancreasSeg(model_data_config)
        dummy_input = tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels))
        _ = pretrain_student_model(dummy_input, training=False)

        pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        pretrain_loss = DiceBCELoss()
        pretrain_metrics = [DiceCoefficient(name='pretrain_student_dice')]
        
        pretrain_student_model.compile(optimizer=pretrain_optimizer, loss=pretrain_loss, metrics=pretrain_metrics)

        if not data_paths['labeled']['images']: sys.exit("CRITICAL ERROR: No labeled data for pre-training.")
        pretrain_labeled_dataset = data_pipeline.build_labeled_dataset(
            data_paths['labeled']['images'], data_paths['labeled']['labels'],
            args.batch_size, is_training=True).prefetch(tf.data.AUTOTUNE)

        if not data_paths['validation']['images']: sys.exit("CRITICAL ERROR: No validation data for pre-training.")
        pretrain_validation_dataset = data_pipeline.build_validation_dataset(
             data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size
        ).prefetch(tf.data.AUTOTUNE)


        if tf.data.experimental.cardinality(pretrain_labeled_dataset) == 0:
            sys.exit("CRITICAL ERROR: Labeled dataset for pre-training is empty after build.")

        pretrain_csv_log_path = exp_config.results_dir / "pretrain_log.csv"
        pretrain_history = pretrain_student_model.fit(
            pretrain_labeled_dataset,
            epochs=args.student_pretrain_epochs,
            validation_data=pretrain_validation_dataset,
            verbose=args.verbose,
            callbacks=[tf.keras.callbacks.CSVLogger(str(pretrain_csv_log_path))]
        )
        tf.print("--- Student Pre-training Phase Completed ---")
        plot_title = f"Student Pre-Training ({args.num_labeled} Labeled)"
        plot_training_summary(pretrain_history, pretrain_csv_log_path, plots_dir, plot_title)
    else:
        pretrain_student_model = None # No pre-training
        tf.print("--- Skipping Student Pre-training Phase ---")


    # --- MEAN TEACHER TRAINING PHASE ---
    tf.print("Instantiating student and teacher models for Mean Teacher phase...")
    student_model = PancreasSeg(model_data_config)
    teacher_model = PancreasSeg(model_data_config)
    dummy_input = tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels))
    _ = student_model(dummy_input, training=False); _ = teacher_model(dummy_input, training=False)

    if pretrain_student_model is not None:
        student_model.set_weights(pretrain_student_model.get_weights())
        tf.print("Main student model weights initialized from pre-trained student.")
        del pretrain_student_model # Free memory
    else:
        tf.print("Main student model initialized with random weights.")
    
    teacher_model.set_weights(student_model.get_weights())
    tf.print("Teacher model weights initialized from student model.")

    mean_teacher_trainer = MeanTeacherTrainer(student_model, teacher_model, args.ema_decay)
    optimizer_mt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate) # Fresh optimizer for MT phase
    supervised_loss_mt = DiceBCELoss()
    student_dice_metric_mt = DiceCoefficient(name='student_dice')
    mean_teacher_trainer.compile(optimizer_mt, supervised_loss_mt, metrics=[student_dice_metric_mt])
    tf.print("MeanTeacherTrainer compiled for MT phase.")

    # Datasets for MT phase
    if not data_paths['labeled']['images']: sys.exit("CRITICAL: No labeled data for MT phase.")
    labeled_train_dataset_mt = data_pipeline.build_labeled_dataset(
        data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(tf.data.AUTOTUNE)

    if data_paths['unlabeled']['images']:
        unlabeled_train_dataset_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(
            data_paths['unlabeled']['images'], args.batch_size, True).repeat().prefetch(tf.data.AUTOTUNE)
    else:
        tf.print("INFO: No actual unlabeled data. Using subset of labeled for dummy unlabeled dataset.", output_stream=sys.stderr)
        num_dummy = min(args.batch_size * 2, len(data_paths['labeled']['images']))
        if num_dummy == 0: sys.exit("CRITICAL: Cannot create dummy unlabeled from empty labeled set.")
        unlabeled_train_dataset_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(
            data_paths['labeled']['images'][:num_dummy], args.batch_size, True).repeat().prefetch(tf.data.AUTOTUNE)

    if not data_paths['validation']['images']: sys.exit("CRITICAL: No validation data for MT phase.")
    validation_dataset_mt = data_pipeline.build_validation_dataset(
         data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size
    ).prefetch(tf.data.AUTOTUNE)

    if any(ds is None or tf.data.experimental.cardinality(ds) == 0 for ds in [labeled_train_dataset_mt, unlabeled_train_dataset_mt, validation_dataset_mt if validation_dataset_mt is not None else labeled_train_dataset_mt]): # check val if exists
        sys.exit("CRITICAL ERROR: One or more datasets for MT phase is None or empty after build.")

    train_dataset_mt = tf.data.Dataset.zip((labeled_train_dataset_mt, unlabeled_train_dataset_mt))

    callbacks_mt = [
        ConsistencyWeightScheduler(mean_teacher_trainer.consistency_weight, args.consistency_max, args.consistency_rampup),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(exp_config.results_dir / "checkpoints" / "best_mt_student_epoch_{epoch:02d}_val_dice_{val_student_dice:.4f}.h5"),
            save_weights_only=True, monitor='val_student_dice', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(exp_config.results_dir / "logs_mt"), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(str(exp_config.results_dir / "mean_teacher_training_log.csv")),
        tf.keras.callbacks.EarlyStopping(monitor='val_student_dice', patience=args.early_stopping_patience, mode='max', verbose=1, restore_best_weights=True)
    ]
    
    tf.print(f"--- Starting Mean Teacher Training Phase ({args.num_epochs} epochs) ---")
    start_time_mt = time.time()
    history_mt = mean_teacher_trainer.fit(
        train_dataset_mt, epochs=args.num_epochs, validation_data=validation_dataset_mt,
        callbacks=callbacks_mt, verbose=args.verbose
    )
    end_time_mt = time.time()
    tf.print(f"Mean Teacher training phase completed in {(end_time_mt - start_time_mt)/60:.2f} minutes.")
    
    plot_title_mt = f"Mean Teacher ({args.num_labeled} Labeled)"
    plot_training_summary(history_mt, exp_config.results_dir / "mean_teacher_training_log.csv", plots_dir, plot_title_mt)

    final_model_path = exp_config.results_dir / "final_mt_student_model.h5"
    student_model.save_weights(str(final_model_path))
    tf.print(f"Final Mean Teacher student model weights saved to {final_model_path}")
    tf.print(f"All results, checkpoints, and logs saved in: {exp_config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mean Teacher training for pancreas segmentation.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--experiment_name", type=str, default=f"MT_Pretrain_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_labeled", type=int, default=30)
    parser.add_argument("--num_validation", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--student_pretrain_epochs", type=int, default=10, help="Epochs for supervised pre-training of student.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs for Mean Teacher training phase.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--consistency_max", type=float, default=10.0)
    parser.add_argument("--consistency_rampup", type=int, default=30, help="Epochs for consistency weight ramp-up IN MT PHASE.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    main(args)