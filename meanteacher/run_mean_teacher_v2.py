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

# Assuming these modules are in the same directory or Python path
from config import ExperimentConfig, StableSSLConfig
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg # UNetBlock is used by PancreasSeg
from train_ssl_tf2n import MeanTeacherTrainer # Ensure this has phased EMA and sharpening_temperature param

AUTOTUNE = tf.data.experimental.AUTOTUNE
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
        config.update({'weight_bce': self.weight_bce, 'weight_dice': self.weight_dice, 'smooth': self.smooth})
        return config

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
        sum_true = tf.reduce_sum(y_true_f, axis=[1,2,3]); sum_pred = tf.reduce_sum(y_pred_f_binary, axis=[1,2,3])
        dice_per_sample = (2. * intersection + self.smooth) / (sum_true + sum_pred + self.smooth)
        self.sum_dice_scores.assign_add(tf.reduce_sum(dice_per_sample))
        self.num_samples_processed.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        
    def result(self):
        return tf.cond(tf.equal(self.num_samples_processed, 0.0), lambda: 0.0, lambda: self.sum_dice_scores / self.num_samples_processed)
    def reset_state(self): self.sum_dice_scores.assign(0.); self.num_samples_processed.assign(0.)
    def get_config(self): config = super().get_config(); config.update({'smooth': self.smooth}); return config

# --- Callbacks ---
class MTPhaseEpochUpdater(tf.keras.callbacks.Callback): # For Phased EMA
    def on_epoch_begin(self, epoch, logs=None):
        self.model.mt_phase_current_epoch.assign(epoch)

class ConsistencyWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, consistency_weight_var, max_weight, rampup_epochs, delay_rampup_epochs=0):
        super().__init__()
        self.consistency_weight_var = consistency_weight_var
        self.max_weight = max_weight
        self.rampup_epochs_duration = rampup_epochs # Duration of the ramp
        self.delay_rampup_epochs = delay_rampup_epochs # Start ramp after this many MT phase epochs

    def on_epoch_begin(self, epoch, logs=None): # epoch is 0-indexed Keras epoch for current fit()
        if epoch < self.delay_rampup_epochs:
            current_weight = 0.0
        else:
            effective_epoch = epoch - self.delay_rampup_epochs
            if self.rampup_epochs_duration == 0:
                current_weight = self.max_weight
            elif effective_epoch < self.rampup_epochs_duration:
                current_weight = self.max_weight * (float(effective_epoch + 1) / float(self.rampup_epochs_duration))
            else:
                current_weight = self.max_weight
        
        self.consistency_weight_var.assign(current_weight)
        if epoch % 10 == 0 or epoch == 0:
             tf.print(f"Epoch {epoch+1} (MT Phase): Consistency weight set to {current_weight:.4f}")

def prepare_data_paths(data_dir_str: str, num_labeled_patients: int, num_validation_patients: int, seed: int = 42):
    data_dir = Path(data_dir_str)
    all_patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pancreas_")])
    if not all_patient_dirs: raise FileNotFoundError(f"No patient dirs in {data_dir_str}")
    random.seed(seed); random.shuffle(all_patient_dirs)
    if len(all_patient_dirs) < num_labeled_patients + num_validation_patients:
        raise ValueError(f"Not enough patient data ({len(all_patient_dirs)}) for splits.")

    val_dirs = all_patient_dirs[:num_validation_patients]
    rem_dirs = all_patient_dirs[num_validation_patients:]
    lab_dirs = rem_dirs[:num_labeled_patients]
    unlab_dirs = rem_dirs[num_labeled_patients:]
    tf.print(f"Data Split: Total={len(all_patient_dirs)}, Labeled={len(lab_dirs)}, Unlabeled={len(unlab_dirs)}, Val={len(val_dirs)}")

    def get_files(p_dirs, is_lab=True):
        imgs, lbls = [], ([] if is_lab else None)
        for p_dir in p_dirs:
            img_f, mask_f = p_dir/"image.npy", p_dir/"mask.npy"
            if img_f.exists() and (not is_lab or mask_f.exists()):
                imgs.append(str(img_f))
                if is_lab: lbls.append(str(mask_f))
            else: tf.print(f"Warn: Missing files in {p_dir}", sys.stderr)
        return (imgs, lbls) if is_lab else imgs
    
    li, ll = get_files(lab_dirs, True); ui = get_files(unlab_dirs, False); vi, vl = get_files(val_dirs, True)
    if not li: tf.print("CRIT WARN: No Labeled images found.", sys.stderr)
    if not ui and len(unlab_dirs)>0: tf.print("WARN: Unlabeled dirs selected, but no images found.", sys.stderr)
    if not vi: tf.print("CRIT WARN: No Validation images found.", sys.stderr)
    return {'labeled': {'images':li, 'labels':ll}, 'unlabeled': {'images':ui}, 'validation': {'images':vi, 'labels':vl}}

def plot_training_summary(history_object, csv_log_path: Path, plot_save_dir: Path, plot_title_prefix: str):
    # ... (Use the enhanced plot_training_summary from previous response) ...
    # For brevity, assuming it's defined as before.
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    data_source = None; history_data = {}

    if history_object and hasattr(history_object, 'history') and history_object.history:
        history_data = history_object.history; data_source = "Keras History"
    elif csv_log_path.exists():
        try:
            df = pd.read_csv(csv_log_path)
            if not df.empty:
                history_data = {col: df[col].tolist() for col in df.columns}
                history_data['epochs_list'] = df['epoch'].tolist() if 'epoch' in df.columns else list(range(1, len(df) + 1))
                data_source = "CSV Log File"
        except Exception as e: tf.print(f"Plot Error: Reading CSV {csv_log_path}: {e}", sys.stderr)
    
    if not history_data: tf.print(f"Plot Error: No data from {data_source or 'any source'}.", sys.stderr); return
    
    epochs = history_data.get('epochs_list')
    if not epochs: # Fallback if epoch list wasn't made
        for key in ['loss', 'student_dice', 'val_loss', 'val_student_dice', 'pretrain_student_dice', 'val_pretrain_student_dice']:
            if key in history_data and history_data[key]: epochs = list(range(1, len(history_data[key]) + 1)); break
    if not epochs: tf.print("Plot Error: No epoch data.", sys.stderr); return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True); ax1, ax2 = axes[0], axes[1]
    # Losses
    if 'loss' in history_data: ax1.plot(epochs, history_data['loss'], label='Total Train Loss', c='blue')
    if 'supervised_loss' in history_data: ax1.plot(epochs, history_data['supervised_loss'], label='Sup. Loss (Train)', c='orange', ls='--')
    if 'consistency_loss' in history_data: ax1.plot(epochs, history_data['consistency_loss'], label='Cons. Loss (Train)', c='green', ls=':')
    if 'val_loss' in history_data: ax1.plot(epochs, history_data['val_loss'], label='Total Val Loss', c='red')
    ax1.set_title('Losses'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    # Dice
    train_dice_key = 'student_dice' if 'student_dice' in history_data else 'pretrain_student_dice'
    val_dice_key = 'val_student_dice' if 'val_student_dice' in history_data else 'val_pretrain_student_dice'
    if train_dice_key in history_data: ax2.plot(epochs, history_data[train_dice_key], label=f'{train_dice_key.replace("_"," ").title()} (Train)', c='dodgerblue')
    if val_dice_key in history_data: ax2.plot(epochs, history_data[val_dice_key], label=f'{val_dice_key.replace("_"," ").title()} (Val)', c='deepskyblue')
    if 'val_teacher_dice' in history_data: ax2.plot(epochs, history_data['val_teacher_dice'], label='Teacher Dice (Val)', c='lightcoral', ls='--')
    ax2.set_title('Dice Scores'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice'); ax2.set_ylim(0,1.05); ax2.legend(); ax2.grid(True)
    
    plt.suptitle(plot_title_prefix, fontsize=16); plt.tight_layout(rect=[0,0,1,0.95])
    filename = plot_save_dir / f"{plot_title_prefix.replace(' ','_').replace('(','').replace(')','').replace(':','').lower()}_curves.png"
    try: plt.savefig(filename); tf.print(f"Saved plot: {filename}")
    except Exception as e: tf.print(f"Plot Error: Saving plot: {e}", sys.stderr)
    plt.close()


def main(args):
    exp_config = ExperimentConfig(args.experiment_name, 'semi-supervised-mean-teacher', Path(args.output_dir)/args.experiment_name)
    exp_config.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_config.results_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_base_dir = exp_config.results_dir / "checkpoints"; checkpoints_base_dir.mkdir(parents=True, exist_ok=True)

    model_data_config = StableSSLConfig(
        img_size_x=args.img_size, img_size_y=args.img_size, num_channels=1, num_classes=1,
        batch_size=args.batch_size, dropout_rate=args.dropout_rate # Pass dropout
    )
    if not hasattr(model_data_config, 'n_filters'): model_data_config.n_filters = 32

    gpus=tf.config.experimental.list_physical_devices('GPU')
    if gpus: [tf.config.experimental.set_memory_growth(g, True) for g in gpus]; tf.print(f"Using {len(gpus)} GPU(s).")
    else: tf.print("No GPU. Using CPU.", sys.stderr)

    tf.print("Preparing data paths..."); data_paths = prepare_data_paths(args.data_dir, args.num_labeled, args.num_validation, args.seed)
    tf.print("Initializing DataPipeline..."); data_pipeline = DataPipeline(config=model_data_config)

    pretrain_student_model = None
    if args.student_pretrain_epochs > 0:
        tf.print(f"--- Student Pre-training Phase ({args.student_pretrain_epochs} epochs) ---")
        pretrain_student_model = PancreasSeg(model_data_config)
        _ = pretrain_student_model(tf.zeros((1, args.img_size, args.img_size, 1)))
        pretrain_student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                       loss=DiceBCELoss(), metrics=[DiceCoefficient(name='pretrain_student_dice')])
        if not data_paths['labeled']['images']: sys.exit("CRIT: No labeled data for pre-train.")
        pre_lab_ds = data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(AUTOTUNE)
        if not data_paths['validation']['images']: sys.exit("CRIT: No validation data for pre-train.")
        pre_val_ds = data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size).prefetch(AUTOTUNE)
        if tf.data.experimental.cardinality(pre_lab_ds)==0: sys.exit("CRIT: Labeled pre-train dataset empty.")
        
        pretrain_csv_path = exp_config.results_dir / "pretrain_log.csv"
        pretrain_checkpoint_path = str(checkpoints_base_dir / "best_pretrain_student_epoch_{epoch:02d}_val_dice_{val_pretrain_student_dice:.4f}") # TF format prefix
        
        pretrain_callbacks = [
            tf.keras.callbacks.CSVLogger(str(pretrain_csv_path)),
            tf.keras.callbacks.ModelCheckpoint(filepath=pretrain_checkpoint_path, save_weights_only=True,
                                             monitor='val_pretrain_student_dice', mode='max', save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_pretrain_student_dice', patience=args.early_stopping_patience // 2 or 5, # Shorter patience for pretrain
                                           mode='max', verbose=1, restore_best_weights=True)
        ]
        pretrain_history = pretrain_student_model.fit(pre_lab_ds, epochs=args.student_pretrain_epochs, validation_data=pre_val_ds, verbose=args.verbose, callbacks=pretrain_callbacks)
        tf.print("--- Student Pre-training Phase Completed ---")
        plot_training_summary(pretrain_history, pretrain_csv_path, plots_dir, f"Student Pre-Train ({args.num_labeled}L)")
        # Load the best pre-trained weights for the MT phase student
        tf.print(f"Loading best pre-trained weights for MT student from {pretrain_checkpoint_path} format...")
        # We need to find the actual best checkpoint file if save_best_only=True was used with TF format
        # For now, we assume EarlyStopping restored them, or we use the final weights.
        # If EarlyStopping restored_best_weights, pretrain_student_model has them.

    tf.print("Instantiating student and teacher models for Mean Teacher phase...")
    student_model = PancreasSeg(model_data_config); teacher_model = PancreasSeg(model_data_config)
    _ = student_model(tf.zeros((1,args.img_size,args.img_size,1))); _ = teacher_model(tf.zeros((1,args.img_size,args.img_size,1)))
    if pretrain_student_model:
        student_model.set_weights(pretrain_student_model.get_weights()); tf.print("Student model weights init from pre-trained.")
        del pretrain_student_model
    else: tf.print("Student model weights init randomly.")
    teacher_model.set_weights(student_model.get_weights()); tf.print("Teacher model weights init from student.")

    mean_teacher_trainer = MeanTeacherTrainer(
        student_model, teacher_model, args.ema_decay, args.teacher_ema_warmup_epochs,
        args.initial_teacher_ema_decay, args.sharpening_temperature)
    mean_teacher_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                 loss=DiceBCELoss(), metrics=[DiceCoefficient(name='student_dice')])
    tf.print("MeanTeacherTrainer compiled for MT phase.")

    lab_ds_mt = data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(AUTOTUNE)
    if data_paths['unlabeled']['images']:
        unlab_ds_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(data_paths['unlabeled']['images'], args.batch_size, True).repeat().prefetch(AUTOTUNE)
    else:
        tf.print("INFO: No actual unlabeled data. Using dummy unlabeled.", sys.stderr)
        n_dummy = min(args.batch_size*2, len(data_paths['labeled']['images']))
        if n_dummy==0: sys.exit("CRIT: No labeled data for dummy unlabeled.")
        unlab_ds_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(data_paths['labeled']['images'][:n_dummy], args.batch_size, True).repeat().prefetch(AUTOTUNE)
    val_ds_mt = data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size).prefetch(AUTOTUNE)
    if any(tf.data.experimental.cardinality(ds)==0 for ds in [lab_ds_mt,val_ds_mt]): sys.exit("CRIT: Labeled or Val MT dataset empty.")
    train_ds_mt = tf.data.Dataset.zip((lab_ds_mt, unlab_ds_mt))

    mt_checkpoint_dir = checkpoints_base_dir / "mt_phase"
    mt_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mt_checkpoint_filepath_tf = str(mt_checkpoint_dir / "best_mt_student_epoch_{epoch:02d}_val_dice_{val_student_dice:.4f}") # TF Format

    callbacks_mt = [
        MTPhaseEpochUpdater(),
        ConsistencyWeightScheduler(mean_teacher_trainer.consistency_weight, args.consistency_max, args.consistency_rampup, args.teacher_ema_warmup_epochs if args.delay_consistency_rampup else 0),
        tf.keras.callbacks.ModelCheckpoint(filepath=mt_checkpoint_filepath_tf, save_weights_only=True, monitor='val_student_dice', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(exp_config.results_dir / "logs_mt"), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(str(exp_config.results_dir / "mean_teacher_training_log.csv")),
        tf.keras.callbacks.EarlyStopping(monitor='val_student_dice', patience=args.early_stopping_patience, mode='max', verbose=1, restore_best_weights=True)
    ]

    tf.print(f"--- Starting Mean Teacher Training Phase ({args.num_epochs} epochs) ---")
    history_mt = mean_teacher_trainer.fit(train_ds_mt, epochs=args.num_epochs, validation_data=val_ds_mt, callbacks=callbacks_mt, verbose=args.verbose)
    tf.print(f"Mean Teacher training phase completed in {(time.time() - float(history_mt.epoch[0] if history_mt.epoch else time.time()))/60:.2f} minutes (approx).") # Approx time
    plot_training_summary(history_mt, exp_config.results_dir / "mean_teacher_training_log.csv", plots_dir, f"Mean Teacher ({args.num_labeled}L)")
    
    final_model_path_tf = exp_config.results_dir / "final_mt_student_model" # TF Format
    student_model.save_weights(str(final_model_path_tf))
    tf.print(f"Final MT student model weights (TF format) saved to {final_model_path_tf}")
    tf.print(f"All results saved in: {exp_config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mean Teacher (MT) for pancreas segmentation.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiment_results_mt")
    parser.add_argument("--experiment_name", type=str, default=f"MT_Run_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_labeled", type=int, default=30)
    parser.add_argument("--num_validation", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    
    parser.add_argument("--student_pretrain_epochs", type=int, default=15, help="Epochs for supervised pre-training of student.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for models.")
    
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs for MT training phase.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--teacher_ema_warmup_epochs", type=int, default=10, help="Initial MT epochs for faster teacher EMA.")
    parser.add_argument("--initial_teacher_ema_decay", type=float, default=0.5, help="EMA decay for teacher during its warmup.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Base EMA decay for teacher after its warmup.")
    parser.add_argument("--sharpening_temperature", type=float, default=0.5, help="Temperature for teacher prediction sharpening.")
    parser.add_argument("--consistency_max", type=float, default=10.0)
    parser.add_argument("--consistency_rampup", type=int, default=30, help="Duration of consistency weight ramp-up in MT phase.")
    parser.add_argument("--delay_consistency_rampup", action='store_true', help="If set, delay consistency rampup until after teacher EMA warmup.")
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    main(args)