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

# --- GPU MEMORY GROWTH SETUP (Run this first) ---
gpus_for_setup = tf.config.experimental.list_physical_devices('GPU')
if gpus_for_setup:
    try:
        for gpu_setup in gpus_for_setup:
            tf.config.experimental.set_memory_growth(gpu_setup, True)
        print(f"SUCCESS: Set memory growth for {len(gpus_for_setup)} GPU(s).")
    except RuntimeError as e_setup:
        print(f"ERROR setting memory growth: {e_setup}", file=sys.stderr)
else:
    print("No GPUs found by TensorFlow for memory growth setup.")
# --- END GPU MEMORY GROWTH SETUP ---


# Assuming these modules are in the same directory or Python path
from config import ExperimentConfig, StableSSLConfig
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg , UNetBlock, DiceCoefficient #is used by PancreasSeg
from train_ssl_tf2n import MeanTeacherTrainer # Ensure this has phased EMA and sharpening_temperature param

# FOR DEBUGGING DATA PIPELINE / CALLBACKS / TF.FUNCTION ISSUES:
tf.config.run_functions_eagerly(True)
tf.print("INFO: TensorFlow is running functions EAGERLY for debugging.")

AUTOTUNE = tf.data.experimental.AUTOTUNE
# --- Define Custom Loss and Metric ---
class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6, name='dice_bce_loss'):
        super().__init__(name=name); self.w_bce=weight_bce; self.w_dice=weight_dice; self.s=smooth
        self.bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, yt, yp_logits):
        yt_f=tf.cast(tf.reshape(yt,[-1]),tf.float32); ypl_f=tf.cast(tf.reshape(yp_logits,[-1]),tf.float32)
        ypp_f_dice=tf.nn.sigmoid(ypl_f); intr=tf.reduce_sum(yt_f*ypp_f_dice)
        d_num=2.*intr+self.s; d_den=tf.reduce_sum(yt_f)+tf.reduce_sum(ypp_f_dice)+self.s; d_loss=1-(d_num/d_den)
        b_loss=self.bce_fn(yt_f,ypl_f); return self.w_dice*d_loss + self.w_bce*b_loss
    
    def get_config(self): conf=super().get_config(); conf.update({'weight_bce':self.w_bce,'weight_dice':self.w_dice,'smooth':self.s}); return conf



# --- Callbacks ---
class MTPhaseEpochUpdater(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'mt_phase_current_epoch'): self.model.mt_phase_current_epoch.assign(epoch)

class TeacherDirectCopyCallback(tf.keras.callbacks.Callback):
    def __init__(self, student_model_to_copy_from, direct_copy_epochs=0):
        super().__init__()
        self.student_to_copy = student_model_to_copy_from
        self.direct_copy_epochs = direct_copy_epochs
        print(f"TeacherDirectCopyCallback: Initialized. Will use explicit copy for first {self.direct_copy_epochs} MT epochs.")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.direct_copy_epochs:
            tf.print(f"MT Epoch {epoch + 1}/{self.direct_copy_epochs}: TeacherDirectCopyCallback - Performing EXPLICIT component-wise weight copy.", output_stream=sys.stderr)
            
            # self.model is the MeanTeacherTrainer instance
            # self.student_to_copy is the actual student Keras model (e.g., student_model_mt)
            if hasattr(self.model, '_perform_explicit_model_copy') and callable(self.model._perform_explicit_model_copy):
                copy_success = self.model._perform_explicit_model_copy(self.student_to_copy, self.model.teacher_model)
                if copy_success:
                    tf.print(f"  SUCCESS: TeacherDirectCopyCallback - Explicit copy for MT Epoch {epoch + 1} successful.", output_stream=sys.stderr)
                else:
                    tf.print(f"  ERROR: TeacherDirectCopyCallback - Explicit copy for MT Epoch {epoch + 1} FAILED.", output_stream=sys.stderr)
            else:
                tf.print("  ERROR: TeacherDirectCopyCallback - MeanTeacherTrainer instance does not have _perform_explicit_model_copy method.", output_stream=sys.stderr)


class ConsistencyWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, consistency_weight_var, max_weight, 
                 rampup_epochs_duration,  # Renamed for clarity from just rampup_epochs
                 delay_rampup_epochs=0):  # Added new parameter with default
        super().__init__()
        self.consistency_weight_var = consistency_weight_var
        self.max_weight = max_weight
        self.rampup_epochs_duration = rampup_epochs_duration # Duration of the ramp itself
        self.delay_rampup_epochs = delay_rampup_epochs     # How many epochs to wait before starting ramp

    def on_epoch_begin(self, epoch, logs=None): # epoch is 0-indexed Keras epoch for current fit()
        current_weight = 0.0
        if epoch >= self.delay_rampup_epochs:
            # Ramp-up phase, relative to the end of the delay
            effective_epoch_for_rampup = epoch - self.delay_rampup_epochs
            
            if self.rampup_epochs_duration == 0: # If ramp duration is 0, immediately set to max_weight
                current_weight = self.max_weight
            elif effective_epoch_for_rampup < self.rampup_epochs_duration:
                current_weight = self.max_weight * (float(effective_epoch_for_rampup + 1) / float(self.rampup_epochs_duration))
            else: # Ramp-up complete
                current_weight = self.max_weight
        # else: current_weight remains 0.0 (during delay phase)
        
        self.consistency_weight_var.assign(current_weight)
        
        if epoch % 10 == 0 or epoch == 0: # Log every 10 epochs or the first epoch
             tf.print(f"Epoch {epoch+1} (MT Phase): ConsistencyWeight set to {current_weight:.4f} "
                      f"(delay_epochs={self.delay_rampup_epochs}, ramp_duration={self.rampup_epochs_duration})")
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
        batch_size=args.batch_size, dropout_rate=args.dropout_rate
    )
    if not hasattr(model_data_config, 'n_filters'): model_data_config.n_filters = 32 # Default if not in config

    print("Preparing data paths..."); data_paths = prepare_data_paths(args.data_dir, args.num_labeled, args.num_validation, args.seed)
    print("Initializing DataPipeline..."); data_pipeline = DataPipeline(config=model_data_config)

    pretrain_student_model = None
    if args.student_pretrain_epochs > 0:
        print(f"--- Student Pre-training Phase ({args.student_pretrain_epochs} epochs) ---")
        pretrain_student_model = PancreasSeg(model_data_config)
        _ = pretrain_student_model(tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels)))
        pretrain_student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                       loss=DiceBCELoss(), metrics=[DiceCoefficient(name='pretrain_student_dice')])
        if not data_paths['labeled']['images']: sys.exit("CRIT: No labeled data for pre-train.")
        pre_lab_ds = data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(AUTOTUNE)
        if not data_paths['validation']['images']: sys.exit("CRIT: No validation data for pre-train.")
        pre_val_ds = data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size).prefetch(AUTOTUNE)
        if tf.data.experimental.cardinality(pre_lab_ds)==0: sys.exit("CRIT: Labeled pre-train dataset empty.")
        
        pretrain_csv_path = exp_config.results_dir / "pretrain_log.csv"
        pretrain_ckpt_dir = checkpoints_base_dir / "pretrain"
        pretrain_ckpt_dir.mkdir(parents=True, exist_ok=True)
        pretrain_checkpoint_prefix = str(pretrain_ckpt_dir / "best_pretrain_epoch_{epoch:02d}_val_dice_{val_pretrain_student_dice:.4f}")
        
        pretrain_callbacks = [
            tf.keras.callbacks.CSVLogger(str(pretrain_csv_path)),
            tf.keras.callbacks.ModelCheckpoint(filepath=pretrain_checkpoint_prefix, save_weights_only=True,
                                             monitor='val_pretrain_student_dice', mode='max', save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_pretrain_student_dice', patience=max(5, args.early_stopping_patience // 2),
                                           mode='max', verbose=1, restore_best_weights=True)
        ]
        pretrain_history = pretrain_student_model.fit(pre_lab_ds, epochs=args.student_pretrain_epochs, validation_data=pre_val_ds, verbose=args.verbose, callbacks=pretrain_callbacks)
        print("--- Student Pre-training Phase Completed ---")
        plot_training_summary(pretrain_history, pretrain_csv_path, plots_dir, f"Student Pre-Train ({args.num_labeled}L)")

    print("Instantiating student and teacher models for Mean Teacher phase...")
    student_model_mt = PancreasSeg(model_data_config)
    teacher_model_mt = PancreasSeg(model_data_config)
    dummy_input_build = tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels))
    _ = student_model_mt(dummy_input_build); _ = teacher_model_mt(dummy_input_build)

    if pretrain_student_model:
        student_model_mt.set_weights(pretrain_student_model.get_weights())
        print("MT Student model weights initialized from best pre-trained student.")
        del pretrain_student_model
    else: print("MT Student model weights initialized randomly (no pre-training).")
    
    teacher_model_mt.set_weights(student_model_mt.get_weights())
    print("MT Teacher model weights initialized from MT student model.")

    # Instantiate MeanTeacherTrainer with only necessary EMA params for the simplified strategy
    mean_teacher_trainer = MeanTeacherTrainer(
        student_model=student_model_mt, 
        teacher_model=teacher_model_mt, 
        ema_decay=args.ema_decay, # This is the base_ema_decay used by trainer's _update_teacher_model
        sharpening_temperature=args.sharpening_temperature,
        teacher_direct_copy_epochs_config=args.teacher_direct_copy_epochs # Pass this
        # teacher_direct_copy_epochs is NOT a parameter of MeanTeacherTrainer.__init__
    )
    mean_teacher_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                 loss=DiceBCELoss(), metrics=[DiceCoefficient(name='student_dice')])
    print("MeanTeacherTrainer compiled for MT phase.")

    lab_ds_mt = data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(AUTOTUNE)
    if data_paths['unlabeled']['images']:
        unlab_ds_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(data_paths['unlabeled']['images'], args.batch_size, True).repeat().prefetch(AUTOTUNE)
    else:
        print("INFO: No actual unlabeled data. Using dummy unlabeled.", file=sys.stderr)
        n_dummy = min(args.batch_size*2, len(data_paths['labeled']['images']))
        if n_dummy==0: sys.exit("CRIT: No labeled data for dummy unlabeled.")
        unlab_ds_mt = data_pipeline.build_unlabeled_dataset_for_mean_teacher(data_paths['labeled']['images'][:n_dummy], args.batch_size, True).repeat().prefetch(AUTOTUNE)
    val_ds_mt = data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size).prefetch(AUTOTUNE)
    if any(tf.data.experimental.cardinality(ds)==0 for ds in [lab_ds_mt,val_ds_mt]): sys.exit("CRIT: Labeled or Val MT dataset empty.")
    train_ds_mt = tf.data.Dataset.zip((lab_ds_mt, unlab_ds_mt))

    mt_checkpoint_dir = checkpoints_base_dir / "mt_phase"
    mt_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mt_checkpoint_filepath_tf = str(mt_checkpoint_dir / "best_mt_student_epoch_{epoch:02d}_val_dice_{val_student_dice:.4f}") 

    callbacks_mt = [
        MTPhaseEpochUpdater(), # Sets trainer.mt_phase_current_epoch
        TeacherDirectCopyCallback(student_model_mt, args.teacher_direct_copy_epochs), # Uses args.teacher_direct_copy_epochs
        ConsistencyWeightScheduler(
            mean_teacher_trainer.consistency_weight, args.consistency_max, args.consistency_rampup, 
            delay_rampup_epochs=args.teacher_direct_copy_epochs if args.delay_consistency_rampup_by_copy_phase else 0),
        tf.keras.callbacks.ModelCheckpoint(filepath=mt_checkpoint_filepath_tf, save_weights_only=True, monitor='val_student_dice', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(exp_config.results_dir / "logs_mt"), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(str(exp_config.results_dir / "mean_teacher_training_log.csv")),
        tf.keras.callbacks.EarlyStopping(monitor='val_student_dice', patience=args.early_stopping_patience, mode='max', verbose=1, restore_best_weights=True)
    ]

    print(f"--- Starting Mean Teacher Training Phase ({args.num_epochs} epochs) ---")
    
    num_labeled_slices_total_mt = len(data_paths['labeled']['images'])
    if num_labeled_slices_total_mt == 0: sys.exit("CRIT: No labeled slices for MT phase steps_per_epoch.")
    steps_per_epoch_mt = (num_labeled_slices_total_mt + args.batch_size - 1) // args.batch_size
    print(f"INFO: For MT phase, using steps_per_epoch = {steps_per_epoch_mt}")

    start_time_mt = time.time() # Define start_time_mt
    history_mt = mean_teacher_trainer.fit(
        train_ds_mt, 
        epochs=args.num_epochs, 
        steps_per_epoch=steps_per_epoch_mt, # Use calculated steps
        validation_data=val_ds_mt, 
        callbacks=callbacks_mt, 
        verbose=args.verbose
    )
    end_time_mt = time.time() # Define end_time_mt
    
    mt_phase_duration_minutes = (end_time_mt - start_time_mt) / 60.0
    print(f"Mean Teacher training phase completed in {mt_phase_duration_minutes:.2f} minutes.")
    plot_training_summary(history_mt, exp_config.results_dir / "mean_teacher_training_log.csv", plots_dir, f"Mean Teacher ({args.num_labeled}L)")
    
    final_model_path_tf = exp_config.results_dir / "final_mt_student_model" 
    student_model_mt.save_weights(str(final_model_path_tf))
    print(f"Final MT student model weights (TF format) saved to {final_model_path_tf}")
    print(f"All results saved in: {exp_config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mean Teacher (MT) for pancreas segmentation.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./MT_FinalStrategy_Runs") # Changed default
    parser.add_argument("--experiment_name", type=str, default=f"MT_Run_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_labeled", type=int, default=30)
    parser.add_argument("--num_validation", type=int, default=20) # Increased default validation
    parser.add_argument("--batch_size", type=int, default=4)
    
    parser.add_argument("--student_pretrain_epochs", type=int, default=15)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    
    parser.add_argument("--num_epochs", type=int, default=150, help="Epochs for MT training phase.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    parser.add_argument("--teacher_direct_copy_epochs", type=int, default=10, 
                        help="Initial MT epochs: Teacher IS student (direct copy by callback).")
    parser.add_argument("--ema_decay", type=float, default=0.999, 
                        help="EMA decay for teacher's per-batch updates (used by MeanTeacherTrainer).")
    
    parser.add_argument("--sharpening_temperature", type=float, default=0.25)
    parser.add_argument("--consistency_max", type=float, default=20.0)
    parser.add_argument("--consistency_rampup", type=int, default=40)
    parser.add_argument("--delay_consistency_rampup_by_copy_phase", action='store_true', 
                        help="If set, delay consistency rampup until after teacher direct copy phase.")
    
    parser.add_argument("--early_stopping_patience", type=int, default=30) 
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    main(args)