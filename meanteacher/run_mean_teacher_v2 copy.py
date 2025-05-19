#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import random
import time
import numpy as np
import tensorflow as tf  # Import TensorFlow first
import matplotlib.pyplot as plt
import pandas as pd

# --- STEP 1: GPU MEMORY GROWTH SETUP (MUST BE VERY EARLY) ---
gpus_for_setup = tf.config.experimental.list_physical_devices('GPU')
if gpus_for_setup:
    try:
        for gpu_setup in gpus_for_setup:
            tf.config.experimental.set_memory_growth(gpu_setup, True)
        print(f"SUCCESS: Set memory growth for {len(gpus_for_setup)} GPU(s).")
    except RuntimeError as e_setup:
        # Memory growth must be set before GPUs have been initialized by other TF ops
        print(f"ERROR setting memory growth: {e_setup}. This must be done before other TF operations.", file=sys.stderr)
else:
    print("No GPUs found by TensorFlow for memory growth setup.")
# --- END GPU MEMORY GROWTH SETUP ---

# # --- STEP 2: ATTEMPT TO DISABLE XLA JIT (BEFORE EAGER MODE OR OTHER TF OPS IF POSSIBLE) ---
print("INFO: Attempting to configure TensorFlow optimizers to NOT use JIT/XLA.")
try:
    # This tries to prevent the optimizer steps from being XLA-compiled,
    # which can conflict with tf.debugging.enable_check_numerics().
    tf.config.optimizer.set_experimental_options({'disable_jit_optimizer': True})
    # As an alternative or in older TF versions, tf.config.optimizer.set_jit(False) might be used.
    # tf.config.optimizer.set_jit(False) 
    print("  INFO: Called tf.config.optimizer options to potentially disable JIT for optimizers.")
except AttributeError:
    print("  WARN: tf.config.optimizer.set_experimental_options or set_jit not available in this TF version. XLA might still be active for optimizer parts.")
except Exception as e_jit_opt:
    print(f"  WARN: Error trying to set optimizer JIT options: {e_jit_opt}", file=sys.stderr)
# # --- END XLA JIT DISABLE ATTEMPT ---

# --- STEP 3: ENABLE NUMERIC CHECKS (BEFORE EAGER MODE IS SET for this debug purpose) ---
tf.debugging.enable_check_numerics() 
print("INFO: tf.debugging.enable_check_numerics() is ON.")
# --- END NUMERIC CHECKS ---

# --- STEP 4: EAGER EXECUTION FOR DEBUGGING (OPTIONAL - KEEP COMMENTED OUT for current NaN debug in graph mode) ---
tf.config.run_functions_eagerly(True) # UNCOMMENT this if enable_check_numerics in graph mode isn't giving enough info
if tf.config.functions_run_eagerly():
   print("INFO: TensorFlow is running functions EAGERLY for debugging.")
else:
   print("INFO: TensorFlow is running in GRAPH mode (tf.function compiled).")
# --- END EAGER EXECUTION ---

# --- STEP 5: OTHER IMPORTS AND AUTOTUNE ---
from config import ExperimentConfig, StableSSLConfig
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg, UNetBlock # Ensure UNetBlock is imported for TeacherDirectCopyCallback
from train_ssl_tf2n import MeanTeacherTrainer 

AUTOTUNE = tf.data.experimental.AUTOTUNE
# --- END OTHER IMPORTS ---

# --- Custom Loss and Metric Definitions ---
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
        return tf.cond(tf.equal(self.num_samples_processed, 0.0), lambda: tf.constant(0.0, dtype=tf.float32), lambda: self.sum_dice_scores / self.num_samples_processed)
    def reset_state(self): self.sum_dice_scores.assign(0.); self.num_samples_processed.assign(0.)
    def get_config(self): config = super().get_config(); config.update({'smooth': self.smooth}); return config

# --- Callbacks ---
class MTPhaseEpochUpdater(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'mt_phase_current_epoch'): # Check if MeanTeacherTrainer instance
             self.model.mt_phase_current_epoch.assign(epoch)

# In run_mean_teacher_v2.py

class TeacherDirectCopyCallback(tf.keras.callbacks.Callback):
    def __init__(self, student_model_ref, direct_copy_epochs=0):
        super().__init__()
        self.student_model_ref = student_model_ref 
        self.direct_copy_epochs = direct_copy_epochs
        print(f"DEBUG TeacherDirectCopyCallback: Initialized. Will copy for first {direct_copy_epochs} MT epochs.")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.direct_copy_epochs:
            tf.print(f"MT Epoch {epoch + 1}: TeacherDirectCopyCallback - Attempting EXPLICIT component-wise direct copy.", output_stream=sys.stderr)
            
            s_model = self.student_model_ref
            t_model = self.model.teacher_model # self.model is MeanTeacherTrainer

            if len(s_model.layers) != len(t_model.layers):
                tf.print(f"  CRITICAL DirectCopy: Top-level layer count MISMATCH! S:{len(s_model.layers)}, T:{len(t_model.layers)}", output_stream=sys.stderr)
                return

            successful_copy = True
            for s_layer, t_layer in zip(s_model.layers, t_model.layers):
                if type(s_layer) != type(t_layer) or s_layer.name.split('/')[-1] != t_layer.name.split('/')[-1].replace('_1',''): # Basic name check without model prefix
                    tf.print(f"  WARN DirectCopy: Layer type/name mismatch? S: {s_layer.name}, T: {t_layer.name}. Skipping this layer pair.", output_stream=sys.stderr)
                    # successful_copy = False # Potentially mark as failed
                    continue # Skip if top-level layers don't seem to match

                # Specifically handle UNetBlock instances by copying their components
                if isinstance(s_layer, UNetBlock) and isinstance(t_layer, UNetBlock):
                    try:
                        # Copy weights for conv1, bn1, conv2, bn2, dropout (if exists)
                        t_layer.conv1.set_weights(s_layer.conv1.get_weights())
                        t_layer.bn1.set_weights(s_layer.bn1.get_weights())
                        t_layer.conv2.set_weights(s_layer.conv2.get_weights())
                        t_layer.bn2.set_weights(s_layer.bn2.get_weights())
                        if hasattr(s_layer, 'dropout') and s_layer.dropout is not None and \
                           hasattr(t_layer, 'dropout') and t_layer.dropout is not None:
                            # Dropout layers don't have weights to copy, but this structure would be for future
                            pass 
                        # tf.print(f"    Copied weights for UNetBlock: {s_layer.name} -> {t_layer.name}", output_stream=sys.stderr)
                    except Exception as e_unet_copy:
                        tf.print(f"    ERROR copying UNetBlock {s_layer.name}: {e_unet_copy}", output_stream=sys.stderr)
                        successful_copy = False
                
                # Handle Conv2DTranspose layers
                elif isinstance(s_layer, tf.keras.layers.Conv2DTranspose) and isinstance(t_layer, tf.keras.layers.Conv2DTranspose):
                    try:
                        t_layer.set_weights(s_layer.get_weights())
                        # tf.print(f"    Copied weights for Conv2DTranspose: {s_layer.name} -> {t_layer.name}", output_stream=sys.stderr)
                    except Exception as e_convT_copy:
                        tf.print(f"    ERROR copying Conv2DTranspose {s_layer.name}: {e_convT_copy}", output_stream=sys.stderr)
                        successful_copy = False

                # Handle MaxPooling2D (no weights)
                elif isinstance(s_layer, tf.keras.layers.MaxPooling2D):
                    pass # No weights to copy

                # Handle final Conv2D layer
                elif s_layer.name.endswith('final_output_conv') and isinstance(s_layer, tf.keras.layers.Conv2D): # Check name
                    try:
                        t_layer.set_weights(s_layer.get_weights())
                        # tf.print(f"    Copied weights for Final Conv: {s_layer.name} -> {t_layer.name}", output_stream=sys.stderr)
                    except Exception as e_finalconv_copy:
                        tf.print(f"    ERROR copying Final Conv {s_layer.name}: {e_finalconv_copy}", output_stream=sys.stderr)
                        successful_copy = False
                else:
                    # For any other layer types, try a general set_weights if they have weights
                    if s_layer.weights: # Check if the layer has weights
                        try:
                            t_layer.set_weights(s_layer.get_weights())
                            # tf.print(f"    Copied weights generally for: {s_layer.name} -> {t_layer.name}", output_stream=sys.stderr)
                        except Exception as e_gen_copy:
                            tf.print(f"    WARN trying general weight copy for {s_layer.name} (Type: {type(s_layer).__name__}): {e_gen_copy}", output_stream=sys.stderr)
                            # Not necessarily a critical error if some layers don't need copying or fail gracefully
                            # successful_copy = False # Uncomment if any failure here should be critical

            if successful_copy:
                tf.print(f"  DirectCopy: Explicit component-wise weight copy attempt finished for MT Epoch {epoch + 1}.", output_stream=sys.stderr)
            else:
                tf.print(f"  CRITICAL DirectCopy: Explicit component-wise weight copy FAILED for MT Epoch {epoch + 1}. Teacher may not be a student copy.", output_stream=sys.stderr)

class ConsistencyWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, consistency_weight_var, max_weight, rampup_epochs_duration, delay_rampup_epochs=0):
        super().__init__()
        self.consistency_weight_var = consistency_weight_var
        self.max_weight = max_weight
        self.rampup_epochs_duration = rampup_epochs_duration
        self.delay_rampup_epochs = delay_rampup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        current_weight = 0.0
        if epoch >= self.delay_rampup_epochs:
            effective_epoch = epoch - self.delay_rampup_epochs
            if self.rampup_epochs_duration == 0:
                current_weight = self.max_weight
            elif effective_epoch < self.rampup_epochs_duration:
                current_weight = self.max_weight * (float(effective_epoch + 1) / float(self.rampup_epochs_duration))
            else:
                current_weight = self.max_weight
        self.consistency_weight_var.assign(current_weight)
        if epoch % 10 == 0 or epoch == 0:
             tf.print(f"Epoch {epoch+1} (MT Phase): Cons. Weight set to {current_weight:.4f} (delay: {self.delay_rampup_epochs}, ramp_dur: {self.rampup_epochs_duration})")

def prepare_data_paths(data_dir_str: str, num_labeled_patients: int, num_validation_patients: int, seed: int = 42):
    # ... (Keep your latest working version of this function) ...
    data_dir = Path(data_dir_str)
    all_patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pancreas_")])
    if not all_patient_dirs: raise FileNotFoundError(f"No patient dirs in {data_dir_str}")
    random.seed(seed); random.shuffle(all_patient_dirs)
    if len(all_patient_dirs) < num_labeled_patients + num_validation_patients:
        raise ValueError(f"Not enough patient data ({len(all_patient_dirs)}) for splits.")
    val_dirs = all_patient_dirs[:num_validation_patients]; rem_dirs = all_patient_dirs[num_validation_patients:]
    lab_dirs = rem_dirs[:num_labeled_patients]; unlab_dirs = rem_dirs[num_labeled_patients:]
    tf.print(f"Data Split: Total={len(all_patient_dirs)}, Labeled={len(lab_dirs)}, Unlabeled={len(unlab_dirs)}, Val={len(val_dirs)}")
    def get_files(p_dirs, is_lab=True):
        imgs, lbls = [], ([] if is_lab else None)
        for p_dir in p_dirs:
            img_f, mask_f = p_dir/"image.npy", p_dir/"mask.npy"
            if img_f.exists() and (not is_lab or mask_f.exists()): imgs.append(str(img_f)); (is_lab and lbls.append(str(mask_f)))
            else: tf.print(f"Warn: Missing files in {p_dir}", sys.stderr)
        return (imgs, lbls) if is_lab else imgs
    li,ll=get_files(lab_dirs,True); ui=get_files(unlab_dirs,False); vi,vl=get_files(val_dirs,True)
    if not li: tf.print("CRIT WARN: No Labeled images.", sys.stderr)
    if not ui and len(unlab_dirs)>0: tf.print("WARN: Unlabeled dirs exist, but no images found.", sys.stderr)
    if not vi: tf.print("CRIT WARN: No Validation images.", sys.stderr)
    return {'labeled':{'images':li,'labels':ll}, 'unlabeled':{'images':ui}, 'validation':{'images':vi,'labels':vl}}

def plot_training_summary(history_object, csv_log_path: Path, plot_save_dir: Path, plot_title_prefix: str):
    # ... (Keep your latest enhanced plot_training_summary function) ...
    plot_save_dir.mkdir(parents=True, exist_ok=True); data_source = None; history_data = {}
    if history_object and hasattr(history_object, 'history') and history_object.history:
        history_data = history_object.history; data_source = "Keras History"
    elif csv_log_path.exists():
        try:
            df = pd.read_csv(csv_log_path)
            if not df.empty:
                history_data = {col:df[col].tolist() for col in df.columns}; history_data['epochs_list']=df['epoch'].tolist() if 'epoch' in df.columns else list(range(1,len(df)+1)); data_source="CSV"
        except Exception as e: tf.print(f"Plot Error CSV: {e}", sys.stderr)
    if not history_data: tf.print(f"Plot Error: No data.", sys.stderr); return
    epochs=history_data.get('epochs_list')
    if not epochs:
        for k in ['loss','student_dice','val_loss','val_student_dice','pretrain_student_dice']:
            if k in history_data and history_data[k]: epochs=list(range(1,len(history_data[k])+1)); break
    if not epochs: tf.print("Plot Error: No epochs.", sys.stderr); return
    fig,axs=plt.subplots(2,1,figsize=(14,10),sharex=True); ax1,ax2=axs[0],axs[1]
    if 'loss' in history_data: ax1.plot(epochs,history_data['loss'],label='Total Train Loss',c='blue')
    if 'supervised_loss' in history_data: ax1.plot(epochs,history_data['supervised_loss'],label='Sup. Loss (Train)',c='orange',ls='--')
    if 'consistency_loss' in history_data: ax1.plot(epochs,history_data['consistency_loss'],label='Cons. Loss (Train)',c='green',ls=':')
    if 'val_loss' in history_data: ax1.plot(epochs,history_data['val_loss'],label='Total Val Loss',c='red')
    ax1.set_title('Losses'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    tdk='student_dice' if 'student_dice' in history_data else 'pretrain_student_dice'
    vdk='val_student_dice' if 'val_student_dice' in history_data else 'val_pretrain_student_dice'
    if tdk in history_data: ax2.plot(epochs,history_data[tdk],label=f'{tdk.replace("_"," ").title()}(Train)',c='dodgerblue')
    if vdk in history_data: ax2.plot(epochs,history_data[vdk],label=f'{vdk.replace("_"," ").title()}(Val)',c='deepskyblue')
    if 'val_teacher_dice' in history_data: ax2.plot(epochs,history_data['val_teacher_dice'],label='Teacher Dice (Val)',c='lightcoral',ls='--')
    ax2.set_title('Dice Scores'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice'); ax2.set_ylim(0,1.05); ax2.legend(); ax2.grid(True)
    plt.suptitle(plot_title_prefix,fontsize=16); plt.tight_layout(rect=[0,0,1,0.95])
    fname=plot_save_dir/f"{plot_title_prefix.replace(' ','_').replace('(','').replace(')','').replace(':','').lower()}_curves.png"
    try: plt.savefig(fname); tf.print(f"Saved plot: {fname}")
    except Exception as e: tf.print(f"Plot Error Save: {e}", sys.stderr)
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
    if not hasattr(model_data_config, 'n_filters'): model_data_config.n_filters = 32

    tf.print("Preparing data paths..."); data_paths = prepare_data_paths(args.data_dir, args.num_labeled, args.num_validation, args.seed)
    tf.print("Initializing DataPipeline..."); data_pipeline = DataPipeline(config=model_data_config)

    pretrain_student_model = None
    if args.student_pretrain_epochs > 0:
        tf.print(f"--- Student Pre-training Phase ({args.student_pretrain_epochs} epochs) ---")
        pretrain_student_model = PancreasSeg(model_data_config)
        _ = pretrain_student_model(tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels))) # Build
        pretrain_student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                       loss=DiceBCELoss(), metrics=[DiceCoefficient(name='pretrain_student_dice')])
        if not data_paths['labeled']['images']: sys.exit("CRIT: No labeled data for pre-train.")
        pre_lab_ds = data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], args.batch_size, True).prefetch(AUTOTUNE)
        if not data_paths['validation']['images']: sys.exit("CRIT: No validation data for pre-train.")
        pre_val_ds = data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], args.batch_size).prefetch(AUTOTUNE)
        if tf.data.experimental.cardinality(pre_lab_ds)==0: sys.exit("CRIT: Labeled pre-train dataset empty.")
            # --- FOR DEBUGGING NANs: Limit pre-train steps ---
        pretrain_steps_per_epoch_debug = None # Set to None for normal run
        if tf.config.functions_run_eagerly(): # If eager mode is ON (our debug mode)
            pretrain_steps_per_epoch_debug = 2 # Run only 2 steps for pre-training epoch
            tf.print(f"DEBUG: Eager mode ON, limiting pre-train steps_per_epoch to {pretrain_steps_per_epoch_debug}")
    # --- END DEBUGGING NANs ---
        pretrain_csv_path = exp_config.results_dir / "pretrain_log.csv"
        # Pre-train checkpoint uses TF format, saved into a 'pretrain' subdirectory
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
        # Pre-training phase fit call:
        num_labeled_slices_pretrain = len(data_paths['labeled']['images'])
        if num_labeled_slices_pretrain == 0: sys.exit("CRIT: No labeled slices for pre-train steps_per_epoch.")
        # For debugging, make pre-train very short, e.g., 1 epoch, 2 steps
        actual_pretrain_epochs = 1 # Override args.student_pretrain_epochs for this debug
        actual_pretrain_steps = 2    # Override calculated steps for this debug
        tf.print(f"DEBUG: Pre-train using epochs={actual_pretrain_epochs}, steps_per_epoch={actual_pretrain_steps}")
        steps_per_epoch_pretrain = (num_labeled_slices_pretrain + args.batch_size - 1) // args.batch_size
        tf.print(f"INFO: For Pre-train phase, using steps_per_epoch = {steps_per_epoch_pretrain}")        


        pretrain_history = pretrain_student_model.fit(
            pre_lab_ds, 
            epochs=args.student_pretrain_epochs, 
            validation_data=pre_val_ds, 
            verbose=args.verbose, 
            callbacks=pretrain_callbacks)
        tf.print("--- Student Pre-training Phase Completed ---")
        plot_training_summary(pretrain_history, pretrain_csv_path, plots_dir, f"Student Pre-Train ({args.num_labeled}L)")
        # pretrain_student_model now holds the best weights due to restore_best_weights=True

    tf.print("Instantiating student and teacher models for Mean Teacher phase...")
    student_model_mt = PancreasSeg(model_data_config)
    teacher_model_mt = PancreasSeg(model_data_config)
    # Build models before setting weights
    dummy_input_build = tf.zeros((1, args.img_size, args.img_size, model_data_config.num_channels))
    _ = student_model_mt(dummy_input_build); _ = teacher_model_mt(dummy_input_build)

    if pretrain_student_model:
        student_model_mt.set_weights(pretrain_student_model.get_weights())
        tf.print("MT Student model weights initialized from best pre-trained student.")
        del pretrain_student_model # Free memory
    else:
        tf.print("MT Student model weights initialized randomly (no pre-training).")
    
    teacher_model_mt.set_weights(student_model_mt.get_weights())
    tf.print("MT Teacher model weights initialized from MT student model.")

    mean_teacher_trainer = MeanTeacherTrainer(
        student_model=student_model_mt, 
        teacher_model=teacher_model_mt, 
        ema_decay=args.ema_decay, # This is the base_ema_decay for MT phase
        sharpening_temperature=args.sharpening_temperature
        # teacher_direct_copy_epochs is handled by callback, not trainer's __init__
    )
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
    mt_checkpoint_filepath_tf = str(mt_checkpoint_dir / "best_mt_student_epoch_{epoch:02d}_val_dice_{val_student_dice:.4f}") 

    callbacks_mt = [
        MTPhaseEpochUpdater(),
        TeacherDirectCopyCallback(student_model_mt, args.teacher_direct_copy_epochs),
        ConsistencyWeightScheduler(
            mean_teacher_trainer.consistency_weight, args.consistency_max, args.consistency_rampup, 
            delay_rampup_epochs=args.teacher_direct_copy_epochs if args.delay_consistency_rampup_by_copy_phase else 0),
        tf.keras.callbacks.ModelCheckpoint(filepath=mt_checkpoint_filepath_tf, save_weights_only=True, monitor='val_student_dice', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(exp_config.results_dir / "logs_mt"), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(str(exp_config.results_dir / "mean_teacher_training_log.csv")),
        tf.keras.callbacks.EarlyStopping(monitor='val_student_dice', patience=args.early_stopping_patience, mode='max', verbose=1, restore_best_weights=True)
    ]

    tf.print(f"--- Starting Mean Teacher Training Phase ({args.num_epochs} epochs) ---")
    start_time_mt = time.time()
    num_labeled_slices_mt = len(data_paths['labeled']['images'])
    if num_labeled_slices_mt == 0: sys.exit("CRIT: No labeled slices for MT phase steps_per_epoch calculation.")
    steps_per_epoch_mt = (num_labeled_slices_mt + args.batch_size - 1) // args.batch_size # Ceiling division
    tf.print(f"INFO: Using estimated steps_per_epoch_mt = {steps_per_epoch_mt} for MT phase based on {num_labeled_slices_mt} labeled slices.")
# Let's aim for a limited number of MT epochs for this specific debug.
    actual_mt_epochs = 5 # For this debug run, override args.num_epochs
    tf.print(f"DEBUG: MT phase running for {actual_mt_epochs} epochs with {steps_per_epoch_mt} steps/epoch.")    
    history_mt = mean_teacher_trainer.fit(
        train_ds_mt, 
        epochs=args.num_epochs,
        validation_data=val_ds_mt, 
        callbacks=callbacks_mt, 
        verbose=args.verbose)
    
    approx_train_time = "N/A"
    if hasattr(history_mt, 'epoch') and history_mt.epoch: # history_mt.epoch is a list of epoch numbers
        # This time calculation is very rough if training was interrupted / early stopped.
        # For a more accurate time, rely on the wall clock time printed by the script.
        if len(history_mt.epoch) > 0 :
             approx_train_time = f"{(time.time() - (start_time_mt if 'start_time_mt' in locals() else time.time() - len(history_mt.epoch)*100 ) )/60:.2f}" # Rough estimate

    tf.print(f"Mean Teacher training phase completed. Approx train time: {approx_train_time} minutes.")
    plot_training_summary(history_mt, exp_config.results_dir / "mean_teacher_training_log.csv", plots_dir, f"Mean Teacher ({args.num_labeled}L)")
    
    final_model_path_tf = exp_config.results_dir / "final_mt_student_model" 
    student_model_mt.save_weights(str(final_model_path_tf))
    tf.print(f"Final MT student model weights (TF format) saved to {final_model_path_tf}")
    tf.print(f"All results saved in: {exp_config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mean Teacher (MT) for pancreas segmentation.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiment_results_mt_final_v2")
    parser.add_argument("--experiment_name", type=str, default=f"MT_FinalRun_v2_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_labeled", type=int, default=30)
    parser.add_argument("--num_validation", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    
    parser.add_argument("--student_pretrain_epochs", type=int, default=15, help="Epochs for supervised pre-training of student.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for models.")
    
    parser.add_argument("--num_epochs", type=int, default=150, help="Epochs for MT training phase.") # Increased MT epochs
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    parser.add_argument("--teacher_direct_copy_epochs", type=int, default=10, 
                        help="Initial MT epochs: Teacher IS student (direct copy by callback at epoch start).")
    parser.add_argument("--ema_decay", type=float, default=0.999, 
                        help="EMA decay for teacher's per-batch updates (used by MeanTeacherTrainer after direct copy phase).")
    
    parser.add_argument("--sharpening_temperature", type=float, default=0.25, 
                        help="Temperature for teacher prediction sharpening (lower is stronger).")
    parser.add_argument("--consistency_max", type=float, default=20.0)
    parser.add_argument("--consistency_rampup", type=int, default=40)
    parser.add_argument("--delay_consistency_rampup_by_copy_phase", action='store_true', 
                        help="If set (default: False), delay consistency rampup until after teacher direct copy phase.")
    
    parser.add_argument("--early_stopping_patience", type=int, default=30) 
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    main(args)