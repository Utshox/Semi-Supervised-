#!/bin/bash
# SLURM batch job script for Mean Teacher v2 (Final Strategy Attempt)
#SBATCH -p gpu
#SBATCH --gres=gpu:2                # Request 1 GPU (Mean Teacher typically doesn't scale well to multi-GPU witho>
#SBATCH -n 8                        # Request 8 CPU cores (for data loading)
#SBATCH --mem=60G                   # Memory request (adjust if needed, 32G might be fine for 256x256)
#SBATCH --time=02:00:00             # Time limit (e.g., 12 hours for a full run)
#SBATCH -o slurm_logs/MT_FinalStrat-%j.out
#SBATCH -e slurm_logs/MT_FinalStrat-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=MT_FinalStrat

# --- Configuration ---
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py" # The one I just provided
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_ROOT="$WORK_DIR/MT_Results_FinalStrategy" 

# --- Experiment Parameters ---
NUM_LABELED=50                  # Number of labeled patient files/slices
STUDENT_PRETRAIN_EPOCHS=20      # Pre-train the student sufficiently
MT_PHASE_EPOCHS=50             # Total epochs for the Mean Teacher training part

# Teacher Update Strategy for MT Phase:
TEACHER_DIRECT_COPY_EPOCHS=10   # For first 10 MT epochs, teacher = student (via callback at epoch_begin)
BASE_EMA_DECAY=0.999            # EMA decay used by MeanTeacherTrainer._update_teacher_model per batch ALWAYS

# Consistency & Sharpening
CONSISTENCY_MAX=10            # A reasonable starting point for consistency weight
CONSISTENCY_RAMPUP=50           # Ramp up over 50 MT epochs
SHARPENING_TEMPERATURE=0.5     # Strong sharpening, good if pseudo-labels from teacher are somewhat okay
DELAY_CONSISTENCY_FLAG="--delay_consistency_rampup_by_copy_phase" # Delay consistency until after direct copy

# Other Parameters
IMG_SIZE=256
NUM_VALIDATION=30               # Increased validation set size for more stable metrics
BATCH_SIZE=8                    # Increased batch size, adjust if OOM with 1 GPU & 48GB
LEARNING_RATE=1e-4
MT_LEARNING_RATE=2e-5              # Standard learning rate
EARLY_STOPPING_PATIENCE=30      # Patience for MT phase early stopping
DROPOUT_RATE=0.2                # Moderate dropout
SEED=44
VERBOSE=1

# Construct Experiment Name
EXPERIMENT_NAME="MT_L${NUM_LABELED}_Pre${STUDENT_PRETRAIN_EPOCHS}_Copy${TEACHER_DIRECT_COPY_EPOCHS}_EMA${BASE_EMA>
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_ROOT/$EXPERIMENT_NAME"

# --- SLURM Preamble & Environment Setup ---
echo "========================================================="
echo "Mean Teacher (Final Strategy) - $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID ; Running on node: $HOSTNAME"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Work Dir: $WORK_DIR ; Python Script: $PYTHON_SCRIPT_PATH"
echo "Data Dir: $DATA_DIR ; Output Dir: $CURRENT_OUTPUT_DIR"
echo "--- Experiment Parameters ---"
echo "Labeled: $NUM_LABELED, Pretrain Epochs: $STUDENT_PRETRAIN_EPOCHS, MT Phase Epochs: $MT_PHASE_EPOCHS"
echo "Teacher Direct Copy Epochs (MT): $TEACHER_DIRECT_COPY_EPOCHS"
echo "Base EMA Decay (MT per-batch): $BASE_EMA_DECAY"
echo "Consistency: Max=$CONSISTENCY_MAX, Rampup=$CONSISTENCY_RAMPUP, Delay Flag Active: ${DELAY_CONSISTENCY_FLAG:>
echo "Sharpening Temp: $SHARPENING_TEMPERATURE, Dropout: $DROPOUT_RATE"
echo "Batch Size: $BATCH_SIZE, LR: $LEARNING_RATE, Validation Patients: $NUM_VALIDATION"
echo "========================================================="
mkdir -p "$CURRENT_OUTPUT_DIR"
mkdir -p "$WORK_DIR/slurm_logs" 
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: Data dir '$DATA_DIR' not found."; exit 1; fi
echo "NVIDIA GPU status:"; nvidia-smi
export TF_FORCE_GPU_ALLOW_GROWTH=true; export TF_CPP_MIN_LOG_LEVEL=1

# Ensure XLA is not causing issues with debug ops (if any were accidentally left on)
# For a performance run, you might remove this or set to "--tf_xla_auto_jit=2" for potential speedup
# IF AND ONLY IF no XLA-related errors occur. For now, being cautious:
export TF_XLA_FLAGS="--tf_xla_auto_jit=0" 
echo "TF_XLA_FLAGS set to: $TF_XLA_FLAGS"

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then echo "ERROR: python3 not found."; exit 1; fi
echo "Using Python: $($PYTHON_CMD --version)"
cd "$WORK_DIR" || { echo "ERROR: Failed to cd to $WORK_DIR"; exit 1; }
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "ERROR: Script '$PYTHON_SCRIPT_PATH' not found."; exit 1; fi
echo "TF GPU check (from Python script's perspective):"; $PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs>
# --- End Environment Setup ---

# --- Execution ---
echo "Executing Python script: $PYTHON_SCRIPT_NAME with arguments:"
echo "  --data_dir $DATA_DIR"
echo "  --output_dir $OUTPUT_DIR_ROOT"
echo "  --experiment_name $EXPERIMENT_NAME"
echo "  --num_labeled $NUM_LABELED"
echo "  --student_pretrain_epochs $STUDENT_PRETRAIN_EPOCHS"
echo "  --num_epochs $MT_PHASE_EPOCHS" # MT phase epochs
echo "  --teacher_direct_copy_epochs $TEACHER_DIRECT_COPY_EPOCHS"
echo "  --ema_decay $BASE_EMA_DECAY"
echo "  --sharpening_temperature $SHARPENING_TEMPERATURE"
echo "  --consistency_max $CONSISTENCY_MAX"
echo "  --consistency_rampup $CONSISTENCY_RAMPUP"
echo "  --dropout_rate $DROPOUT_RATE"
echo "  Delay Consistency Flag will be: ${DELAY_CONSISTENCY_FLAG}"
echo "  Other args: img_size=$IMG_SIZE, num_val=$NUM_VALIDATION, batch=$BATCH_SIZE, lr=$LEARNING_RATE, early_stop>

$PYTHON_CMD "$PYTHON_SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_ROOT" \
    --experiment_name "$EXPERIMENT_NAME" \
    --img_size "$IMG_SIZE" \
    --num_labeled "$NUM_LABELED" \
    --num_validation "$NUM_VALIDATION" \
    --batch_size "$BATCH_SIZE" \
    --student_pretrain_epochs "$STUDENT_PRETRAIN_EPOCHS" \
    --num_epochs "$MT_PHASE_EPOCHS" \
    --dropout_rate "$DROPOUT_RATE" \
    --learning_rate "$LEARNING_RATE" \
    --mt_learning_rate "$MT_LEARNING_RATE" \
    --teacher_direct_copy_epochs "$TEACHER_DIRECT_COPY_EPOCHS" \
    --ema_decay "$BASE_EMA_DECAY" \
    --sharpening_temperature "$SHARPENING_TEMPERATURE" \
    --consistency_max "$CONSISTENCY_MAX" \
    --consistency_rampup "$CONSISTENCY_RAMPUP" \
    ${DELAY_CONSISTENCY_FLAG} \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --seed "$SEED" \
    --verbose "$VERBOSE"

# --- Post-Execution ---
echo "========================================================="
echo "Script finished at $(date)."
echo "Output/logs in $CURRENT_OUTPUT_DIR"
echo "SLURM error file: $WORK_DIR/slurm_logs/MT_FinalStrat-$SLURM_JOB_ID.err"
echo "========================================================="
exit 0