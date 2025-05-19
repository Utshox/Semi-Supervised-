#!/bin/bash
# SLURM batch job script for Mean Teacher v2 
# Strategy: Student Pre-training, then MT phase with:
#           - Teacher is DIRECT COPY of Student for N initial MT epochs (via callback)
#           - Teacher uses standard slow EMA for all subsequent MT epochs (per batch)
#           - Teacher Prediction Sharpening for consistency loss

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=06:00:00             # Increased time for full run
#SBATCH -o slurm_logs/MT_SimpCopyEMA-%j.out
#SBATCH -e slurm_logs/MT_SimpCopyEMA-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=MT_SimpCopyEMA

# --- Configuration ---
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py"
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_ROOT="$WORK_DIR/MT_Experiments_SimpCopyEMA" # New output root for this strategy

# --- Experiment Parameters ---
NUM_LABELED=30
STUDENT_PRETRAIN_EPOCHS=15
MT_PHASE_EPOCHS=150 # Total epochs for the Mean Teacher training part

# Teacher Update Strategy for MT Phase:
TEACHER_DIRECT_COPY_EPOCHS=10 # For first 10 MT epochs, teacher = student (via callback at epoch_begin)
BASE_EMA_DECAY=0.999          # EMA decay used by MeanTeacherTrainer._update_teacher_model per batch ALWAYS

# Consistency & Sharpening
CONSISTENCY_MAX=20.0
CONSISTENCY_RAMPUP=40 # Ramp up over 40 MT epochs
SHARPENING_TEMPERATURE=0.25
# Flag to delay consistency ramp-up until after direct copy phase (optional)
DELAY_CONSISTENCY_FLAG="--delay_consistency_rampup_by_copy_phase" # To activate delay
# DELAY_CONSISTENCY_FLAG="" # To have consistency ramp from MT epoch 0

# Other Parameters
IMG_SIZE=256
NUM_VALIDATION=10
BATCH_SIZE=4
LEARNING_RATE=1e-4
EARLY_STOPPING_PATIENCE=30
DROPOUT_RATE=0.1
SEED=42
VERBOSE=1

# Construct Experiment Name
EXPERIMENT_NAME="MT_L${NUM_LABELED}_Pre${STUDENT_PRETRAIN_EPOCHS}_Copy${TEACHER_DIRECT_COPY_EPOCHS}_BaseEMA${BASE_EMA_DECAY}_ConsMax${CONSISTENCY_MAX}_SharpT${SHARPENING_TEMPERATURE}_DR${DROPOUT_RATE}_$(date +%Y%m%d_%H%M%S)"
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_ROOT/$EXPERIMENT_NAME"

# --- SLURM Preamble & Environment Setup (Keep as is from your last script) ---
echo "========================================================="
echo "Mean Teacher (Simplified: Direct Copy then Base EMA) - $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID ; Running on node: $HOSTNAME"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Work Dir: $WORK_DIR ; Python Script: $PYTHON_SCRIPT_PATH"
echo "Data Dir: $DATA_DIR ; Output Dir: $CURRENT_OUTPUT_DIR"
echo "--- Experiment Parameters ---"
echo "Labeled: $NUM_LABELED, Pretrain Epochs: $STUDENT_PRETRAIN_EPOCHS, MT Phase Epochs: $MT_PHASE_EPOCHS"
echo "Teacher Direct Copy Epochs (MT): $TEACHER_DIRECT_COPY_EPOCHS"
echo "Base EMA Decay (MT per-batch): $BASE_EMA_DECAY"
echo "Consistency: Max=$CONSISTENCY_MAX, Rampup=$CONSISTENCY_RAMPUP, Delay Flag Active: ${DELAY_CONSISTENCY_FLAG:-"No"}"
echo "Sharpening Temp: $SHARPENING_TEMPERATURE, Dropout: $DROPOUT_RATE"
echo "Batch Size: $BATCH_SIZE, LR: $LEARNING_RATE"
echo "========================================================="
mkdir -p "$CURRENT_OUTPUT_DIR"
mkdir -p "$WORK_DIR/slurm_logs" 
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: Data dir '$DATA_DIR' not found."; exit 1; fi
echo "NVIDIA GPU status:"; nvidia-smi
export TF_FORCE_GPU_ALLOW_GROWTH=true; export TF_CPP_MIN_LOG_LEVEL=1; export TF_XLA_FLAGS="--tf_xla_cpu_global_jit=false --tf_xla_gpu_global_jit=false"
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then echo "ERROR: python3 not found."; exit 1; fi
echo "Using Python: $($PYTHON_CMD --version)"
cd "$WORK_DIR" || { echo "ERROR: Failed to cd to $WORK_DIR"; exit 1; }
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "ERROR: Script '$PYTHON_SCRIPT_PATH' not found."; exit 1; fi
echo "TF GPU check:"; $PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"
# --- End Environment Setup ---

# --- Execution ---
echo "Executing Python script: $PYTHON_SCRIPT_NAME"
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
    \
    --teacher_direct_copy_epochs "$TEACHER_DIRECT_COPY_EPOCHS" \
    --ema_decay "$BASE_EMA_DECAY" \
    \
    --sharpening_temperature "$SHARPENING_TEMPERATURE" \
    --consistency_max "$CONSISTENCY_MAX" \
    --consistency_rampup "$CONSISTENCY_RAMPUP" \
    ${DELAY_CONSISTENCY_FLAG} \
    \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --seed "$SEED" \
    --verbose "$VERBOSE"


# --- Post-Execution ---
echo "========================================================="
echo "Script finished at $(date)."
echo "Output/logs in $CURRENT_OUTPUT_DIR"
echo "SLURM error file: $WORK_DIR/slurm_logs/MT_Pre_PhaseEMA-$SLURM_JOB_ID.err"
echo "========================================================="
exit 0