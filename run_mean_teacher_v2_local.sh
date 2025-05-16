#!/bin/bash
# SLURM batch job script for Mean Teacher v2 (with Student Pre-training & Phased EMA)

#SBATCH -p gpu
#SBATCH --gres=gpu:1                # Request 1 GPU (can start with 1)
#SBATCH -n 8                        # Request 8 CPU cores
#SBATCH --mem=32G                   # Request 32GB RAM
#SBATCH --time=00:10:00             # Time limit (e.g., 6 hours, for pre-training + MT phase)
#SBATCH -o slurm_logs/MT_Pre_PhaseEMA-%j.out  # Standard output file (in a slurm_logs subdir)
#SBATCH -e slurm_logs/MT_Pre_PhaseEMA-%j.err  # Standard error file (in a slurm_logs subdir)
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --job-name=MT_PrePhaseEMA   # Job name in SLURM

# --- Configuration ---
# HPC specific paths
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py"
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"

# Base output directory for this series of experiments
OUTPUT_DIR_ROOT="$WORK_DIR/MT_Experiments_Pretrain_PhasedEMA" 

# Experiment specific parameters
NUM_LABELED=30                  # Number of labeled patient volumes for training (passed as --num_labeled)
STUDENT_PRETRAIN_EPOCHS=15      # Epochs for supervised student pre-training
MT_PHASE_EPOCHS=100             # Epochs for the Mean Teacher phase itself

# Teacher EMA configuration (for Phased EMA in MT phase)
TEACHER_EMA_WARMUP_EPOCHS=10    # Initial MT epochs with faster EMA
INITIAL_TEACHER_EMA_DECAY=0.95  # EMA decay during teacher warmup
BASE_EMA_DECAY=0.999            # EMA decay after teacher warmup (passed as --ema_decay)

# Construct a descriptive experiment name
EXPERIMENT_NAME="MT_L${NUM_LABELED}_Pre${STUDENT_PRETRAIN_EPOCHS}_TWarm${TEACHER_EMA_WARMUP_EPOCHS}_MT${MT_PHASE_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_ROOT/$EXPERIMENT_NAME"

# Other training parameters
IMG_SIZE=256
NUM_VALIDATION=10
BATCH_SIZE=4
LEARNING_RATE=1e-4
CONSISTENCY_MAX=10.0             # <<< CORRECTED: Set to a non-zero value
CONSISTENCY_RAMPUP=30            # Epochs for consistency weight ramp-up (during MT_PHASE_EPOCHS)
EARLY_STOPPING_PATIENCE=25       # Increased patience slightly for MT phase
SEED=42
VERBOSE=1

# --- SLURM Preamble ---
echo "========================================================="
echo "Mean Teacher (Student Pre-training & Phased EMA) - $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID ; Running on node: $HOSTNAME"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Work Dir: $WORK_DIR ; Python Script: $PYTHON_SCRIPT_PATH"
echo "Data Dir: $DATA_DIR ; Output Dir: $CURRENT_OUTPUT_DIR"
echo "--- Experiment Parameters ---"
echo "Labeled Patients: $NUM_LABELED"
echo "Student Pre-train Epochs: $STUDENT_PRETRAIN_EPOCHS"
echo "MT Phase Epochs: $MT_PHASE_EPOCHS"
echo "Teacher EMA Warmup Epochs (MT Phase): $TEACHER_EMA_WARMUP_EPOCHS"
echo "Initial Teacher EMA Decay (MT Phase Warmup): $INITIAL_TEACHER_EMA_DECAY"
echo "Base EMA Decay (MT Phase Post-Warmup): $BASE_EMA_DECAY"
echo "Consistency Max Weight: $CONSISTENCY_MAX ; Rampup Epochs: $CONSISTENCY_RAMPUP"
echo "Batch Size: $BATCH_SIZE ; Learning Rate: $LEARNING_RATE"
echo "========================================================="

# --- Environment Setup ---
mkdir -p "$CURRENT_OUTPUT_DIR"
mkdir -p "$WORK_DIR/slurm_logs" 
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: Data dir '$DATA_DIR' not found."; exit 1; fi
echo "NVIDIA GPU status:"; nvidia-smi
export TF_FORCE_GPU_ALLOW_GROWTH=true; export TF_CPP_MIN_LOG_LEVEL=1

# --- Python Environment ---
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then echo "ERROR: python3 not found."; exit 1; fi
echo "Using Python: $($PYTHON_CMD --version)"

# --- Execution ---
cd "$WORK_DIR" || { echo "ERROR: Failed to cd to $WORK_DIR"; exit 1; }
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "ERROR: Script '$PYTHON_SCRIPT_PATH' not found."; exit 1; fi
echo "TF GPU check:"; $PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"

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
    --teacher_ema_warmup_epochs "$TEACHER_EMA_WARMUP_EPOCHS" \
    --initial_teacher_ema_decay "$INITIAL_TEACHER_EMA_DECAY" \
    --ema_decay "$BASE_EMA_DECAY" \
    --learning_rate "$LEARNING_RATE" \
    --consistency_max "$CONSISTENCY_MAX" \
    --consistency_rampup "$CONSISTENCY_RAMPUP" \
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