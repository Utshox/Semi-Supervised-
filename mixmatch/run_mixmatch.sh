#!/bin/bash
# SLURM batch job script for MixMatch semi-supervised learning - Performance Run

#SBATCH -p gpu
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH -n 16                        # Request 8 CPU cores
#SBATCH --mem=60G                   # Request 60GB RAM (adjust to 64G if that's your node's unit)
#SBATCH --time=00:15:00             # Time limit (e.g., 12 hours for a substantial run)
#SBATCH -o slurm_logs/MixMatch_Perf-%j.out
#SBATCH -e slurm_logs/MixMatch_Perf-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=MM_PerfRun

# --- Configuration ---
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"
PYTHON_SCRIPT_NAME="run_mixmatch.py" 
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_ROOT="$WORK_DIR/MixMatch_Performance_Runs" 

# --- Experiment Parameters for MixMatch ---
NUM_LABELED_MM=30
BATCH_SIZE_MM=8 # Try with 8, can increase to 16 if memory allows with 2 GPUs
NUM_EPOCHS_MM=150 
LEARNING_RATE_MM=0.002
DROPOUT_RATE_MM=0.1

# MixMatch Hyperparameters
MIXMATCH_T=0.5
MIXMATCH_K=2
MIXMATCH_ALPHA=0.75
MIXMATCH_CONSISTENCY_MAX=75.0 
# Calculate MIXMATCH_RAMPUP_STEPS:
# If 1 epoch = (NUM_LABELED_MM / BATCH_SIZE_MM) steps ~ 30/8 ~ 4 steps
# For ramp-up over ~75 epochs: 75 * 4 = 300 steps.
# For ramp-up over ~100 epochs: 100 * 4 = 400 steps.
# The original paper often uses a large number of total iterations.
# Let's aim for ramp-up over a significant portion of initial training.
# If total steps = 150 epochs * 4 steps/epoch = 600 steps.
# A ramp-up over 300-400 steps seems reasonable.
MIXMATCH_RAMPUP_STEPS=400

# General params
NUM_VALIDATION_MM=10
IMG_SIZE_MM=256
EARLY_STOPPING_PATIENCE_MM=30 # Patience for early stopping
SEED_MM=42
VERBOSE_MM=1

# Dynamic experiment name is constructed inside the Python script
# The python script will use args.experiment_name as a prefix and add more details.
# We pass a base prefix from here.
EXPERIMENT_NAME_PREFIX="MM_L${NUM_LABELED_MM}_B${BATCH_SIZE_MM}"

# --- SLURM Preamble & Environment Setup ---
echo "========================================================="
echo "Running MixMatch Performance Run (HPC) - $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID ; Running on node: $HOSTNAME"
echo "GPU(s): $CUDA_VISIBLE_DEVICES ; Expected 2 GPUs from SBATCH"
echo "Work Dir: $WORK_DIR ; Python Script: $PYTHON_SCRIPT_PATH"
echo "Data Dir: $DATA_DIR ; Output Root: $OUTPUT_DIR_ROOT"
echo "--- MixMatch Parameters ---"
echo "Labeled: $NUM_LABELED_MM, Epochs: $NUM_EPOCHS_MM, BatchSize: $BATCH_SIZE_MM, LR: $LEARNING_RATE_MM"
echo "K: $MIXMATCH_K, T: $MIXMATCH_T, Alpha: $MIXMATCH_ALPHA, Lambda_U_Max: $MIXMATCH_CONSISTENCY_MAX, RampupSteps: $MIXMATCH_RAMPUP_STEPS"
echo "========================================================="
mkdir -p "$OUTPUT_DIR_ROOT" # Python script handles specific experiment subfolder
mkdir -p "$WORK_DIR/slurm_logs" 
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: Data dir '$DATA_DIR' not found."; exit 1; fi
echo "NVIDIA GPU status:"; nvidia-smi

# --- TensorFlow & XLA Configuration ---
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TF info messages, show warnings and errors

# Attempt to disable XLA JIT compilation via environment variables
# This is to prevent conflicts with debugging ops if they were accidentally left on,
# or to see if XLA was contributing to any instability.
# For a performance run, you might experiment with XLA ON if no NaNs occur.
echo "Attempting to disable XLA JIT via TF_XLA_FLAGS..."
export TF_XLA_FLAGS="--tf_xla_auto_jit=0" 
# An alternative, more forceful way if the above doesn't work:
# export TF_XLA_FLAGS="--tf_xla_cpu_global_jit=false --tf_xla_gpu_global_jit=false --tf_xla_enable_xla_devices=false"
# For now, let's try the simpler --tf_xla_auto_jit=0

# --- Python Environment ---
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then echo "ERROR: python3 not found."; exit 1; fi
echo "Using Python: $($PYTHON_CMD --version)"

# --- Execution ---
cd "$WORK_DIR" || { echo "ERROR: Failed to cd to $WORK_DIR"; exit 1; }
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "ERROR: Script '$PYTHON_SCRIPT_PATH' not found."; exit 1; fi
echo "TF GPU check (from Python script's perspective):"; $PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available to TF:', len(tf.config.list_physical_devices('GPU')))"

echo "Executing Python script: $PYTHON_SCRIPT_NAME"
$PYTHON_CMD "$PYTHON_SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_ROOT" \
    --experiment_name "$EXPERIMENT_NAME_PREFIX" \
    --img_size "$IMG_SIZE_MM" \
    --num_labeled "$NUM_LABELED_MM" \
    --num_validation "$NUM_VALIDATION_MM" \
    --batch_size "$BATCH_SIZE_MM" \
    --num_epochs "$NUM_EPOCHS_MM" \
    --learning_rate "$LEARNING_RATE_MM" \
    --dropout_rate "$DROPOUT_RATE_MM" \
    --mixmatch_T "$MIXMATCH_T" \
    --mixmatch_K "$MIXMATCH_K" \
    --mixmatch_alpha "$MIXMATCH_ALPHA" \
    --mixmatch_consistency_max "$MIXMATCH_CONSISTENCY_MAX" \
    --mixmatch_rampup_steps "$MIXMATCH_RAMPUP_STEPS" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE_MM" \
    --seed "$SEED_MM" \
    --verbose "$VERBOSE_MM"

# --- Post-Execution ---
echo "========================================================="
echo "MixMatch Script finished at $(date)."
echo "Check SLURM output/error files in $WORK_DIR/slurm_logs/"
echo "Main experiment results should be in a subdirectory of $OUTPUT_DIR_ROOT"
echo "========================================================="
exit 0