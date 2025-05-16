#!/bin/bash
# SLURM batch job script for MixMatch

#SBATCH -p gpu
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH -n 8                        # Request 8 CPU cores
#SBATCH --mem=32G                   # Request 32GB RAM
#SBATCH --time=00:10:00             # Time limit (e.g., 12 hours)
#SBATCH -o slurm_logs/MixMatch-%A_%a-%j.out  # Standard output file (job array compatible)
#SBATCH -e slurm_logs/MixMatch-%A_%a-%j.err  # Standard error file (job array compatible)
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --job-name=MixMatchPancreas # Job name in SLURM

# --- Configuration ---
# HPC specific paths
# Ensure this points to your *preprocessed_v2* directory
PREPROCESSED_DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2" 
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14" # Your project directory
PYTHON_SCRIPT_NAME="run_mixmatch.py" # The new Python script
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"

# Base output directory for this series of experiments
OUTPUT_DIR_ROOT="$WORK_DIR/MixMatch_Experiments_L30_40_50" 

# Experiment specific parameters (example for L=30)
# For job array, NUM_LABELED can be set based on SLURM_ARRAY_TASK_ID
# declare -a NUM_LABELED_OPTIONS=(30 40 50)
# NUM_LABELED=${NUM_LABELED_OPTIONS[$SLURM_ARRAY_TASK_ID]}
NUM_LABELED=30 # Start with 30

# Training parameters
NUM_EPOCHS=150                     # Total epochs for MixMatch phase
BATCH_SIZE=4                       # Keep small for stability with MixUp
LEARNING_RATE=1e-4
IMG_SIZE=256                       # Match preprocessed image size
NUM_VALIDATION=15                  # Number of validation patients
EARLY_STOPPING_PATIENCE=30         # Patience for early stopping

# MixMatch Hyperparameters
MIXMATCH_T=0.5                     # Sharpening temperature
MIXMATCH_K=2                       # Augmentations for pseudo-labeling
MIXMATCH_ALPHA=0.75                # MixUp Beta distribution alpha
CONSISTENCY_MAX_WEIGHT=10.0        # Max consistency loss weight (lambda_u)
# Rampup in *steps*. If avg 100 steps/epoch, 50 epochs = 5000 steps.
CONSISTENCY_RAMPUP_STEPS=5000      

# Teacher EMA Hyperparameters
INITIAL_TEACHER_EMA_DECAY=0.95
FINAL_TEACHER_EMA_DECAY=0.999      # Target EMA decay
# Rampup in *steps*. If avg 100 steps/epoch, 20 epochs = 2000 steps.
TEACHER_EMA_WARMUP_STEPS=2000

SEED=42
VERBOSE_LEVEL=1 # Keras verbosity (1 for progress bar, 2 for one line per epoch)
# STEPS_PER_EPOCH: Set if you want to fix it, otherwise estimated.
# For ~30 labeled patients * ~30 slices/patient / batch_size 4 = ~225 steps.
# Let's set it to a reasonable number like 200 if your estimation is tricky.
STEPS_PER_EPOCH=200


# Construct a descriptive experiment name
EXPERIMENT_NAME="MixMatch_L${NUM_LABELED}_Epochs${NUM_EPOCHS}_BS${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_ROOT/$EXPERIMENT_NAME" # This is the base output passed to python script

# --- SLURM Preamble ---
echo "========================================================="
echo "MixMatch Pancreas Segmentation - $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID ; Array Task ID: $SLURM_ARRAY_TASK_ID (if applicable)"
echo "Running on node: $HOSTNAME ; GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Work Dir: $WORK_DIR ; Python Script: $PYTHON_SCRIPT_PATH"
echo "Preprocessed Data Dir: $PREPROCESSED_DATA_DIR"
echo "Output Dir Root: $OUTPUT_DIR_ROOT ; Current Experiment Output (base): $CURRENT_OUTPUT_DIR"
echo "--- Experiment Parameters ---"
echo "Labeled Patients: $NUM_LABELED"
echo "Epochs: $NUM_EPOCHS ; Steps per Epoch: $STEPS_PER_EPOCH"
echo "Batch Size: $BATCH_SIZE ; Learning Rate: $LEARNING_RATE"
echo "MixMatch T: $MIXMATCH_T ; K: $MIXMATCH_K ; Alpha: $MIXMATCH_ALPHA"
echo "Consistency Max Weight: $CONSISTENCY_MAX_WEIGHT ; Rampup Steps: $CONSISTENCY_RAMPUP_STEPS"
echo "Initial EMA Decay: $INITIAL_TEACHER_EMA_DECAY ; Final EMA Decay: $FINAL_TEACHER_EMA_DECAY ; EMA Warmup Steps: $TEACHER_EMA_WARMUP_STEPS"
echo "========================================================="

# --- Environment Setup ---
mkdir -p "$OUTPUT_DIR_ROOT" # Python script will make $CURRENT_OUTPUT_DIR inside its logic based on args
mkdir -p "$WORK_DIR/slurm_logs" 
if [ ! -d "$PREPROCESSED_DATA_DIR" ]; then echo "ERROR: Preprocessed data dir '$PREPROCESSED_DATA_DIR' not found!"; exit 1; fi
echo "NVIDIA GPU status:"; nvidia-smi
export TF_FORCE_GPU_ALLOW_GROWTH=true # Already in python script, but good practice
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TF info logs

# --- Python Environment ---
PYTHON_CMD="python3" # Or specify path to your venv python
if ! command -v $PYTHON_CMD &> /dev/null; then echo "ERROR: $PYTHON_CMD not found."; exit 1; fi
echo "Using Python: $($PYTHON_CMD --version)"
echo "TensorFlow GPU check (from shell before script execution):"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available (TF): ', len(tf.config.list_physical_devices('GPU')))"


# --- Execution ---
cd "$WORK_DIR" || { echo "ERROR: Failed to cd to $WORK_DIR"; exit 1; }
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "ERROR: Script '$PYTHON_SCRIPT_PATH' not found."; exit 1; fi

echo "Executing Python script: $PYTHON_SCRIPT_NAME"
$PYTHON_CMD "$PYTHON_SCRIPT_PATH" \
    --data_dir "$PREPROCESSED_DATA_DIR" \
    --output_dir "$OUTPUT_DIR_ROOT" \
    --experiment_name "$EXPERIMENT_NAME" \
    --img_size "$IMG_SIZE" \
    --num_labeled "$NUM_LABELED" \
    --num_validation "$NUM_VALIDATION" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --learning_rate "$LEARNING_RATE" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --seed "$SEED" \
    --mixmatch_T "$MIXMATCH_T" \
    --mixmatch_K "$MIXMATCH_K" \
    --mixmatch_alpha "$MIXMATCH_ALPHA" \
    --consistency_max "$CONSISTENCY_MAX_WEIGHT" \
    --consistency_rampup_steps "$CONSISTENCY_RAMPUP_STEPS" \
    --initial_teacher_ema_decay "$INITIAL_TEACHER_EMA_DECAY" \
    --ema_decay "$FINAL_TEACHER_EMA_DECAY" \
    --teacher_ema_warmup_steps "$TEACHER_EMA_WARMUP_STEPS" \
    --verbose "$VERBOSE_LEVEL"

# --- Post-Execution ---
echo "========================================================="
echo "Script finished at $(date)."
# The Python script now handles its own output directory structure under --output_dir / --experiment_name
echo "Check output/logs in $OUTPUT_DIR_ROOT/$EXPERIMENT_NAME"
echo "SLURM error file: $WORK_DIR/slurm_logs/MixMatch-Job-$SLURM_JOB_ID.err (or similar for array)"
echo "========================================================="
exit 0