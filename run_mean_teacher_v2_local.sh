#!/bin/bash
# Modified SLURM batch job script for Mean Teacher v2 semi-supervised learning

#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00 # Adjusted time limit
#SBATCH -o meanteacher_v2-%j.out
#SBATCH -e meanteacher_v2-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=pancreas_mt_v2

# --- Configuration ---
# Use local directory for development and testing
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2" # Path for preprocessed data on HPC
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14" # Working directory on HPC
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py" # The main script
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_BASE="$WORK_DIR/mean_teacher_v2_results" # Base directory for experiment outputs

# Experiment specific parameters
EXPERIMENT_NAME="mt_v2_local_test_$(date +%Y%m%d_%H%M%S)"
IMG_SIZE=256
NUM_LABELED=5 # Reduced for testing
NUM_VALIDATION=2 # Reduced for testing
BATCH_SIZE=2
NUM_EPOCHS=10 # Reduced for testing
LEARNING_RATE=1e-4
EMA_DECAY=0.999
CONSISTENCY_MAX=10.0
CONSISTENCY_RAMPUP=30 # Epochs
EARLY_STOPPING_PATIENCE=5
SEED=42
VERBOSE=1 # 0=silent, 1=progress bar, 2=one line per epoch

# --- SLURM Preamble ---
echo "========================================================="
echo "Running Mean Teacher v2 Learning - $(date)"
echo "========================================================="
echo "Running on node: $HOSTNAME"
echo "Working directory: $(pwd)"
echo "Python script: $PYTHON_SCRIPT_PATH"
echo "Data directory: $DATA_DIR"
echo "Base Output directory: $OUTPUT_DIR_BASE"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "========================================================="

# --- Environment Setup ---
# Create output directory
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_BASE/$EXPERIMENT_NAME"
mkdir -p "$CURRENT_OUTPUT_DIR"
echo "Results will be saved in: $CURRENT_OUTPUT_DIR"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"
echo "Using data directory: $DATA_DIR"

# --- First, create sample data for testing ---
echo "Creating sample preprocessed data structure..."
python3 -c "
import os
import numpy as np
from pathlib import Path

# Create sample data directories
data_dir = Path('$DATA_DIR')
data_dir.mkdir(exist_ok=True, parents=True)

# Create 10 sample patient directories
for i in range(1, 11):
    patient_dir = data_dir / f'pancreas_{i:03d}'
    patient_dir.mkdir(exist_ok=True)
    
    # Create sample image (random noise)
    img = np.random.rand(256, 256, 1).astype(np.float32)
    np.save(patient_dir / 'image.npy', img)
    
    # Create sample mask (random binary mask)
    mask = (np.random.rand(256, 256, 1) > 0.85).astype(np.float32)
    np.save(patient_dir / 'mask.npy', mask)

print(f'Created sample data for 10 patients in {data_dir}')
"

# --- TensorFlow GPU environment settings ---
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TensorFlow informational messages

# --- Python Environment ---
# Find Python executable (python3 is preferred)
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python executable (python3 or python) not found."
    exit 1
fi
echo "Using Python: $($PYTHON_CMD --version)"

# --- Execution ---
# Change to the working directory
cd "$WORK_DIR" || { echo "ERROR: Failed to change directory to $WORK_DIR"; exit 1; }

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability check:"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Run the Mean Teacher v2 training script
echo "Running Mean Teacher v2 training script: $PYTHON_SCRIPT_NAME"
$PYTHON_CMD "$PYTHON_SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_BASE" \
    --experiment_name "$EXPERIMENT_NAME" \
    --img_size "$IMG_SIZE" \
    --num_labeled "$NUM_LABELED" \
    --num_validation "$NUM_VALIDATION" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --ema_decay "$EMA_DECAY" \
    --consistency_max "$CONSISTENCY_MAX" \
    --consistency_rampup "$CONSISTENCY_RAMPUP" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --seed "$SEED" \
    --verbose "$VERBOSE"

# --- Post-Execution ---
echo "========================================================="
echo "Mean Teacher v2 training script finished at $(date)."
echo "Output and logs are in $CURRENT_OUTPUT_DIR"
echo "========================================================="

exit 0
