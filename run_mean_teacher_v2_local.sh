#!/bin/bash
# SLURM batch job script for Mean Teacher v2 semi-supervised learning

#SBATCH -p gpu
#SBATCH --gres=gpu:2                # Request 2 GPUs (adjust if needed, e.g., gpu:1 for single GPU)
#SBATCH -n 8                        # Request 8 CPU cores (good for data loading)
#SBATCH --mem=64G                   # Request 64GB RAM
#SBATCH --time=02:00:00             # Time limit (e.g., 2 hours, adjust as needed for more epochs)
#SBATCH -o meanteacher_v2_hpc-%j.out  # Standard output file
#SBATCH -e meanteacher_v2_hpc-%j.err  # Standard error file
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --job-name=pancreas_mt_v2_hpc # Job name in SLURM

# --- Configuration ---
# HPC specific paths
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2" # Root directory of preprocessed data
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"                # Directory containing Python scripts
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py"                     # The main Python script
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_BASE="$WORK_DIR/mean_teacher_v2_hpc_results"         # Base directory for HPC experiment outputs

# Experiment specific parameters (adjust for a full run)
EXPERIMENT_NAME="mt_v2_hpc_$(date +%Y%m%d_%H%M%S)"
IMG_SIZE=256
NUM_LABELED=30                   # Number of labeled patient volumes for training
NUM_VALIDATION=10                # Number of patient volumes for validation
BATCH_SIZE=4                     # Batch size (adjust based on GPU memory and dataset)
NUM_EPOCHS=100                   # Number of training epochs for a proper run
LEARNING_RATE=1e-4
EMA_DECAY=0.999
CONSISTENCY_MAX=0.0
CONSISTENCY_RAMPUP=30            # Epochs for consistency weight ramp-up
EARLY_STOPPING_PATIENCE=20
SEED=42
VERBOSE=1                        # Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)

# --- SLURM Preamble ---
echo "========================================================="
echo "Running Mean Teacher v2 Learning (HPC) - $(date)"
echo "========================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory (where script is run from): $(pwd)" # Will be $WORK_DIR after cd
echo "Python script: $PYTHON_SCRIPT_PATH"
echo "Data directory: $DATA_DIR"
echo "Base Output directory: $OUTPUT_DIR_BASE"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "========================================================="

# --- Environment Setup ---
# Create the specific output directory for this experiment run
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_BASE/$EXPERIMENT_NAME"
mkdir -p "$CURRENT_OUTPUT_DIR"
echo "Results will be saved in: $CURRENT_OUTPUT_DIR"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory '$DATA_DIR' not found. Please ensure your preprocessed data is there."
    exit 1
fi
echo "Using data from: $DATA_DIR"

# Check for available GPU with nvidia-smi
echo "NVIDIA GPU status:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TensorFlow informational messages
# Optional: Consider mixed precision if your model/TF version supports it well
# export TF_ENABLE_AUTO_MIXED_PRECISION=1 

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
echo "Python executable path: $(command -v $PYTHON_CMD)"
echo "PYTHONPATH: $PYTHONPATH" # Good to check this in the HPC environment

# Optional: Activate a virtual environment if you use one on HPC
# source /path/to/your/hpc_venv/bin/activate

# --- Execution ---
# Change to the directory where the Python script and its modules are located.
# This is crucial for Python to resolve relative imports within your project.
echo "Changing directory to work dir: $WORK_DIR"
cd "$WORK_DIR" || { echo "ERROR: Failed to change directory to $WORK_DIR"; exit 1; }
echo "Current directory after cd: $(pwd)"

# Check if TensorFlow can see the GPU (after cd and potential venv activation)
echo "TensorFlow GPU availability check (from $WORK_DIR):"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Verify the Python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT_PATH' not found in $WORK_DIR."
    exit 1
fi
# Optional: Add a check for a specific module file to ensure imports will work
if [ ! -f "config.py" ]; then # Assuming config.py is in WORK_DIR
    echo "WARNING: config.py not found in $(pwd). Python imports might fail."
fi

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
echo "Mean Teacher v2 training script (HPC) finished at $(date)."
echo "Output and logs are in $CURRENT_OUTPUT_DIR"
echo "Check SLURM error file: meanteacher_v2_hpc-$SLURM_JOB_ID.err for any issues."
echo "========================================================="

exit 0