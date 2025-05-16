#!/bin/bash
# SLURM batch job script for Mean Teacher v2 semi-supervised learning

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
# Adjust these paths and parameters as needed for your HPC environment and experiment
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2/" # Root directory of preprocessed data
# WORK_DIR should be the directory where your Python scripts and this shell script are located on the HPC.
# This corresponds to /stud3/2023/mdah0000/smm/Semi-Supervised- in your local setup.
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14" # Example, adjust if your HPC path is different
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py" # The new main script
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_BASE="$WORK_DIR/mean_teacher_v2_results" # Base directory for experiment outputs

# Experiment specific parameters (can be overridden by command-line args to this script if implemented)
EXPERIMENT_NAME="mt_v2_$(date +%Y%m%d_%H%M%S)"
IMG_SIZE=256
NUM_LABELED=30 # Example: 30 labeled patients
NUM_VALIDATION=10 # Example: 10 validation patients
BATCH_SIZE=4
NUM_EPOCHS=150
LEARNING_RATE=1e-4
EMA_DECAY=0.999
CONSISTENCY_MAX=10.0
CONSISTENCY_RAMPUP=30 # Epochs
EARLY_STOPPING_PATIENCE=20
SEED=42
VERBOSE=1 # 0=silent, 1=progress bar, 2=one line per epoch

# --- SLURM Preamble ---
echo "========================================================="
echo "Running Mean Teacher v2 Learning - $(date)"
echo "========================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory (where script is run from): $(pwd)" # Should be $WORK_DIR after cd
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

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TensorFlow informational messages

# Optional: Add user's local bin to PATH if Python packages are installed there
# export PATH="$WORK_DIR/.local/bin:$PATH" # Adjust if you have a .local in your WORK_DIR

# Optional: Memory optimization settings (use with caution, defaults are often fine)
# export TF_GPU_ALLOCATOR=cuda_malloc_async
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

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

# Optional: Activate a virtual environment if you use one
# source /path/to/your/venv/bin/activate

# --- Execution ---
# Change to the directory where the Python script is located.
# This is crucial for Python to resolve relative imports within your project.
echo "Changing directory to: $WORK_DIR"
cd "$WORK_DIR" || { echo "ERROR: Failed to change directory to $WORK_DIR"; exit 1; }

# Check if TensorFlow can see the GPU (after cd and potential venv activation)
echo "TensorFlow GPU availability check (from $WORK_DIR):"
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

# Example: You might want to copy important results to a more permanent storage
# cp "$CURRENT_OUTPUT_DIR/training_log.csv" "/path/to/permanent/storage/"
# cp "$CURRENT_OUTPUT_DIR/checkpoints/best_student_model*.weights.h5" "/path/to/permanent/storage/"

exit 0
