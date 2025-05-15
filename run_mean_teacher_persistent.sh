#!/bin/bash
# SLURM batch job script for Mean Teacher semi-supervised learning with advanced techniques
# This script EXECUTES an EXISTING Python script, it does NOT generate it.

#SBATCH -p gpu                        # Use gpu partition 
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH -n 8                          # Request 8 CPU cores
#SBATCH --mem=64G                     # Request 64GB RAM
#SBATCH --time=00:10:00               # Time limit to 10 minutes (adjust as needed)
#SBATCH -o meanteacher-persistent-%j.out     # Output file name with job ID
#SBATCH -e meanteacher-persistent-%j.err     # Error file name with job ID
#SBATCH --mail-type=END,FAIL          # Send email when job ends or fails
#SBATCH --job-name=pancreas_mt_persist # Add descriptive job name

# Path to your data directory - using your HPC paths
DATA_DIR="/scratch/lustre/home/mdah0000/images/preprocessed_v2/"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14" # This is /stud3/2023/mdah0000/smm/Semi-Supervised- in your local mapping
PYTHON_SCRIPT_PATH="/scratch/lustre/home/mdah0000/smm/v14/run_mean_teacher_enhanced.py" # Absolute path to the Python script
OUTPUT_DIR="$WORK_DIR/mean_teacher_persistent_results" # Changed output directory to avoid conflicts

echo "========================================================"
echo "Running PERSISTENT Advanced Mean Teacher Learning - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory: $WORK_DIR"
echo "Python script: $PYTHON_SCRIPT_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/visualizations

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Add user's local bin to PATH to find pip-installed executables
export PATH="/scratch/lustre/home/mdah0000/smm/v14/.local/bin:$PATH"

# Memory optimization settings
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_GPU_HOST_MEM_LIMIT_IN_MB=4096 # Adjust as needed
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"  # Enable XLA JIT compilation

# Find Python executable
PYTHON_CMD=""
for cmd in python3 python /usr/bin/python3 /usr/bin/python; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        echo "Found Python: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python executable not found."
    exit 1
fi

# Install required packages (consider if these are already in your environment)
# If they are in a loaded module or virtual environment, this might not be necessary or could be harmful.
# $PYTHON_CMD -m pip install --quiet tensorflow==2.15.0 tqdm matplotlib psutil scipy albumentations

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
$PYTHON_CMD -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change current directory to the Python script's directory.
# This ensures that the Python script can correctly import its sibling helper modules
# (e.g., config.py, main.py, train_ssl_tf2n.py) which are assumed to be co-located
# with the PYTHON_SCRIPT_PATH on the HPC.
TARGET_PYTHON_SCRIPT_DIR=$(dirname "$PYTHON_SCRIPT_PATH")
echo "Changing to Python script's directory for module resolution: $TARGET_PYTHON_SCRIPT_DIR"
cd "$TARGET_PYTHON_SCRIPT_DIR"

# Make the Python script executable (optional, as we call it with $PYTHON_CMD)
# chmod +x $PYTHON_SCRIPT_PATH 
# echo "Made script $PYTHON_SCRIPT_PATH executable (if not already)"

# Run the advanced Mean Teacher training script
echo "Running Advanced Mean Teacher training using $PYTHON_SCRIPT_PATH..."
$PYTHON_CMD $PYTHON_SCRIPT_PATH \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "mean_teacher_persistent_slurm" \
    --batch_size 2 \
    --num_epochs 150 \
    --img_size 256 \
    --num_labeled 30 \
    --num_validation 56

echo "========================================================="
echo "Advanced Mean Teacher Learning (Persistent Script) completed - $(date)"
echo "========================================================="
echo "Results are located in: $OUTPUT_DIR"
echo "========================================================="
