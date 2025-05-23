#!/bin/bash
# SLURM batch job script for supervised learning

#SBATCH -p gpu                     # Use gpu partition 
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 4                       # Request 4 CPU cores
#SBATCH --mem=48G                  # Request 48GB RAM (12GB per CPU as per HPC docs)
#SBATCH --time=02:00:00            # Set time limit to 48 hours (max for gpu partition)
#SBATCH -o supervised-%j.out       # Output file name with job ID
#SBATCH -e supervised-%j.err       # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"
WORK_DIR="/scratch/lustre/home/mdah0000/smm/v14"

echo "========================================================"
echo "Running Supervised Learning - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory: $WORK_DIR"
echo "Data directory: $DATA_DIR"
echo "========================================================"

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# TensorFlow GPU environment settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=0  # Enable verbose logging for debugging

# Install the correct TensorFlow version if needed (will be skipped if already installed)
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1"

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Change to the working directory
cd $WORK_DIR

# Create cache directory if it doesn't exist
mkdir -p preprocessed_cache
echo "Created preprocessed_cache directory for saving processed data"

# Run with optimal batch size for memory management
echo "Running with batch size 2 for memory optimization"
python3 $WORK_DIR/main.py \
  --training_type supervised \
  --experiment_name supervised_baseline \
  --data_dir $DATA_DIR \

echo "========================================================"
echo "Supervised Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: $WORK_DIR/supervised_results"
echo "========================================================"