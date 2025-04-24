#!/bin/bash
# SLURM batch job script for supervised learning with conda environment

#SBATCH -p gpu                     # Use gpu partition as specified in VU MIF HPC docs
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 4                       # Request 4 CPU cores
#SBATCH --mem=48G                  # Request 48GB RAM (12GB per CPU as per HPC docs)
#SBATCH --time=48:00:00            # Set time limit to 48 hours (max for gpu partition)
#SBATCH -o supervised-conda-%j.out # Output file name with job ID
#SBATCH -e supervised-conda-%j.err # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"

echo "========================================================"
echo "Running Supervised Learning with Conda - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Load conda module if available
module load anaconda3 || module load conda || echo "Please make sure conda is available"

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU with nvidia-smi:"
nvidia-smi

# Initialize conda in bash script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
echo "Activating conda environment tf-gpu..."
conda activate tf-gpu || echo "Failed to activate conda environment. Make sure it exists."

# Verify Python and TensorFlow versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Set environment variables for TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=0  # Enable verbose logging for debugging

# Run with smaller batch size for better memory management
echo "Running with batch size 2 for memory optimization"
python /scratch/lustre/home/mdah0000/smm/v14/main.py \
  --method supervised \
  --data_dir $DATA_DIR \
  --batch_size 2

echo "========================================================"
echo "Supervised Learning completed - $(date)"
echo "========================================================"
echo "Results are located in: supervised_results"
echo "========================================================"