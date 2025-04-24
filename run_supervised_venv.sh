#!/bin/bash
# SLURM batch job script for supervised learning with Python virtual environment

#SBATCH -p gpu                     # Use gpu partition as specified in VU MIF HPC docs
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH -n 4                       # Request 4 CPU cores
#SBATCH --mem=48G                  # Request 48GB RAM (12GB per CPU as per HPC docs)
#SBATCH --time=48:00:00            # Set time limit to 48 hours (max for gpu partition)
#SBATCH -o supervised-venv-%j.out  # Output file name with job ID
#SBATCH -e supervised-venv-%j.err  # Error file name with job ID

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"

echo "========================================================"
echo "Running Supervised Learning with Virtual Env - $(date)"
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Check for available GPU with nvidia-smi
echo "Checking NVIDIA GPU with nvidia-smi:"
nvidia-smi

# Activate the virtual environment
source $HOME/tf_gpu_venv/bin/activate

# Set environment variables for TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=0  # Enable verbose logging for debugging
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/nvidia/current:$LD_LIBRARY_PATH"

# Verify Python and TensorFlow versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Check if TensorFlow can see the GPU
echo "TensorFlow GPU availability:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Verify CUDA setup
python -c "
import tensorflow as tf
import os
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH'))
print('TF_FORCE_GPU_ALLOW_GROWTH:', os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'))
print('Is built with CUDA:', tf.test.is_built_with_cuda())
print('Is GPU available:', tf.test.is_gpu_available())
"

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