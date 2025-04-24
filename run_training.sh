#!/bin/bash

# SLURM job submission script for GPU-based training
#SBATCH --job-name=pancreas_ssl       # Job name
#SBATCH --output=job_%j.log           # Standard output and error log
#SBATCH --partition=gpu               # Request the GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=24:00:00               # Request 24 hours runtime
#SBATCH --mem=32G                     # Request 32GB of memory
#SBATCH --cpus-per-task=4             # Request 4 CPUs

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate your environment if needed
# source /path/to/your/environment/bin/activate

# Check if NVIDIA is available
echo "Checking for NVIDIA GPU:"
nvidia-smi

# Load necessary modules (adjust as needed for your HPC)
# module load cuda/11.8
# module load cudnn/8.6.0

# Set data path
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"

# Run the training script
echo "Starting training with Mean Teacher approach..."
python main.py --method mean_teacher --data_dir $DATA_DIR

echo "Job finished at $(date)"