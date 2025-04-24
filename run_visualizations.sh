#!/bin/bash
# SLURM batch job script for generating visualizations for MS thesis

#SBATCH -p gpu                      # Use GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU (visualization doesn't need 2)
#SBATCH -n 2                        # Request 2 CPU cores
#SBATCH --mem=32G                   # Request 32GB RAM
#SBATCH --time=12:00:00             # Set time limit to 12 hours
#SBATCH -o /scratch/lustre/home/mdah0000/smm/logs/visualizations-%j.out  # Output file
#SBATCH -e /scratch/lustre/home/mdah0000/smm/logs/visualizations-%j.err  # Error file

# Create logs directory if it doesn't exist
LOG_DIR="/scratch/lustre/home/mdah0000/smm/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/thesis_visualizations_${SLURM_JOB_ID}.log"

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"

# Create output directory for visualizations 
VIZ_DIR="/scratch/lustre/home/mdah0000/smm/thesis_visualizations"
mkdir -p $VIZ_DIR

# Path to the models directory (containing saved models)
MODELS_DIR="/scratch/lustre/home/mdah0000/smm/models"

# Print job information
echo "========================================================" | tee -a "$LOG_FILE"
echo "Starting Visualization Generation for MS Thesis - $(date)" | tee -a "$LOG_FILE"
echo "SLURM Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Running on node: $HOSTNAME" | tee -a "$LOG_FILE"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Visualization output directory: $VIZ_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# Check for available GPUs
echo "Checking NVIDIA GPU status:" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"

# Setup for TensorFlow GPU
echo "Setting up GPU environment variables:" | tee -a "$LOG_FILE"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_VISIBLE_DEVICES=0

echo "Environment variables set:" | tee -a "$LOG_FILE"
echo "TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH" | tee -a "$LOG_FILE"
echo "TF_GPU_ALLOCATOR=$TF_GPU_ALLOCATOR" | tee -a "$LOG_FILE" 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

# Install TensorFlow and other required packages
echo "Installing TensorFlow with CUDA support:" | tee -a "$LOG_FILE"
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1" | tee -a "$LOG_FILE"
pip install --quiet matplotlib seaborn scikit-image pandas SimpleITK scipy | tee -a "$LOG_FILE"

# Check TensorFlow GPU detection
echo "Checking TensorFlow GPU detection:" | tee -a "$LOG_FILE"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices available:', tf.config.list_physical_devices('GPU')); print('Number of GPUs:', len(tf.config.list_physical_devices('GPU')))" | tee -a "$LOG_FILE"

# Set script to executable mode
chmod +x generate_visualizations.py

# Run the visualization generation
echo "========================================================" | tee -a "$LOG_FILE"
echo "Running visualization generation..." | tee -a "$LOG_FILE" 
echo "Command: python3 generate_visualizations.py --data_dir $DATA_DIR --output_dir $VIZ_DIR --model_dir $MODELS_DIR --batch_size 2 --compare_methods" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# Run the script
python3 generate_visualizations.py \
  --data_dir $DATA_DIR \
  --output_dir $VIZ_DIR \
  --model_dir $MODELS_DIR \
  --batch_size 2 \
  --compare_methods 2>&1 | tee -a "$LOG_FILE"

# Check exit status
VISUALIZATION_STATUS=${PIPESTATUS[0]}

echo "========================================================" | tee -a "$LOG_FILE"
if [ $VISUALIZATION_STATUS -eq 0 ]; then
  echo "Visualization generation completed successfully - $(date)" | tee -a "$LOG_FILE"
else
  echo "Visualization generation failed with status $VISUALIZATION_STATUS - $(date)" | tee -a "$LOG_FILE"
fi
echo "========================================================" | tee -a "$LOG_FILE"

# Copy results to your home directory for easier access 
HOME_VIZ_DIR="$HOME/thesis_visualizations"
mkdir -p $HOME_VIZ_DIR
echo "Copying visualization results to $HOME_VIZ_DIR for easy access..." | tee -a "$LOG_FILE"
cp -r $VIZ_DIR/* $HOME_VIZ_DIR/

echo "All visualizations have been copied to $HOME_VIZ_DIR" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"