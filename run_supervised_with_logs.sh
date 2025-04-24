#!/bin/bash
# SLURM batch job script for supervised learning with improved logging

#SBATCH -p gpu                      # Use GPU partition
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH -n 4                        # Request 4 CPU cores
#SBATCH --mem=48G                   # Request 48GB RAM
#SBATCH --time=48:00:00             # Set time limit to 48 hours
#SBATCH -o /scratch/lustre/home/mdah0000/smm/logs/supervised-%j.out   # Output file with job ID
#SBATCH -e /scratch/lustre/home/mdah0000/smm/logs/supervised-%j.err   # Error file with job ID

# Create logs directory if it doesn't exist
LOG_DIR="/scratch/lustre/home/mdah0000/smm/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/supervised_training_${SLURM_JOB_ID}.log"
MEMORY_LOG="${LOG_DIR}/gpu_memory_${SLURM_JOB_ID}.log"

# Path to your data directory
DATA_DIR="/scratch/lustre/home/mdah0000/images/cropped"

# Print job information
echo "========================================================" | tee -a "$LOG_FILE"
echo "Starting Supervised Learning - $(date)" | tee -a "$LOG_FILE"
echo "SLURM Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Running on node: $HOSTNAME" | tee -a "$LOG_FILE"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Memory monitoring log: $MEMORY_LOG" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# Check for available GPUs
echo "Checking NVIDIA GPU status:" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"

# Setup for TensorFlow GPU
echo "Setting up GPU environment variables:" | tee -a "$LOG_FILE"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1

echo "Environment variables set:" | tee -a "$LOG_FILE"
echo "TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH" | tee -a "$LOG_FILE"
echo "TF_GPU_ALLOCATOR=$TF_GPU_ALLOCATOR" | tee -a "$LOG_FILE" 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

# Install correct TensorFlow version
echo "Installing TensorFlow with CUDA support:" | tee -a "$LOG_FILE"
pip install --quiet "tensorflow[and-cuda]==2.15.0.post1" | tee -a "$LOG_FILE"

# Check TensorFlow GPU detection
echo "Checking TensorFlow GPU detection:" | tee -a "$LOG_FILE"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices available:', tf.config.list_physical_devices('GPU')); print('Number of GPUs:', len(tf.config.list_physical_devices('GPU')))" | tee -a "$LOG_FILE"

# Start GPU memory monitoring in the background
echo "Starting GPU memory monitoring (every 30 seconds)..." | tee -a "$LOG_FILE"
{
  echo "Timestamp,GPU ID,GPU Name,Memory Used,Memory Total,GPU Utilization" > "$MEMORY_LOG"
  while true; do
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    while IFS=, read -r gpu_id gpu_name memory_used memory_total gpu_util; do
      echo "$timestamp,$gpu_id,$gpu_name,$memory_used,$memory_total,$gpu_util" >> "$MEMORY_LOG"
    done
    sleep 30
  done
} &
MONITOR_PID=$!

# Make sure to kill the monitoring process when the script ends
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Run quick GPU test to check memory
echo "Running quick GPU memory test:" | tee -a "$LOG_FILE"
python3 -c "
import tensorflow as tf
print('Available GPU Memory before allocation:')
for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
    print(f'GPU {i}:', tf.config.experimental.get_memory_info(f'GPU:{i}'))
    
# Allocate some tensors to test GPU memory
tensors = []
for i in range(2):  # For both GPUs
    with tf.device(f'/GPU:{i}'):
        # Allocate approximately 1GB on each GPU
        tensors.append(tf.random.normal([8000, 8000], dtype=tf.float32))
        
print('\\nAvailable GPU Memory after 1GB allocation per GPU:')
for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
    print(f'GPU {i}:', tf.config.experimental.get_memory_info(f'GPU:{i}'))
" | tee -a "$LOG_FILE"

# Run with optimal batch size for memory management
echo "Starting main training with batch size 4:" | tee -a "$LOG_FILE"
echo "Command: python3 /scratch/lustre/home/mdah0000/smm/v14/main.py --method supervised --data_dir $DATA_DIR --batch_size 4" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# Run the actual training
python3 /scratch/lustre/home/mdah0000/smm/v14/main.py \
  --method supervised \
  --data_dir $DATA_DIR \
  --batch_size 4 2>&1 | tee -a "$LOG_FILE"

# Check exit status
TRAINING_STATUS=${PIPESTATUS[0]}

echo "========================================================" | tee -a "$LOG_FILE"
if [ $TRAINING_STATUS -eq 0 ]; then
  echo "Supervised Learning completed successfully - $(date)" | tee -a "$LOG_FILE"
else
  echo "Supervised Learning failed with status $TRAINING_STATUS - $(date)" | tee -a "$LOG_FILE"
  echo "Checking final GPU status:" | tee -a "$LOG_FILE"
  nvidia-smi | tee -a "$LOG_FILE"
fi
echo "========================================================" | tee -a "$LOG_FILE"

# Print information about log files
echo "Log files available at:"
echo "- Main log: $LOG_FILE"
echo "- GPU memory log: $MEMORY_LOG"
echo "- SLURM output: logs/supervised-${SLURM_JOB_ID}.out"
echo "- SLURM errors: logs/supervised-${SLURM_JOB_ID}.err"
echo "========================================================"