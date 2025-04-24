#!/bin/bash
# Enhanced helper script for TensorFlow GPU detection on VU MIF HPC

# Print diagnostic information
echo "Setting up GPU environment for VU MIF HPC system..."
echo "Current CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Ensure CUDA_VISIBLE_DEVICES is set from SLURM allocation
# On VU MIF HPC, SLURM sets CUDA_VISIBLE_DEVICES automatically when using --gres=gpu:N
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  # If not set, check SLURM env variables
  if [ -n "${SLURM_STEP_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS}
    echo "Setting CUDA_VISIBLE_DEVICES from SLURM_STEP_GPUS: ${SLURM_STEP_GPUS}"
  elif [ -n "${SLURM_JOB_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
    echo "Setting CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS: ${SLURM_JOB_GPUS}"
  else
    # If still not set, default to use all GPUs (0,1)
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "No SLURM GPU variables found, defaulting to CUDA_VISIBLE_DEVICES=0,1"
  fi
fi

# Set GPU memory growth and optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
export TF_FORCE_GPU=1

# Look for CUDA installation in common locations on VU MIF HPC
# Based on the documentation, the system has NVIDIA Tesla V100 GPUs
CUDA_LOCATIONS=(
  "/scratch/lustre/cuda"
  "/opt/nvidia"
  "/usr/local/cuda"
  "/usr/local/cuda-12.4"
  "/usr/local/cuda-12.0"
  "/usr/local/cuda-11.8"
  "/usr/local/cuda-11.0"
  "/opt/cuda"
)

echo "Searching for CUDA directories on VU MIF HPC system..."

found_cuda=false
for cuda_path in "${CUDA_LOCATIONS[@]}"; do
  if [ -d "$cuda_path" ]; then
    echo "Found CUDA installation at $cuda_path"
    export CUDA_HOME=$cuda_path
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    found_cuda=true
    break
  else
    echo "Checked $cuda_path - not found"
  fi
done

if [ "$found_cuda" = false ]; then
  echo "Trying alternative method to locate CUDA on VU MIF HPC system..."
  
  # Look for any loaded CUDA module
  if command -v module &> /dev/null; then
    echo "Checking for loaded CUDA modules..."
    module list | grep -i cuda
    
    # Try to load a CUDA module if available
    echo "Trying to load a CUDA module..."
    module load cuda 2>/dev/null || module load nvidia 2>/dev/null || echo "No CUDA module available"
    module list | grep -i cuda
  fi
  
  # Use find command as last resort
  echo "Looking for CUDA libraries with find command..."
  potential_cuda_path=$(find /usr -type d -name "cuda*" -maxdepth 2 2>/dev/null | grep -v "include" | head -n 1)
  
  if [ -n "$potential_cuda_path" ] && [ -d "$potential_cuda_path" ]; then
    echo "Found potential CUDA installation at $potential_cuda_path"
    export CUDA_HOME=$potential_cuda_path
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
  else
    echo "WARNING: Could not locate CUDA installation. GPU acceleration may not be available."
    
    # Check if NVIDIA driver is available without CUDA
    if command -v nvidia-smi &> /dev/null; then
      echo "NVIDIA driver is available (nvidia-smi found) but CUDA libraries not located."
      echo "Setting up environment for basic GPU usage without specific CUDA path."
      
      # Try standard system paths for CUDA libraries
      export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    fi
  fi
fi

# TensorFlow specific configuration for V100 GPUs on VU MIF HPC
export TF_CUDA_PATHS=$CUDA_HOME
export TF_KERAS_GPU_MEMORY_LIMIT=16384  # V100s have 32GB, use 16GB per GPU to be safe

# Print new settings
echo "New CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "New CUDA_HOME: ${CUDA_HOME}"
echo "New LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "TF_FORCE_GPU_ALLOW_GROWTH: ${TF_FORCE_GPU_ALLOW_GROWTH}"
echo "TF_FORCE_GPU: ${TF_FORCE_GPU}"

# Check if CUDA is available using nvidia-smi
echo "Checking NVIDIA GPU:"
nvidia-smi

# Check UofM MIF HPC resource usage (useful for debugging quota issues)
if [ -n "$SLURM_JOB_ID" ]; then
  echo "Current resource usage for your account:"
  sreport -T cpu,mem,gres/gpu cluster AccountUtilizationByUser Start=$(date +%m01) End=$(date +%m31) User=$USER 2>/dev/null || echo "Could not get resource usage report"
fi

# Execute the passed command with the environment variables set
echo "Running command: $@"
"$@"