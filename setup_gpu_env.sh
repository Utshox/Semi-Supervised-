#!/bin/bash
# Enhanced helper script for TensorFlow GPU detection on VU MIF HPC

# Print diagnostic information
echo "Setting up GPU environment..."
echo "Current CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Preserve all GPUs allocated by SLURM
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

# Check if the GPU is available with nvidia-smi
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
  echo "ERROR: nvidia-smi not available or not working. GPU acceleration will not be available."
  exit 1
fi

# Extract CUDA version from nvidia-smi
echo "Extracting CUDA version from nvidia-smi..."
CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | cut -d ' ' -f 3)
echo "Detected CUDA version: $CUDA_VERSION"

# Find nvidia libraries using ldconfig
echo "Finding NVIDIA libraries via ldconfig..."
NVIDIA_LIBS=$(ldconfig -p | grep -E "nvidia|cuda|cudnn" | awk '{print $4}' | uniq | xargs dirname 2>/dev/null | sort | uniq)

if [ -z "$NVIDIA_LIBS" ]; then
  echo "No NVIDIA libraries found via ldconfig"
else
  echo "Found NVIDIA libraries in:"
  for lib_path in $NVIDIA_LIBS; do
    echo "  - $lib_path"
    export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
  done
fi

# Find CUDA libraries directly - restrict search to system directories only
echo "Searching for CUDA libraries in system directories..."
CUDA_LIB_PATHS=$(find /usr /lib /usr/lib /usr/lib64 /usr/local/cuda* -name "libcudart.so*" -o -name "libcublas.so*" -o -name "libcudnn.so*" -o -name "libcufft.so*" 2>/dev/null | xargs dirname 2>/dev/null | sort | uniq)

if [ -z "$CUDA_LIB_PATHS" ]; then
  echo "No CUDA libraries found directly"
else
  echo "Found CUDA libraries in:"
  for lib_path in $CUDA_LIB_PATHS; do
    echo "  - $lib_path"
    export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
  done
fi

# Only search in your own directories in /scratch/lustre
echo "Searching for CUDA libraries in your own scratch directory..."
MY_USERNAME=$(whoami)
if [ -d "/scratch/lustre/home/$MY_USERNAME" ]; then
  USER_CUDA_PATHS=$(find "/scratch/lustre/home/$MY_USERNAME" -name "libcuda*.so*" -o -name "libcudnn*.so*" 2>/dev/null | xargs dirname 2>/dev/null | sort | uniq)
  if [ -n "$USER_CUDA_PATHS" ]; then
    echo "Found CUDA libraries in your home directory:"
    for lib_path in $USER_CUDA_PATHS; do
      echo "  - $lib_path"
      export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
    done
  fi
fi

# Add common library paths
echo "Adding common system library paths..."
export LD_LIBRARY_PATH="/lib:/lib64:/usr/lib:/usr/lib64:/usr/local/lib:/usr/local/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Try to set CUDA_HOME based on driver version
if [[ "$CUDA_VERSION" == "12.4" ]]; then
  export CUDA_HOME="/usr/local/cuda-12.4"
  if [ ! -d "$CUDA_HOME" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi

# If CUDA_HOME still not set or doesn't exist, try to find it
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
  for cuda_path in "/usr/local/cuda" "/usr/local/cuda-$CUDA_VERSION" "/usr/lib/cuda"; do
    if [ -d "$cuda_path" ]; then
      export CUDA_HOME="$cuda_path"
      echo "Found CUDA_HOME: $CUDA_HOME"
      break
    fi
  done
fi

# If CUDA_HOME still not set, use a fallback
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
  echo "WARNING: Could not find CUDA_HOME directory. Using driver paths only."
  # Check if /usr/lib/x86_64-linux-gnu/nvidia/current exists (common on some systems)
  if [ -d "/usr/lib/x86_64-linux-gnu/nvidia/current" ]; then
    export CUDA_HOME="/usr/lib/x86_64-linux-gnu/nvidia/current"
    echo "Using NVIDIA driver directory as CUDA_HOME: $CUDA_HOME"
  else
    export CUDA_HOME="/usr"
    echo "Using /usr as CUDA_HOME fallback"
  fi
fi

# Check if we have key CUDA libraries for TensorFlow
echo "Checking for essential CUDA libraries for TensorFlow..."
for lib in libcudart.so libcublas.so libcufft.so libcusolver.so; do
  LIB_PATH=$(ldconfig -p | grep "$lib" | head -n 1 | awk '{print $4}' 2>/dev/null)
  if [ -n "$LIB_PATH" ]; then
    echo "Found $lib: $LIB_PATH"
  else
    echo "Warning: $lib not found in library cache"
  fi
done

# Essential TensorFlow environment settings for GPU usage
echo "Setting critical TensorFlow environment variables for GPU usage..."
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_JIT=1
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_auto_jit=2"
export TF_FORCE_GPU=1
export TF_NEED_CUDA=1
export TF_USE_CUDNN=1
export TF_KERAS_GPU_MEMORY_LIMIT=16384  # V100s have 32GB, use 16GB per GPU
export TF_CUDA_VERSION="$CUDA_VERSION"

# Add specific driver paths for CUDA 12.4
echo "Adding CUDA 12.4 specific paths..."
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Add any found CUDA bin directories to PATH - only from system directories
for bin_path in $(find /usr /usr/local -path "*/cuda*/bin" 2>/dev/null); do
  if [ -d "$bin_path" ]; then
    echo "Adding CUDA bin path: $bin_path"
    export PATH="$bin_path:$PATH"
  fi
done

# Print final environment settings
echo "Final GPU environment settings:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "TF_FORCE_GPU_ALLOW_GROWTH: ${TF_FORCE_GPU_ALLOW_GROWTH}"
echo "TF_FORCE_GPU: ${TF_FORCE_GPU}"
echo "TF_NEED_CUDA: ${TF_NEED_CUDA}"
echo "TF_KERAS_GPU_MEMORY_LIMIT: ${TF_KERAS_GPU_MEMORY_LIMIT}"
echo "PATH: ${PATH}"

# Check if CUDA is available using nvidia-smi
echo "Final GPU status check:"
nvidia-smi

# Execute the passed command with the environment variables set
echo "Running command: $@"
"$@"