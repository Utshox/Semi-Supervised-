import os
import sys
import time
import tensorflow as tf
import numpy as np
import subprocess

print("===== GPU Test with CPU Fallback =====")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Set environment variables that might help with GPU detection
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Verbose logging

# Helper function to run shell commands
def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Error ({e.returncode}): {e.output}"

# Check if NVIDIA driver is available
print("\nChecking NVIDIA driver:")
nvidia_smi_output = run_command("nvidia-smi")
print(nvidia_smi_output)

# Print environment variables
print("\nEnvironment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Check for CUDA libraries
cuda_lib_paths = run_command("find /usr -name 'libcuda.so*' 2>/dev/null")
print("\nCUDA library paths:")
print(cuda_lib_paths or "No CUDA libraries found in /usr")

# Try to detect GPUs
print("\nDetecting physical devices:")
physical_devices = tf.config.list_physical_devices()
print(f"All physical devices: {physical_devices}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {gpus}")

# Try to force-enable GPU
print("\nTrying to directly create a logical GPU device:")
try:
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"Logical GPU devices: {logical_gpus}")
except Exception as e:
    print(f"Error getting logical GPU devices: {e}")

# Configure GPU memory growth if GPUs are available
if gpus:
    print("Configuring GPU memory growth...")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except Exception as e:
        print(f"Error configuring GPU memory growth: {e}")

# Try alternate way of detecting GPU
print("\nTrying alternate method of detecting GPU:")
try:
    with tf.device('/device:GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result: {c}")
        print("GPU access successful via device placement!")
except Exception as e:
    print(f"Error using GPU via device placement: {e}")

# Create two random matrices for performance test
print("\nCreating test matrices...")
matrix_size = 5000
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# Test computation speed with explicit device placement
print("\nTesting matrix multiplication:")

# Try GPU first if available
if gpus:
    print("Attempting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            start = time.time()
            c_gpu = tf.matmul(a, b)
            # Force execution
            result = c_gpu.numpy()
            gpu_time = time.time() - start
            print(f"GPU time: {gpu_time:.4f} seconds")
    except Exception as e:
        print(f"Error during GPU computation: {e}")
        gpu_time = float('inf')
else:
    print("No GPU available, skipping GPU test")
    gpu_time = float('inf')

# Always test on CPU for comparison
print("Attempting CPU computation...")
with tf.device('/CPU:0'):
    start = time.time()
    c_cpu = tf.matmul(a, b)
    # Force execution
    result = c_cpu.numpy()
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")

# Compare times
if gpu_time != float('inf'):
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\nGPU is {speedup:.2f}x faster than CPU!")
    else:
        slowdown = gpu_time / cpu_time
        print(f"\nGPU is {slowdown:.2f}x slower than CPU. Something might be wrong with GPU acceleration.")
else:
    print("\nCould not compare CPU and GPU performance because GPU computation failed.")

print("\nGPU test complete!")