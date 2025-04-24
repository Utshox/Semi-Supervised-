#!/bin/bash
# Script to set up a Python virtual environment with TensorFlow GPU support for VU MIF HPC

echo "======================================================"
echo "Setting up Python virtual environment for TensorFlow GPU training"
echo "======================================================"

# Define the virtual environment path
VENV_DIR="$HOME/tf_gpu_venv"

# Remove existing environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new Python virtual environment
echo "Creating new Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate the environment
echo "Activating environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install specific TensorFlow version that works with GPU
echo "Installing TensorFlow 2.15.0.post1 with CUDA support..."
pip install "tensorflow[and-cuda]==2.15.0.post1"

# Install other requirements (excluding tensorflow which we installed separately)
echo "Installing other requirements..."
pip install --no-cache-dir numpy matplotlib pandas nibabel tqdm scikit-image Pillow

# Create an activation script
echo "Creating environment activation script..."
cat > /scratch/lustre/home/mdah0000/smm/activate_tf_env.sh << EOL
#!/bin/bash
# Script to activate the TensorFlow GPU environment

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Set environment variables for TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=0

# Check if the system has the correct CUDA libraries in LD_LIBRARY_PATH
# According to VU MIF HPC docs, GPUs are Tesla V100
if [[ -z "\$LD_LIBRARY_PATH" || ! "\$LD_LIBRARY_PATH" =~ "cuda" ]]; then
    echo "Adding CUDA libraries to LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/nvidia/current:\$LD_LIBRARY_PATH"
fi

# Verify the setup
echo "Python: \$(python --version)"
echo "TensorFlow: \$(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Checking GPU availability in TensorFlow:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"
EOL

chmod +x /scratch/lustre/home/mdah0000/smm/activate_tf_env.sh

# Test the environment to verify GPU detection works
echo "======================================================"
echo "Testing TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU available:', bool(tf.config.list_physical_devices('GPU')))"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

echo "======================================================"
echo "Setup complete!"
echo "To activate this environment manually, run:"
echo "source /scratch/lustre/home/mdah0000/smm/activate_tf_env.sh"
echo "To run training with SLURM, submit:"
echo "sbatch /scratch/lustre/home/mdah0000/smm/run_supervised_venv.sh"
echo "======================================================"