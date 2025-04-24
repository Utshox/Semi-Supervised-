#!/bin/bash
# Script to set up a conda environment with TensorFlow GPU support

echo "======================================================"
echo "Setting up conda environment for TensorFlow GPU training"
echo "======================================================"

# Load conda module (common in HPC environments)
module load anaconda3 2>/dev/null || module load conda 2>/dev/null || echo "No conda module found, assuming conda is in PATH"

# Initialize conda for bash shell
source $(conda info --base)/etc/profile.d/conda.sh

# Remove existing environment if it exists
conda env remove -n tf-gpu 2>/dev/null
echo "Creating new conda environment tf-gpu with Python 3.10..."

# Create a new environment with Python 3.10
conda create -n tf-gpu python=3.10 -y

# Activate the environment
echo "Activating environment..."
conda activate tf-gpu

# Install CUDA and cuDNN through conda
echo "Installing CUDA toolkit and cuDNN..."
conda install -c conda-forge cudatoolkit=12.1 cudnn=8.9 -y

# Set LD_LIBRARY_PATH to help TensorFlow find CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install requirements with pip
echo "Installing TensorFlow and other requirements..."
pip install --no-cache-dir -r /stud3/2023/mdah0000/smm/Semi-Supervised-/requirements.txt

# Create a setup script that can be sourced to activate the environment
echo "Creating environment activation script..."
cat > /stud3/2023/mdah0000/smm/Semi-Supervised-/activate_tf_env.sh << 'EOL'
#!/bin/bash
# Script to activate the TensorFlow GPU environment

# Load conda
module load anaconda3 2>/dev/null || module load conda 2>/dev/null || echo "No conda module found, assuming conda is in PATH"

# Initialize conda for bash shell
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate tf-gpu

# Set environment variables for TensorFlow
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Verify the setup
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Checking GPU availability in TensorFlow:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"
EOL

chmod +x /stud3/2023/mdah0000/smm/Semi-Supervised-/activate_tf_env.sh

# Test the environment
echo "======================================================"
echo "Testing TensorFlow installation..."
conda activate tf-gpu
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU available:', bool(tf.config.list_physical_devices('GPU')))"
echo "======================================================"
echo "Setup complete!"
echo "To activate this environment manually, run:"
echo "source /stud3/2023/mdah0000/smm/Semi-Supervised-/activate_tf_env.sh"
echo "To run training with SLURM, submit:"
echo "sbatch run_supervised_conda.sh"
echo "======================================================"