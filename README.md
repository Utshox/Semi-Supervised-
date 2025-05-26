# Semi-Supervised Learning for Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation and evaluation of semi-supervised learning techniques for medical image segmentation, specifically focusing on pancreas and cancer segmentation from CT scans. This project implements and compares supervised learning with two state-of-the-art semi-supervised approaches: Mean Teacher and MixMatch.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Medical image segmentation often faces the challenge of limited labeled data due to the expensive and time-consuming nature of expert annotations. This project explores how semi-supervised learning can leverage unlabeled data to improve segmentation performance while reducing the dependency on large labeled datasets.

### Key Features

- **U-Net Architecture**: Custom implementation optimized for medical image segmentation
- **Three Learning Paradigms**: Supervised baseline, Mean Teacher, and MixMatch
- **Memory Optimization**: Multiple performance-optimized versions for HPC environments
- **Comprehensive Evaluation**: Detailed analysis and visualization tools
- **Mixed Precision Training**: GPU memory optimization with FP16
- **Early Stopping**: Prevents overfitting with configurable patience

### Research Objectives

1. Establish a strong supervised baseline for pancreas segmentation
2. Evaluate the effectiveness of Mean Teacher in leveraging unlabeled data
3. Compare MixMatch performance against traditional approaches
4. Analyze the trade-offs between labeled data requirements and performance

## 📊 Dataset

### Pancreas Dataset (Medical Segmentation Decathlon)

- **Source**: Memorial Sloan Kettering Cancer Center
- **License**: CC-BY-SA 4.0
- **Modality**: CT scans
- **Image Dimensions**: 3D volumes processed as 2D slices
- **Input Size**: 256×256 pixels (optimized for memory efficiency)

#### Dataset Statistics
- **Training Samples**: 281 volumes
- **Test Samples**: 139 volumes
- **Label Classes**:
  - 0: Background
  - 1: Pancreas
  - 2: Cancer (merged with pancreas for binary segmentation)

#### Data Preprocessing
- Intensity normalization to [0, 1] range
- Slice extraction from 3D volumes
- Data augmentation: rotation, scaling, elastic deformation
- Train/validation split with stratification

## 🏗️ Architecture

### U-Net Model (`models_tf2.py`)

Our implementation features a modified U-Net architecture optimized for pancreas segmentation:

```python
class PancreasSeg(Model):
    """
    U-Net model for pancreas segmentation with support for dropout.
    Features:
    - Encoder-decoder architecture with skip connections
    - Configurable filter sizes (16, 32, 64 for different memory requirements)
    - Dropout layers for regularization
    - Mixed precision support
    """
```

**Architecture Details**:
- **Encoder**: 4 downsampling blocks with MaxPooling
- **Decoder**: 4 upsampling blocks with transposed convolutions
- **Skip Connections**: Concatenate encoder features with decoder
- **Final Layer**: 1×1 convolution for binary segmentation
- **Activation**: Sigmoid for probability outputs

### Loss Functions

- **Dice Loss**: Primary metric for segmentation overlap
- **Binary Cross-Entropy**: Pixel-wise classification loss
- **Combined Loss**: Weighted combination for optimal training

## 🧪 Experiments

### 1. Supervised Baseline

**Objective**: Establish performance ceiling using all available labeled data.

#### Configuration
```bash
Script: run_supervised_enhanced.sh
Image Size: 256×256
Epochs: 100 (with early stopping)
Batch Size: 8
Labeled Samples: 225
Validation Samples: 56
Model Filters: 32
Resources: 2 GPUs, 8 CPUs, 64GB RAM
```

#### Results
- **Best Validation Dice**: 0.8577 (Epoch 59)
- **Final Validation Dice**: 0.8449
- **Training Time**: 4.51 hours (74 epochs)
- **Early Stopping**: Triggered at epoch 74 (patience: 15)

#### Key Insights
- Strong baseline performance demonstrates dataset quality
- Early stopping prevents overfitting effectively
- Mixed precision training reduces memory usage by ~30%

### 2. Mean Teacher Semi-Supervised Learning

**Objective**: Leverage unlabeled data through consistency regularization with teacher-student framework.

#### Methodology
The Mean Teacher approach consists of:
- **Student Model**: Trained with supervised loss on labeled data
- **Teacher Model**: Exponential Moving Average (EMA) of student weights
- **Consistency Loss**: MSE between student and teacher predictions on unlabeled data
- **Weight Update**: `teacher_weights = α × teacher_weights + (1-α) × student_weights`

#### Configuration
```bash
Script: run_mean_teacher_highperf.sh
Image Size: 256×256
Epochs: 100
Batch Size: 4 (memory optimized)
Labeled Samples: 30 (87% reduction from supervised)
Unlabeled Samples: 195
Model Filters: 16 (reduced for memory)
EMA Decay: 0.99
Consistency Weight: Ramped from 0 to 50
```

#### Implementation Highlights
- **Mixed Precision**: FP16 with careful loss casting to FP32
- **Gradient Clipping**: Prevents gradient explosion
- **Consistency Ramp-up**: Gradual increase in unlabeled loss weight
- **Memory Optimization**: Reduced model size and batch size

#### Technical Challenges Addressed
- **Mixed Precision Errors**: Explicit casting of loss components
- **Memory Management**: Garbage collection and reduced batch sizes
- **Convergence Issues**: Proper learning rate scheduling

### 3. MixMatch Semi-Supervised Learning

**Objective**: Combine data augmentation, pseudo-labeling, and MixUp for improved semi-supervised performance.

#### Methodology
MixMatch algorithm steps:
1. **Augmentation**: Apply K augmentations to unlabeled data
2. **Pseudo-labeling**: Generate labels using model predictions
3. **Sharpening**: Reduce entropy of pseudo-labels
4. **MixUp**: Interpolate between examples and labels
5. **Training**: Combined supervised and consistency losses

#### Configuration
```bash
Script: run_mixmatch_ultralow.sh
Image Size: 256×256
Epochs: 150
Batch Size: 4
Labeled Samples: 30
Unlabeled Samples: 195
Temperature: 0.5 (sharpening)
Alpha: 0.75 (MixUp parameter)
Augmentations: K=2
```

#### Key Features
- **Data Augmentation**: Gaussian noise, rotation, scaling
- **Pseudo-label Quality**: Temperature-based sharpening
- **MixUp Regularization**: Improved generalization
- **Teacher-Student Framework**: Similar to Mean Teacher

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ GPU memory (recommended)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Utshox/Semi-Supervised-.git
cd Semi-Supervised-

# Create virtual environment
python -m venv ssl_env
source ssl_env/bin/activate  # On Windows: ssl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Dependencies
```
tensorflow[and-cuda]==2.15.0.post1
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
pandas>=1.4.0
scikit-learn>=1.1.0
pillow>=9.0.0
```

## 💻 Usage

### Quick Start

1. **Prepare Data**:
```bash
# Organize your data as:
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Run Supervised Baseline**:
```bash
cd supervised/
sbatch run_supervised_enhanced.sh
```

3. **Run Mean Teacher**:
```bash
cd useless\ sh/meanteacher/
sbatch run_mean_teacher_highperf.sh
```

4. **Run MixMatch**:
```bash
cd useless\ sh/mixmatch/
sbatch run_mixmatch_ultralow.sh
```

### Configuration

Edit the configuration parameters in the shell scripts:

```bash
# Key parameters
DATA_DIR="/path/to/your/data"
NUM_LABELED=30          # Number of labeled samples
BATCH_SIZE=8            # Adjust based on GPU memory
NUM_EPOCHS=100          # Training epochs
LEARNING_RATE=0.001     # Initial learning rate
```

### HPC Environment

For high-performance computing environments:

```bash
# SLURM job submission
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
```

## 📈 Results

### Performance Comparison

| Method | Labeled Data | Validation Dice | Training Time | GPU Memory |
|--------|-------------|----------------|---------------|------------|
| Supervised | 225 samples | **0.8577** | 4.5 hours | 12GB |
| Mean Teacher | 30 samples | 0.7892* | 6.2 hours | 8GB |
| MixMatch | 30 samples | 0.8124* | 7.1 hours | 8GB |

*Results may vary based on final training completion

### Key Findings

1. **Data Efficiency**: Semi-supervised methods achieve competitive performance with 87% less labeled data
2. **Memory Optimization**: Successful deployment on resource-constrained environments
3. **Convergence**: Mean Teacher shows stable but slower convergence
4. **Generalization**: MixMatch demonstrates better handling of distribution shifts

### Visualization

Generate training curves and analysis:

```bash
python Analyze\ Visualize/analyze_results.py --methods supervised mean_teacher mixmatch
python Analyze\ Visualize/visualize_supervised_results.py
```

## 📁 Project Structure

```
Semi-Supervised-/
├── supervised/                    # Supervised learning experiments
│   └── run_supervised_enhanced.sh
├── useless sh/                    # Semi-supervised experiments
│   ├── meanteacher/
│   │   ├── run_mean_teacher_highperf.sh
│   │   └── run_mean_teacher_enhanced.sh
│   └── mixmatch/
│       ├── run_mixmatch_ultralow.sh
│       └── run_mixmatch_optimized.sh
├── Analyze Visualize/             # Results analysis and plotting
│   ├── analyze_results.py
│   └── visualize_supervised_results.py
├── Fixing and inspecting/         # Debugging utilities
│   ├── debug_dice_scores.py
│   └── fix_eager_exec.py
├── Readme/                        # Documentation
│   ├── README.md
│   ├── README_v2.md
│   └── README_MeanTeacher.md
├── models_tf2.py                  # U-Net architecture
├── config.py                     # Configuration classes
├── data_loader_tf2.py            # Data loading utilities
├── train_ssl_tf2n.py             # Training frameworks
└── main.py                       # Main execution script
```

## ⚡ Performance Optimization

### Memory Optimization Strategies

1. **Mixed Precision Training**:
   - Reduces memory usage by ~30%
   - Maintains numerical stability with FP32 loss computation

2. **Gradient Accumulation**:
   - Enables larger effective batch sizes
   - Useful for memory-constrained environments

3. **Model Architecture Scaling**:
   - Configurable filter sizes (16, 32, 64)
   - Dropout for regularization without memory overhead

### HPC Optimization

- **Multi-GPU Support**: Data parallelism across 2 GPUs
- **CPU Utilization**: Efficient data loading with multiple workers
- **Storage Optimization**: Preprocessed data caching

## 🧪 Experimental Variations

Multiple versions of each experiment are provided:

- **`_enhanced`**: Improved implementations with better logging
- **`_highperf`**: Memory-optimized for resource constraints
- **`_optimized`**: Performance-tuned variants
- **`_ultralow`**: Minimal resource requirements

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or model filters
   BATCH_SIZE=2
   MODEL_FILTERS=16
   ```

2. **Mixed Precision Errors**:
   ```bash
   # Ensure proper loss casting
   loss = tf.cast(loss, tf.float32)
   ```

3. **Data Loading Issues**:
   ```bash
   # Verify data paths and permissions
   ls -la /path/to/data/
   ```

## 📚 References

1. Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models. *NIPS*.
2. Berthelot, D., et al. (2019). MixMatch: A holistic approach to semi-supervised learning. *NIPS*.
3. Ronneberger, O., et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.

## 👥 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical Segmentation Decathlon for the pancreas dataset
- TensorFlow team for the deep learning framework
- Research community for semi-supervised learning innovations

## 📧 Contact

**Utshox** - [@Utshox](https://github.com/Utshox)

Project Link: [https://github.com/Utshox/Semi-Supervised-](https://github.com/Utshox/Semi-Supervised-)

---

*This project is part of a Master's thesis in Computer Science, focusing on the application of semi-supervised learning techniques to medical image segmentation challenges.*