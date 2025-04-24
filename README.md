# Semi-Supervised Medical Image Segmentation

A TensorFlow 2.x implementation of semi-supervised learning methods for medical image segmentation, specifically focused on pancreas segmentation in CT scans.

## Features

- Supports multiple semi-supervised learning methods:
  - Supervised Learning (baseline)
  - Mean Teacher
  - MixMatch
- GPU acceleration with Tesla V100 support
- Configurable data augmentation pipeline
- Memory-efficient processing of 3D medical volumes

## Requirements

- Python 3.10+
- TensorFlow 2.15.0.post1 (with CUDA support)
- CUDA-compatible GPU (tested with Tesla V100)
- Other dependencies listed in `requirements.txt`

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Semi-Supervised-
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Running Training Jobs with SLURM

The project is configured for SLURM-based HPC environments. Use the provided scripts:

- Supervised learning:
  ```bash
  sbatch run_supervised.sh
  ```

- Mean Teacher:
  ```bash
  sbatch run_mean_teacher.sh
  ```

- MixMatch:
  ```bash
  sbatch run_mixmatch.sh
  ```

- Run all methods in sequence:
  ```bash
  ./run_all_methods.sh
  ```

#### Manual Execution

You can run the training manually with:

```bash
python main.py --method supervised --data_dir /path/to/data --batch_size 4
```

Available methods: `supervised`, `mean_teacher`, `mixmatch`

## Project Structure

- `main.py`: Entry point for training
- `models_tf2.py`: Model architecture definitions
- `data_loader_tf2.py`: Data loading and augmentation pipeline
- `train_ssl_tf2n.py`: Implementation of semi-supervised learning methods
- `config.py`: Configuration parameters
- `run_*.sh`: SLURM job submission scripts

## Data Format

The system expects:
- Input images as 3D numpy arrays (.npy files)
- Segmentation masks as binary 3D numpy arrays (.npy files)
- Data directory structure should contain separate folders for labeled images, labels, and unlabeled images

## Monitoring

When using the provided SLURM scripts, you can monitor GPU usage and training progress in the logs directory:
```bash
tail -f logs/supervised_training_<job_id>.log
```

## GPU Memory Optimization

The system is optimized for Tesla V100 GPUs with 32GB memory. To avoid out-of-memory errors:
- Use `TF_FORCE_GPU_ALLOW_GROWTH=true` environment variable
- Set appropriate batch size (4 for 2 GPUs, 2 for 1 GPU)
- Image resolution can be adjusted in the config.py file

## Citation

If you use this code in your research, please cite:
```
# Add your citation here
```

## License

[Add License Information]