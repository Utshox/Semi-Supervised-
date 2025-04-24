import tensorflow as tf
import os
from pathlib import Path
import argparse
import time
from datetime import datetime
import sys

# from visualization import generate_report_visualizations
from config import StableSSLConfig
from train_ssl_tf2n import ImprovedSSLTrainer, MixMatchTrainer, HighPerformanceSSLTrainer, StableSSLTrainer, SupervisedTrainer
from data_loader_tf2 import DataPipeline  # Import DataPipeline

def setup_gpu():
    """Setup GPU for training with better detection and error handling"""
    print("Setting up GPU...")
    
    # Force TensorFlow to allow growth and optimize for memory
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Try nvidia-smi first to confirm hardware is present
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("NVIDIA GPU detected through nvidia-smi!")
        else:
            print("No NVIDIA GPU detected by nvidia-smi.")
            return False
    except Exception as e:
        print(f"Error checking GPU with nvidia-smi: {e}")
        return False
    
    # Print TensorFlow version for debugging
    print(f"Using TensorFlow version: {tf.__version__}")
    
    # Check CUDA environment variables
    print("CUDA environment check:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Test for CUDA libraries
    try:
        import subprocess
        cuda_lib_paths = subprocess.run('find /usr -name "libcudart.so*" 2>/dev/null', 
                                       shell=True, stdout=subprocess.PIPE, text=True).stdout
        print(f"CUDA runtime libraries found: {cuda_lib_paths or 'None'}")
    except Exception as e:
        print(f"Error checking CUDA libraries: {e}")
    
    # Limit GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s) and enabled memory growth")
            
            # Set memory limit per GPU to avoid OOM errors
            try:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=24000)]  # 24GB limit (adjust as needed)
                    )
                print("Set memory limits for GPUs")
            except Exception as e:
                print(f"Warning: Could not set memory limits: {e}")
            
            # Set mixed precision policy for better memory efficiency
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Using {policy.name} precision policy for memory efficiency")
            except Exception as e:
                print(f"Warning: Could not set mixed precision policy: {e}")
            
            # Test direct GPU placement
            try:
                with tf.device('/device:GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                    result = c.numpy()  # Force execution
                    print("GPU test result:", result)
                    print("GPU is accessible via direct device placement!")
                    return True
            except Exception as e:
                print(f"Error using GPU via direct placement: {e}")
                print("Trying alternate GPU detection method...")
                
                # Try alternate device specification
                try:
                    with tf.device('/GPU:0'):
                        tf_test = tf.constant([1.0])
                        _ = tf_test.numpy()  # Force execution
                        print("Basic GPU test passed!")
                        return True
                except Exception as e2:
                    print(f"Basic GPU test failed: {e2}")
        except RuntimeError as e:
            print(f"Error setting up GPUs: {e}")
    else:
        print("No GPUs found in tf.config.list_physical_devices('GPU')")
        # Try to diagnose the issue
        print("Diagnostic information:")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        try:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
        except Exception as e:
            print(f"Could not list local devices: {e}")
    
    # As a final fallback, try a very simple check
    try:
        gpu_devices = tf.config.list_logical_devices('GPU')
        if len(gpu_devices) > 0:
            print(f"Found {len(gpu_devices)} logical GPU devices")
            return True
    except Exception as e:
        print(f"Error checking logical GPU devices: {e}")
    
    print("WARNING: Could not configure GPU. Training will proceed on CPU, which will be much slower.")
    return False

def prepare_data_paths(data_dir, num_labeled=15, num_validation=63, method='mean_teacher'):
    """Prepare data paths with adjusted splits based on method"""
    def get_case_number(path):
        return int(str(path).split('pancreas_')[-1][:3])

    print("Finding data pairs...")
    all_image_paths = []
    all_label_paths = []

    for folder in sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x)):
        folder_no_nii = str(folder).replace('.nii', '')
        img_path = Path(folder_no_nii) / 'img_cropped.npy'
        mask_path = Path(folder_no_nii) / 'mask_cropped.npy'

        if img_path.exists() and mask_path.exists():
            all_image_paths.append(str(img_path))
            all_label_paths.append(str(mask_path))

    print(f"\nTotal pairs found: {len(all_image_paths)}")
    
    # For supervised learning, use all available data except validation set
    if method == 'supervised':
        # Reserve validation data
        val_images = all_image_paths[-num_validation:]
        val_labels = all_label_paths[-num_validation:]
        
        # Use all remaining data as labeled training data
        train_images = all_image_paths[:-num_validation]  
        train_labels = all_label_paths[:-num_validation]
        
        # Empty unlabeled set (not used in supervised learning)
        unlabeled_images = []
        
        print(f"Supervised training mode: Using all available labeled data")
        print(f"Labeled training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
    else:
        # For semi-supervised methods, use the original split
        train_images = all_image_paths[:num_labeled]
        train_labels = all_label_paths[:num_labeled]
        val_images = all_image_paths[-num_validation:]
        val_labels = all_label_paths[-num_validation:]
        unlabeled_images = all_image_paths[num_labeled:-num_validation]
        
        print(f"Semi-supervised training mode ({method})")
        print(f"Labeled training images: {len(train_images)}")
        print(f"Unlabeled images: {len(unlabeled_images)}")
        print(f"Validation images: {len(val_images)}")

    return {
        'labeled': {
            'images': train_images,
            'labels': train_labels,
        },
        'unlabeled': {
            'images': unlabeled_images
        },
        'validation': {
            'images': val_images,
            'labels': val_labels,
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                        help='Path to data directory', default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--method', type=str, choices=['supervised', 'mean_teacher', 'mixmatch'],
                        default='mean_teacher', help='Which method to run')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    args = parser.parse_args()

    print(f"Starting {args.method} method with batch size {args.batch_size}...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    sys.stdout.flush()  # Force flush stdout

    # Setup
    print("Setting up GPU environment...")
    sys.stdout.flush()
    gpu_available = setup_gpu()
    print(f"GPU available: {gpu_available}")
    sys.stdout.flush()
    
    print("Initializing config...")
    sys.stdout.flush()
    config = StableSSLConfig()

    # Ensure proper dimensions
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = args.batch_size
    print(f"Config initialized: img_size={config.img_size_x}x{config.img_size_y}, batch_size={config.batch_size}")
    sys.stdout.flush()

    # Prepare data
    print("Preparing data paths...")
    sys.stdout.flush()
    data_paths = prepare_data_paths(args.data_dir, method=args.method)
    print("Data paths prepared.")
    sys.stdout.flush()

    # Create DataPipeline
    print("Initializing data pipeline...")
    sys.stdout.flush()
    try:
        data_pipeline = DataPipeline(config)
        print("Setting up training data...")
        sys.stdout.flush()
        datasets = data_pipeline.setup_training_data(
            data_paths['labeled'],
            data_paths['unlabeled'],
            data_paths['validation'],
            batch_size=config.batch_size
        )
        print("Training data setup complete.")
        sys.stdout.flush()

        train_ds = datasets['labeled']
        val_ds = datasets['validation']
        
        # Print dataset info
        print(f"Training dataset batch size: {config.batch_size}")
        try:
            for batch in train_ds.take(1):
                print(f"Sample batch shape: {batch[0].shape}")
            print("Successfully loaded a batch from training data")
        except Exception as e:
            print(f"Error checking training batch: {e}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error setting up data pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # Create trainer
    print(f"\nInitializing {args.method} trainer...")
    sys.stdout.flush()
    try:
        if args.method == 'mean_teacher':
            trainer = StableSSLTrainer(config, data_pipeline)
        elif args.method == 'supervised':
            trainer = SupervisedTrainer(config, data_pipeline)
        else:
            trainer = MixMatchTrainer(config, data_pipeline)
        
        print("Trainer initialized successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # Print debug info
    print("\nModel configuration:")
    print(f"Learning rate: {trainer.optimizer.get_config()['learning_rate']}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    sys.stdout.flush()

    # Train
    print("\nStarting training process...")
    sys.stdout.flush()
    try:
        trainer.train(data_paths)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    sys.stdout.flush()
    print(f"Process finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()