import tensorflow as tf
from pathlib import Path
import argparse
import time
from datetime import datetime

# from visualization import generate_report_visualizations  # Import if you have this module
from config import StableSSLConfig
from train_ssl_tf2n import ImprovedSSLTrainer, MixMatchTrainer, HighPerformanceSSLTrainer, StableSSLTrainer
from data_loader_tf2 import DataPipeline  # Import DataPipeline


def setup_gpu():
    """Setup GPU for training"""
    print("Setting up GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU found")
        return False

def prepare_data_paths(data_dir, num_labeled=2, num_validation=63):
    """Prepare data paths"""
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

    # Create splits with explicit slicing
    train_images = all_image_paths[:num_labeled]  # Take exactly num_labeled images
    train_labels = all_label_paths[:num_labeled]  # Take exactly num_labeled labels

    val_images = all_image_paths[-num_validation:]  # Take last num_validation images
    val_labels = all_label_paths[-num_validation:]  # Take last num_validation labels

    # Rest for unlabeled (excluding validation set)
    unlabeled_images = all_image_paths[num_labeled:-num_validation]

    # Print split sizes
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
                        help='Path to data directory', default='/content/drive/MyDrive/Local_contrastive_loss_data/Task07_Pancreas/cropped')
    parser.add_argument('--method', type=str, choices=['mean_teacher', 'mixmatch'],
                        default='mean_teacher', help='Which method to run')
    args = parser.parse_args()

    # Setup
    gpu_available = setup_gpu()
    config = StableSSLConfig()

    # Ensure proper dimensions
    config.img_size_x = 512 # You can change this to 512 if your GPU has enough memory
    config.img_size_y = 512 # You can change this to 512 if your GPU has enough memory
    config.num_channels = 1
    config.batch_size = 16 

    # Prepare data
    data_paths = prepare_data_paths(args.data_dir)

    # Create DataPipeline
    data_pipeline = DataPipeline(config)
    datasets = data_pipeline.setup_training_data(
        data_paths['labeled'],
        data_paths['unlabeled'],
        data_paths['validation'],
        batch_size=config.batch_size
    )

    train_ds = datasets['labeled']
    val_ds = datasets['validation']

    # Create trainer
    if args.method == 'mean_teacher':
        print("\nRunning Mean Teacher...")
        trainer = StableSSLTrainer(config, data_pipeline) # Pass the data_pipeline to the trainer
    else:
        print("\nRunning MixMatch...")
        trainer = MixMatchTrainer(config, data_pipeline) # Pass the data_pipeline to the trainer

    # Print debug info
    print("\nInitial model configuration:")
    print(f"Learning rate: {trainer.optimizer.get_config()['learning_rate']}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")

    # Train
    trainer.train(data_paths, train_ds, val_ds)  # Pass the datasets to the train method

    print("\nTraining completed!")

if __name__ == '__main__':
    main()