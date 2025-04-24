#!/usr/bin/env python3
"""
Script to generate visualizations for trained models for MS thesis
This script will create comprehensive visualizations for all three methods:
- Supervised learning
- Mean Teacher
- MixMatch
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime

from visualization import (
    generate_report_visualizations,
    visualize_comparison,
    visualize_3D,
    calculate_metrics, 
    compare_methods,
    plot_metrics_over_time
)
from config import StableSSLConfig
from train_ssl_tf2n import ImprovedSSLTrainer, MixMatchTrainer, StableSSLTrainer, SupervisedTrainer
from data_loader_tf2 import DataPipeline
from main import prepare_data_paths, setup_gpu

# Force TensorFlow to allow memory growth for GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def setup_environment():
    """Setup GPU and print TensorFlow version information"""
    print(f"TensorFlow version: {tf.__version__}")
    gpu_available = setup_gpu()
    print(f"GPU available: {gpu_available}")
    
    # Print number of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs: {len(gpus)}")
    
    # Return physical devices for later use
    return gpus

def load_model(model_path, method):
    """Load a trained model"""
    print(f"Loading model from {model_path}")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model path {model_path} does not exist")
        return None

def load_trainer(method, config, data_pipeline, checkpoint_dir=None):
    """Load the appropriate trainer based on method"""
    print(f"Initializing trainer for {method}...")
    
    if method == 'supervised':
        trainer = SupervisedTrainer(config, data_pipeline)
    elif method == 'mean_teacher':
        trainer = StableSSLTrainer(config, data_pipeline)
    elif method == 'mixmatch':
        trainer = MixMatchTrainer(config, data_pipeline)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # If a checkpoint directory is provided, try to load from it
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        try:
            print(f"Loading trainer checkpoint from {checkpoint_dir}")
            trainer.load_checkpoint(checkpoint_dir)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return trainer

def generate_visualizations_for_method(method, data_dir, output_dir, config, model_dir=None):
    """Generate visualizations for a single method"""
    print(f"Generating visualizations for {method}...")
    
    # Create output directory
    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)
    
    # Prepare data paths
    data_paths = prepare_data_paths(Path(data_dir), method=method)
    
    # Setup data pipeline
    data_pipeline = DataPipeline(config)
    datasets = data_pipeline.setup_training_data(
        data_paths['labeled'],
        data_paths['unlabeled'],
        data_paths['validation'],
        batch_size=config.batch_size
    )
    
    # Load trainer
    trainer = load_trainer(method, config, data_pipeline, model_dir)
    
    # Generate visualizations
    trainer.val_dataset = datasets['validation']  # Ensure val_dataset is set
    metrics = generate_report_visualizations(trainer, method_output_dir, method)
    
    print(f"Visualizations for {method} saved to {method_output_dir}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for MS thesis')
    parser.add_argument('--data_dir', type=str, default='/scratch/lustre/home/mdah0000/images/cropped',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./thesis_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory containing trained models')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['supervised', 'mean_teacher', 'mixmatch'],
                        help='Methods to visualize')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for prediction')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Generate comparison visualizations between methods')
    args = parser.parse_args()
    
    # Setup environment
    gpus = setup_environment()
    
    # Setup config
    print("Initializing config...")
    config = StableSSLConfig()
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = args.batch_size
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving visualizations to {args.output_dir}")
    
    # Generate visualizations for each method
    all_metrics = {}
    
    for method in args.methods:
        method_model_dir = os.path.join(args.model_dir, method)
        metrics = generate_visualizations_for_method(
            method, 
            args.data_dir,
            args.output_dir,
            config,
            method_model_dir
        )
        all_metrics[method] = metrics
    
    # If requested, generate comparison visualizations between methods
    if args.compare_methods and len(args.methods) > 1:
        print("Generating method comparisons...")
        comparison_dir = os.path.join(args.output_dir, 'method_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Load validation data once for comparison
        print("Loading validation data for comparison...")
        data_paths = prepare_data_paths(Path(args.data_dir), method='supervised')
        data_pipeline = DataPipeline(config)
        datasets = data_pipeline.setup_training_data(
            data_paths['labeled'], 
            data_paths['unlabeled'],
            data_paths['validation'],
            batch_size=config.batch_size
        )
        
        # Collect images, ground truths, and predictions for each method
        val_images = []
        val_masks = []
        predictions_dict = {method: [] for method in args.methods}
        
        # Get trainers for each method
        trainers = {}
        for method in args.methods:
            method_model_dir = os.path.join(args.model_dir, method)
            trainers[method] = load_trainer(method, config, data_pipeline, method_model_dir)
        
        # Get predictions from each method for comparison
        for i, (images, masks) in enumerate(datasets['validation'].take(10)):
            val_images.extend(images.numpy())
            val_masks.extend(masks.numpy())
            
            for method, trainer in trainers.items():
                preds = trainer.model.predict(images)
                predictions_dict[method].extend(preds)
                
        # Compare methods and save visualizations
        compare_methods(
            val_images,
            val_masks,
            predictions_dict,
            all_metrics,
            comparison_dir
        )
        
        print(f"Method comparison visualizations saved to {comparison_dir}")
    
    print("Visualization generation completed!")
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Starting visualization generation at {start_time}")
    
    try:
        main()
        print(f"Visualization generation completed successfully in {datetime.now() - start_time}")
    except Exception as e:
        print(f"Error during visualization generation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Process finished at: {datetime.now()}")