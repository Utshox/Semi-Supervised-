import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

def visualize_mixmatch_history(history_file_path, output_dir=None):
    """
    Visualize training metrics from a saved MixMatch model history file (.npy format)
    
    Args:
        history_file_path (str): Path to the .npy history file
        output_dir (str, optional): Directory to save output plots. If None, saves in the same directory as history file
    """
    print(f"Loading history from: {history_file_path}")
    
    # Check if the file exists
    if not os.path.exists(history_file_path):
        print(f"Error: File {history_file_path} not found")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(history_file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the history file
    try:
        history = np.load(history_file_path, allow_pickle=True).item()
        print(f"Successfully loaded history file")
        print(f"Available metrics: {list(history.keys())}")
    except Exception as e:
        print(f"Error loading history file: {e}")
        return
    
    # Create plots
    create_training_plots(history, output_dir)
    
def create_training_plots(history, output_dir):
    """Create various training plots from the history dictionary"""
    
    # Plot training and validation metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot dice scores
    ax = axes[0, 0]
    if 'val_dice' in history:
        ax.plot(history['val_dice'], label='Validation Dice', color='blue')
    if 'teacher_dice' in history:
        ax.plot(history['teacher_dice'], label='Teacher Dice', color='red')
    ax.set_title('Dice Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    
    # Plot loss
    ax = axes[0, 1]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Training Loss', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation Loss', color='red')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot other metrics if available
    ax = axes[1, 0]
    if 'consistency_loss' in history:
        ax.plot(history['consistency_loss'], label='Consistency Loss', color='green')
    if 'ssl_loss' in history:
        ax.plot(history['ssl_loss'], label='SSL Loss', color='purple')
    ax.set_title('Semi-Supervised Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot lr if available
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(history['lr'], label='Learning Rate', color='orange')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixmatch_training_metrics.png'), dpi=300)
    print(f"Saved plot to {os.path.join(output_dir, 'mixmatch_training_metrics.png')}")
    plt.close()
    
    # Create a separate high-resolution plot for Dice score
    if 'val_dice' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['val_dice'], label='Validation Dice', color='blue', linewidth=2)
        if 'teacher_dice' in history:
            plt.plot(history['teacher_dice'], label='Teacher Dice', color='red', linewidth=2)
        
        plt.title('MixMatch Segmentation Dice Score', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Dice Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mixmatch_dice_score.png'), dpi=300)
        print(f"Saved Dice score plot to {os.path.join(output_dir, 'mixmatch_dice_score.png')}")
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MixMatch training history')
    parser.add_argument('--history_file', type=str, default='/smm/v14/mixmatch_results/checkpoints/20250501_035426/best_mixmatch_model_history.npy',
                        help='Path to the history .npy file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output plots (default: same as history file)')
    
    args = parser.parse_args()
    visualize_mixmatch_history(args.history_file, args.output_dir)