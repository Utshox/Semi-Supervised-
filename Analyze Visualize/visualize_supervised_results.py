#!/usr/bin/env python3
# Visualization script for supervised learning results
# This script can be run at any time to visualize current progress, even if training is incomplete

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import csv
from datetime import datetime
import re

def parse_log_file(log_file):
    """Parse a CSV log file and extract training metrics"""
    epochs = []
    losses = []
    val_dices = []
    learning_rates = []
    times = []
    
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 5:  # Ensure we have all expected columns
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))
                    val_dices.append(float(row[2]))
                    learning_rates.append(float(row[3]))
                    times.append(float(row[4]))
    except Exception as e:
        print(f"Error parsing log file {log_file}: {e}")
        return None
    
    return {
        'epochs': epochs,
        'losses': losses, 
        'val_dices': val_dices, 
        'learning_rates': learning_rates,
        'times': times
    }

def parse_slurm_output(output_file):
    """Parse a slurm output file to extract training metrics when log files aren't available"""
    epochs = []
    losses = []
    val_dices = []
    learning_rates = []
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
            # Regular expressions to match the training output lines
            epoch_pattern = re.compile(r'Epoch (\d+)/\d+')
            metrics_pattern = re.compile(r'Time: (\d+\.\d+)s \| Loss: (\d+\.\d+) \| Val Dice: (\d+\.\d+) \| LR: (\d+\.\d+e[-+]?\d+)')
            
            current_epoch = None
            
            for line in lines:
                # Check for epoch start
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # Check for metrics line
                metrics_match = metrics_pattern.search(line)
                if metrics_match and current_epoch is not None:
                    # time = float(metrics_match.group(1))
                    loss = float(metrics_match.group(2))
                    val_dice = float(metrics_match.group(3))
                    lr = float(metrics_match.group(4))
                    
                    epochs.append(current_epoch)
                    losses.append(loss)
                    val_dices.append(val_dice)
                    learning_rates.append(lr)
    except Exception as e:
        print(f"Error parsing slurm output file {output_file}: {e}")
        return None
    
    return {
        'epochs': epochs,
        'losses': losses,
        'val_dices': val_dices,
        'learning_rates': learning_rates
    }

def find_best_checkpoint(results_dir):
    """Find the best validation Dice score from the log files."""
    best_dice = 0.0
    best_epoch = -1
    log_dir = Path(results_dir) / 'logs'
    
    try:
        log_files = glob.glob(f"{log_dir}/*.csv")
        if not log_files:
            print(f"No log files found in {log_dir}")
            # Try parsing slurm output as a fallback
            output_files = glob.glob(f"{Path(results_dir).parent}/supervised*.out") # Look in parent dir for slurm logs
            output_files.sort(key=os.path.getmtime, reverse=True)
            if output_files:
                print(f"Attempting to parse Slurm output file: {output_files[0]}")
                metrics = parse_slurm_output(output_files[0])
                if metrics and metrics.get('val_dices'):
                    best_dice = max(metrics['val_dices'])
                    best_epoch = metrics['epochs'][metrics['val_dices'].index(best_dice)]
            else:
                 print(f"No Slurm output files found matching pattern 'supervised*.out' in {Path(results_dir).parent}")

        else:
            log_files.sort(key=os.path.getmtime, reverse=True)
            latest_log = log_files[0]
            print(f"Parsing latest log file for best score: {latest_log}")
            metrics = parse_log_file(latest_log)
            if metrics and metrics.get('val_dices'):
                best_dice = max(metrics['val_dices'])
                best_epoch = metrics['epochs'][metrics['val_dices'].index(best_dice)]

        if best_epoch != -1:
            print(f"Best validation Dice found in logs: {best_dice:.4f} at epoch {best_epoch}")
            # Note: Checkpoint filenames might be like 'epoch_{best_epoch}_...' or similar.
        else:
            print("Could not determine best validation Dice from log or Slurm output files.")
            
    except Exception as e:
        print(f"Error finding best score from logs: {e}")
    
    return best_dice, best_epoch
                    
def visualize_training_history(results_dir, output_dir=None):
    """Generate visualization plots from training logs"""
    if output_dir is None:
        output_dir = Path(results_dir) / 'visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")
    
    # Look for log files
    log_files = glob.glob(f"{results_dir}/logs/*.csv")
    metrics = None
    
    if log_files:
        # Use the most recent log file
        log_files.sort(key=os.path.getmtime, reverse=True)
        latest_log = log_files[0]
        print(f"Using log file: {latest_log}")
        metrics = parse_log_file(latest_log)
    else:
        # If no log files, try to parse from slurm output files in parent directory
        # Use a pattern that might match your supervised job output name
        output_files = glob.glob(f"{Path(results_dir).parent}/supervised*.out") 
        if output_files:
            output_files.sort(key=os.path.getmtime, reverse=True)
            latest_output = output_files[0]
            print(f"No log files found. Using slurm output file: {latest_output}")
            metrics = parse_slurm_output(latest_output)
        else:
            print(f"No log files found in {results_dir}/logs and no Slurm output files found matching 'supervised*.out' in {Path(results_dir).parent}")
            return
    
    if not metrics or not metrics['epochs']:
        print("No training metrics found in the log files")
        return
    
    # Generate plots
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epochs'], metrics['losses'], 'b-', label='Training Loss')
    plt.title('Training Loss Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    
    # Plot 2: Validation Dice Score
    plt.subplot(2, 2, 2)
    plt.plot(metrics['epochs'], metrics['val_dices'], 'g-', label='Validation Dice')
    
    # Add horizontal line for best dice score if available
    if metrics['val_dices']:
        best_dice = max(metrics['val_dices'])
        best_epoch = metrics['epochs'][metrics['val_dices'].index(best_dice)]
        plt.axhline(y=best_dice, color='r', linestyle='--', 
                    label=f'Best Dice: {best_dice:.4f} (Epoch {best_epoch})')
    else:
        best_dice = 0.0
        best_epoch = -1
    
    plt.title('Validation Dice Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(metrics['epochs'], metrics['learning_rates'], 'r-')
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 4: Epoch Duration (if available)
    if 'times' in metrics and metrics['times']:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['epochs'], metrics['times'], 'm-')
        plt.title('Epoch Duration', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True)
    
    # Add summary statistics
    if metrics['epochs']: # Check if metrics were loaded
        plt.figtext(0.5, 0.01, 
                    f"Summary: Epochs: {max(metrics['epochs'])}, Best Dice: {best_dice:.4f} (Epoch {best_epoch})\n"
                    f"Final Loss: {metrics['losses'][-1]:.4f}, Final Dice: {metrics['val_dices'][-1]:.4f}",
                    ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    else:
         plt.figtext(0.5, 0.01, "Summary: No data found", ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Supervised Learning Training Metrics', fontsize=16, y=0.98)
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/training_metrics_{timestamp}.png", dpi=150)
    plt.savefig(f"{output_dir}/latest_training_metrics.png", dpi=150)
    print(f"Saved training metrics visualization to {output_dir}")
    plt.close()
    
    # Create a progress summary file
    summary_file = f"{output_dir}/training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Supervised Learning Training Summary ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs completed: {max(metrics['epochs'])}\n")
        f.write(f"Best validation Dice score: {best_dice:.4f} (Epoch {best_epoch})\n")
        f.write(f"Final validation Dice score: {metrics['val_dices'][-1]:.4f}\n")
        f.write(f"Final training loss: {metrics['losses'][-1]:.4f}\n")
        
        # Calculate average epoch time if available
        if 'times' in metrics and metrics['times']:
            avg_time = sum(metrics['times']) / len(metrics['times'])
            f.write(f"Average epoch duration: {avg_time:.2f} seconds\n")
            f.write(f"Estimated total training time: {sum(metrics['times']):.2f} seconds "
                    f"({sum(metrics['times'])/3600:.2f} hours)\n")
        
        # Check if training appears to have finished
        if 'val_dices' in metrics and len(metrics.get('val_dices', [])) >= 2:
            # ... rest of status check ...
            pass # Keep existing logic
        elif not metrics.get('epochs'):
             f.write("\nTraining Status: No training data found.\n")
        else:
             f.write("\nTraining Status: Insufficient data for analysis.\n")
                
    print(f"Saved training summary to {summary_file}")
    
    # Return the metrics for further analysis
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Visualize supervised learning results')
    parser.add_argument('--results_dir', type=str, 
                        default='/scratch/lustre/home/mdah0000/smm/v14/supervised_results', # Corrected default path
                        help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (default: results_dir/visualizations)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best checkpoint score from logs
    print(f"\nAnalyzing results in: {results_dir}")
    best_dice, best_epoch = find_best_checkpoint(results_dir)
    
    # Generate visualizations
    metrics = visualize_training_history(results_dir, output_dir)
    
    if metrics and metrics.get('epochs'): # Check if metrics were loaded
        print("\nTraining Summary:")
        print(f"- Number of epochs completed: {len(metrics['epochs'])}")
        # Use the best_dice/best_epoch found earlier, as metrics might be incomplete if parsed from Slurm
        if best_epoch != -1:
             print(f"- Best validation Dice score: {best_dice:.4f} (found at Epoch {best_epoch})")
        else:
             print("- Best validation Dice score: Not determined")
        print(f"- Final validation Dice score: {metrics['val_dices'][-1]:.4f}")
        print(f"- Final training loss: {metrics['losses'][-1]:.4f}")
    else:
        print("\nCould not generate training summary.")

if __name__ == "__main__":
    main()