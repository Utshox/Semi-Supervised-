#!/usr/bin/env python3
"""
Results analysis script for semi-supervised learning MS thesis.
This script analyzes and compares the performance of three methods:
- Supervised learning
- Mean Teacher semi-supervised
- MixMatch semi-supervised
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import glob
import re

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

def find_latest_experiment(base_dir, method):
    """Find the most recent experiment directory for a given method"""
    import glob
    import os
    
    # First try the original pattern
    if method == 'supervised':
        patterns = [
            os.path.join(base_dir, "supervised_*_*"),
            os.path.join(base_dir, "experiment_results/supervised_*"),
            os.path.join(base_dir, "experiment_results/supervised_supervised_*")
        ]
    elif method == 'mean_teacher':
        patterns = [
            os.path.join(base_dir, "mean_teacher_*_*"),
            os.path.join(base_dir, "experiment_results/mean_teacher_*"),
            os.path.join(base_dir, "experiment_results/supervised_mean_teacher_*")
        ]
    elif method == 'mixmatch':
        patterns = [
            os.path.join(base_dir, "mixmatch_*_*"),
            os.path.join(base_dir, "experiment_results/mixmatch_*"),
            os.path.join(base_dir, "experiment_results/supervised_mixmatch_*")
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Try all patterns
    all_dirs = []
    for pattern in patterns:
        matched_dirs = glob.glob(pattern)
        all_dirs.extend(matched_dirs)
    
    # Sort by modification time (most recent first)
    all_dirs.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    
    if all_dirs:
        print(f"Found experiment directory for {method}: {all_dirs[0]}")
        return all_dirs[0]
    
    print(f"No results directory found for {method}. Tried patterns: {patterns}")
    return None

def load_results(results_dir, method):
    """Load results for a specific method"""
    try:
        # Try to load numpy history file
        history_path = os.path.join(results_dir, "training_history.npy")
        if os.path.exists(history_path):
            history = np.load(history_path, allow_pickle=True).item()
            return history
            
        # If no .npy file, try a CSV file
        csv_path = os.path.join(results_dir, "training_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return {col: df[col].tolist() for col in df.columns}
        
        print(f"No history file found for {method} in {results_dir}")
        return None
    except Exception as e:
        print(f"Error loading results for {method}: {e}")
        return None
        
def plot_learning_curves(histories, save_dir):
    """Plot learning curves for all methods"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = {
        'supervised': 'blue',
        'mean_teacher': 'green',
        'mixmatch': 'red'
    }
    
    # Plot training loss
    ax = axes[0]
    for method, history in histories.items():
        if history and 'train_loss' in history:
            ax.plot(history['train_loss'], 
                   label=f"{method.replace('_', ' ').title()}", 
                   color=colors.get(method, 'gray'),
                   linewidth=2)
    
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot validation dice
    ax = axes[1]
    for method, history in histories.items():
        if history and 'val_dice' in history:
            ax.plot(history['val_dice'], 
                   label=f"{method.replace('_', ' ').title()}", 
                   color=colors.get(method, 'gray'),
                   linewidth=2)
    
    ax.set_title('Validation Dice Score Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(histories, save_dir):
    """Create a summary table comparing all methods"""
    data = {
        'Method': [],
        'Best Validation Dice': [],
        'Final Validation Dice': [],
        'Epochs to Converge': [],
        'Labeled Sample Size': []
    }
    
    labeled_sizes = {
        'supervised': 225,
        'mean_teacher': 20,
        'mixmatch': 20
    }
    
    for method, history in histories.items():
        if not history or 'val_dice' not in history:
            continue
            
        val_dice = history['val_dice']
        best_dice = max(val_dice)
        final_dice = val_dice[-1]
        
        # Find when model reached 95% of best performance
        threshold = 0.95 * best_dice
        epochs_to_converge = next((i for i, v in enumerate(val_dice) if v >= threshold), len(val_dice))
        
        data['Method'].append(method.replace('_', ' ').title())
        data['Best Validation Dice'].append(f"{best_dice:.4f}")
        data['Final Validation Dice'].append(f"{final_dice:.4f}")
        data['Epochs to Converge'].append(epochs_to_converge)
        data['Labeled Sample Size'].append(labeled_sizes[method])
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    
    # Also return as a formatted markdown table
    return df.to_markdown(index=False)

def create_efficiency_plot(histories, save_dir):
    """Create a plot showing model performance vs labeled data size"""
    methods = []
    best_dices = []
    labeled_sizes = []
    
    for method, history in histories.items():
        if not history or 'val_dice' not in history:
            continue
            
        methods.append(method.replace('_', ' ').title())
        best_dices.append(max(history['val_dice']))
        
        # Get labeled sample size
        if method == 'supervised':
            labeled_sizes.append(225)
        else:
            labeled_sizes.append(20)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    bars = plt.bar(methods, best_dices, color=['blue', 'green', 'red'])
    plt.title('Best Dice Score by Method')
    plt.ylabel('Dice Score')
    plt.ylim(0, 1.0)
    
    # Add labeled sample size as text on bars
    for bar, size in zip(bars, labeled_sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{size} labeled\nsamples',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_convergence(histories, save_dir):
    """Analyze and compare convergence speed"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'supervised': 'blue',
        'mean_teacher': 'green',
        'mixmatch': 'red'
    }
    
    for method, history in histories.items():
        if not history or 'val_dice' not in history:
            continue
            
        val_dice = history['val_dice']
        normalized_dice = [d / max(val_dice) for d in val_dice]  # Normalize to range [0, 1]
        
        ax.plot(normalized_dice, 
               label=f"{method.replace('_', ' ').title()}", 
               color=colors.get(method, 'gray'),
               linewidth=2)
    
    # Add 90% convergence line
    ax.axhline(y=0.9, color='grey', linestyle='--', label='90% of max performance')
    
    ax.set_title('Convergence Speed Comparison (Normalized)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Validation Dice')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare results from different training methods")
    parser.add_argument('--results_dir', type=str, default='.', help='Base directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--supervised_dir', type=str, default=None, help='Specific directory for supervised results')
    parser.add_argument('--mean_teacher_dir', type=str, default=None, help='Specific directory for mean teacher results')
    parser.add_argument('--mixmatch_dir', type=str, default=None, help='Specific directory for mixmatch results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect results
    histories = {}
    methods = ['supervised', 'mean_teacher', 'mixmatch']
    
    for method in methods:
        if method == 'supervised' and args.supervised_dir:
            result_dir = args.supervised_dir
        elif method == 'mean_teacher' and args.mean_teacher_dir:
            result_dir = args.mean_teacher_dir
        elif method == 'mixmatch' and args.mixmatch_dir:
            result_dir = args.mixmatch_dir
        else:
            # Try to find latest experiment
            result_dir = find_latest_experiment(args.results_dir, method)
        
        if result_dir:
            print(f"Loading results for {method} from {result_dir}")
            history = load_results(result_dir, method)
            if history:
                histories[method] = history
                print(f"Loaded history with {len(history['val_dice'])} epochs")
        else:
            print(f"No results directory found for {method}")
    
    # If we have at least two methods with results, create comparisons
    if len(histories) >= 1:
        # Plot learning curves
        plot_learning_curves(histories, args.output_dir)
        
        # Create summary table
        summary_table = create_summary_table(histories, args.output_dir)
        print("\n=== Performance Summary ===\n")
        print(summary_table)
        
        # Create efficiency plot
        if len(histories) >= 2:
            create_efficiency_plot(histories, args.output_dir)
            analyze_convergence(histories, args.output_dir)
            
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    else:
        print("Not enough results to create comparisons")

if __name__ == "__main__":
    main()