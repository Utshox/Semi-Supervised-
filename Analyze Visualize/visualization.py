import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ExperimentVisualizer:
    def __init__(self, save_dir='results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        # Use default Matplotlib style
        plt.style.use('default')  # Changed from 'seaborn'

    def plot_training_curves(self, mean_teacher_history, mixmatch_history=None):
        """Plot training curves comparing different approaches"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Dice scores
        ax1.plot(mean_teacher_history['val_dice'], label='Mean Teacher', color='blue')
        if mixmatch_history:
            ax1.plot(mixmatch_history['val_dice'], label='MixMatch', color='red')
        ax1.set_title('Validation Dice Score')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dice Score')
        ax1.legend()
        ax1.grid(True)

        # Plot losses
        ax2.plot(mean_teacher_history['train_loss'], label='Mean Teacher', color='blue')
        if mixmatch_history:
            ax2.plot(mixmatch_history['train_loss'], label='MixMatch', color='red')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sample_predictions(self, images, labels, mean_teacher_preds, mixmatch_preds=None):
        """Plot sample predictions for visual comparison"""
        n_samples = min(4, len(images))
        fig, axes = plt.subplots(n_samples, 3 + (1 if mixmatch_preds is not None else 0),
                                 figsize=(15, 4 * n_samples))

        for i in range(n_samples):
            # Original image
            axes[i, 0].imshow(images[i, ..., 0], cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(labels[i, ..., 1], cmap='RdYlBu')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Mean Teacher prediction
            axes[i, 2].imshow(mean_teacher_preds[i, ..., 1], cmap='RdYlBu')
            axes[i, 2].set_title('Mean Teacher')
            axes[i, 2].axis('off')

            # MixMatch prediction if available
            if mixmatch_preds is not None:
                axes[i, 3].imshow(mixmatch_preds[i, ..., 1], cmap='RdYlBu')
                axes[i, 3].set_title('MixMatch')
                axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self, metrics_dict):
        """Plot performance metrics comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(metrics_dict.keys())
        metrics = ['Dice', 'Precision', 'Recall']  # Make sure these metrics are in your metrics_dict
        x = np.arange(len(methods))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [metrics_dict[method][metric] for method in methods]
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()