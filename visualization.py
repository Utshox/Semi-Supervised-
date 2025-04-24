import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import seaborn as sns
from skimage import measure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

def normalize_image(image):
    """Normalize image to 0-1 range for display"""
    img_min, img_max = np.min(image), np.max(image)
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return image

def create_custom_colormap():
    """Create a custom colormap for overlay visualization"""
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # Transparent to red with alpha
    return LinearSegmentedColormap.from_list('custom_cmap', colors)

def visualize_slice(image, mask=None, prediction=None, slice_idx=None, alpha=0.5, title=None):
    """
    Visualize a single slice with optional mask and prediction overlays
    
    Args:
        image: 3D image volume (can be grayscale or already normalized)
        mask: Ground truth mask (optional)
        prediction: Predicted mask (optional)
        slice_idx: Index of slice to visualize (if None, middle slice is used)
        alpha: Transparency of overlay
        title: Title for the plot
    """
    if len(image.shape) == 4:  # Handle batch dimension
        image = image[0]
    if len(image.shape) == 4 and image.shape[-1] == 1:  # Handle channel dimension
        image = np.squeeze(image, axis=-1)
    
    if slice_idx is None:
        slice_idx = image.shape[2] // 2  # Middle slice by default
    
    # Get the slice
    img_slice = image[:, :, slice_idx]
    
    # Normalize for visualization
    img_slice = normalize_image(img_slice)
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(img_slice, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot with ground truth if available
    if mask is not None:
        if len(mask.shape) == 4:  # Handle batch dimension
            mask = mask[0]
        if len(mask.shape) == 4 and mask.shape[-1] == 1:  # Handle channel dimension
            mask = np.squeeze(mask, axis=-1)
            
        mask_slice = mask[:, :, slice_idx]
        
        plt.subplot(132)
        plt.imshow(img_slice, cmap='gray')
        plt.imshow(mask_slice, cmap='Reds', alpha=alpha)
        plt.title('Ground Truth Overlay')
        plt.axis('off')
    
    # Plot with prediction if available
    if prediction is not None:
        if len(prediction.shape) == 4:  # Handle batch dimension
            prediction = prediction[0]
        if len(prediction.shape) == 4 and prediction.shape[-1] == 1:  # Handle channel dimension
            prediction = np.squeeze(prediction, axis=-1)
            
        pred_slice = prediction[:, :, slice_idx]
        
        plt.subplot(133)
        plt.imshow(img_slice, cmap='gray')
        plt.imshow(pred_slice, cmap='Blues', alpha=alpha)
        plt.title('Prediction Overlay')
        plt.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_comparison(image, ground_truth, prediction, slice_indices=None, save_path=None):
    """
    Create a detailed comparison visualization with multiple slices
    
    Args:
        image: 3D image volume
        ground_truth: Ground truth mask
        prediction: Predicted mask
        slice_indices: List of slice indices to visualize (if None, 3 evenly spaced slices are chosen)
        save_path: Path to save the visualization (if None, plot is displayed)
    """
    # Remove batch dimension if present
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image[0]
    if len(ground_truth.shape) == 4 and ground_truth.shape[0] == 1:
        ground_truth = ground_truth[0]
    if len(prediction.shape) == 4 and prediction.shape[0] == 1:
        prediction = prediction[0]
    
    # Remove channel dimension if present
    if len(image.shape) == 4 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
    if len(ground_truth.shape) == 4 and ground_truth.shape[-1] == 1:
        ground_truth = np.squeeze(ground_truth, axis=-1)
    if len(prediction.shape) == 4 and prediction.shape[-1] == 1:
        prediction = np.squeeze(prediction, axis=-1)
    
    # Select slice indices if not provided
    if slice_indices is None:
        depth = image.shape[2]
        slice_indices = [int(depth * 0.3), int(depth * 0.5), int(depth * 0.7)]
    
    num_slices = len(slice_indices)
    
    # Create subplots
    fig, axes = plt.subplots(3, num_slices, figsize=(4*num_slices, 10))
    
    # Create custom colormaps
    gt_cmap = create_custom_colormap()
    pred_cmap = 'Blues'
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        # Ensure slice_idx is within bounds
        slice_idx = min(max(0, slice_idx), image.shape[2]-1)
        
        # Get the slices
        img_slice = normalize_image(image[:, :, slice_idx])
        gt_slice = ground_truth[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
        
        # Original image
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].set_title(f'Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Ground truth overlay
        axes[1, i].imshow(img_slice, cmap='gray')
        gt_mask = axes[1, i].imshow(gt_slice, cmap=gt_cmap, alpha=0.7)
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Prediction overlay
        axes[2, i].imshow(img_slice, cmap='gray')
        pred_mask = axes[2, i].imshow(pred_slice, cmap=pred_cmap, alpha=0.7)
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    # Add a color bar for ground truth and prediction
    gt_ax = fig.add_axes([0.92, 0.66, 0.02, 0.2])
    pred_ax = fig.add_axes([0.92, 0.33, 0.02, 0.2])
    
    fig.colorbar(gt_mask, cax=gt_ax, label='Ground Truth')
    fig.colorbar(pred_mask, cax=pred_ax, label='Prediction')
    
    plt.suptitle('Segmentation Results Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return fig

def calculate_metrics(ground_truth, prediction, threshold=0.5):
    """
    Calculate evaluation metrics for segmentation
    
    Args:
        ground_truth: Ground truth mask
        prediction: Predicted mask (probabilities)
        threshold: Threshold for binarizing prediction
    
    Returns:
        dict: Dictionary of metrics
    """
    # Binarize prediction if needed
    if np.max(prediction) > 1:
        prediction = prediction / np.max(prediction)
    
    pred_binary = (prediction >= threshold).astype(np.float32)
    gt_binary = (ground_truth >= threshold).astype(np.float32)
    
    # Flatten arrays
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(gt_flat * pred_flat)
    union = np.sum(gt_flat) + np.sum(pred_flat) - intersection
    
    # Calculate metrics
    metrics = {}
    metrics['dice'] = (2.0 * intersection) / (np.sum(gt_flat) + np.sum(pred_flat) + 1e-7)
    metrics['iou'] = intersection / (union + 1e-7)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum(gt_flat * pred_flat)
    fp = np.sum(pred_flat) - tp
    fn = np.sum(gt_flat) - tp
    tn = np.sum((1-gt_flat) * (1-pred_flat))
    
    metrics['sensitivity'] = tp / (tp + fn + 1e-7)
    metrics['specificity'] = tn / (tn + fp + 1e-7)
    metrics['precision'] = tp / (tp + fp + 1e-7)
    
    # Calculate volumetric similarity
    vol_gt = np.sum(gt_flat)
    vol_pred = np.sum(pred_flat)
    metrics['volumetric_similarity'] = 1 - abs(vol_gt - vol_pred) / (vol_gt + vol_pred + 1e-7)
    
    # Surface distance
    try:
        # Convert to binary
        gt_binary_3d = gt_binary.astype(np.bool)
        pred_binary_3d = pred_binary.astype(np.bool)
        
        # Extract surface voxels
        gt_surface = measure.find_contours(gt_binary_3d[:,:,gt_binary_3d.shape[2]//2], 0.5)
        pred_surface = measure.find_contours(pred_binary_3d[:,:,pred_binary_3d.shape[2]//2], 0.5)
        
        if len(gt_surface) > 0 and len(pred_surface) > 0:
            gt_voxels = np.array(gt_surface[0])
            pred_voxels = np.array(pred_surface[0])
            
            # Calculate distances
            from scipy.spatial.distance import cdist
            distances = cdist(gt_voxels, pred_voxels, 'euclidean')
            
            # Hausdorff distance (maximum distance)
            metrics['hausdorff'] = np.max([np.min(distances, axis=0).max(), np.min(distances, axis=1).max()])
            
            # Average surface distance
            metrics['avg_surface_distance'] = (np.mean(np.min(distances, axis=0)) + 
                                             np.mean(np.min(distances, axis=1))) / 2
        else:
            metrics['hausdorff'] = float('nan')
            metrics['avg_surface_distance'] = float('nan')
    except Exception as e:
        metrics['hausdorff'] = float('nan')
        metrics['avg_surface_distance'] = float('nan')
    
    return metrics

def plot_metrics_over_time(history, save_path=None):
    """
    Plot training metrics over time
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Dice score
    plt.subplot(2, 2, 2)
    if 'dice_coef' in history:
        plt.plot(history['dice_coef'], label='Training Dice')
    if 'val_dice_coef' in history:
        plt.plot(history['val_dice_coef'], label='Validation Dice')
    plt.title('Dice Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.title('Learning Rate Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
    # Plot any additional metric (e.g. IoU)
    for metric in history.keys():
        if metric not in ['loss', 'val_loss', 'dice_coef', 'val_dice_coef', 'lr']:
            plt.subplot(2, 2, 4)
            plt.plot(history[metric], label=f'Training {metric}')
            val_metric = f'val_{metric}'
            if val_metric in history:
                plt.plot(history[val_metric], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_confusion_matrix(ground_truth, prediction, threshold=0.5, normalize=True, save_path=None):
    """
    Plot confusion matrix for segmentation results
    
    Args:
        ground_truth: Ground truth mask
        prediction: Predicted mask
        threshold: Threshold for binarizing prediction
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (if None, plot is displayed)
    """
    # Binarize prediction if needed
    if np.max(prediction) > 1:
        prediction = prediction / np.max(prediction)
    
    pred_binary = (prediction >= threshold).astype(np.int32)
    gt_binary = (ground_truth >= threshold).astype(np.int32)
    
    # Flatten arrays
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt_flat, pred_flat)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=['Background', 'Pancreas'], 
                yticklabels=['Background', 'Pancreas'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return plt.gcf()

def visualize_3D(image, mask=None, prediction=None, threshold=0.5, spacing=(1,1,1), save_path=None):
    """
    Generate 3D visualization of segmentation
    This is a placeholder that saves key slices since direct 3D rendering is challenging
    
    Args:
        image: 3D image volume
        mask: Ground truth mask (optional)
        prediction: Predicted mask (optional)
        threshold: Threshold for binarizing prediction
        spacing: Voxel spacing (for proper aspect ratio)
        save_path: Path to save the visualization (if None, plot is displayed)
    """
    # Select key slices for visualization (axial, coronal, sagittal)
    axial_idx = image.shape[2] // 2
    coronal_idx = image.shape[1] // 2
    sagittal_idx = image.shape[0] // 2
    
    # Normalize image
    norm_image = normalize_image(image)
    
    # Prepare binary masks
    if mask is not None:
        if np.max(mask) > 1:
            mask = mask / np.max(mask)
        binary_mask = (mask >= threshold).astype(np.float32)
    
    if prediction is not None:
        if np.max(prediction) > 1:
            prediction = prediction / np.max(prediction)
        binary_pred = (prediction >= threshold).astype(np.float32)
    
    # Create figure
    fig, axes = plt.subplots(3, 3 if prediction is not None else 2, figsize=(12, 12))
    
    # Plot axial view
    axes[0, 0].imshow(norm_image[:, :, axial_idx], cmap='gray')
    axes[0, 0].set_title('Axial View')
    axes[0, 0].axis('off')
    
    if mask is not None:
        axes[0, 1].imshow(norm_image[:, :, axial_idx], cmap='gray')
        axes[0, 1].imshow(binary_mask[:, :, axial_idx], cmap='Reds', alpha=0.5)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
    
    if prediction is not None:
        axes[0, 2].imshow(norm_image[:, :, axial_idx], cmap='gray')
        axes[0, 2].imshow(binary_pred[:, :, axial_idx], cmap='Blues', alpha=0.5)
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
    
    # Plot coronal view
    axes[1, 0].imshow(norm_image[:, coronal_idx, :].transpose(), cmap='gray')
    axes[1, 0].set_title('Coronal View')
    axes[1, 0].axis('off')
    
    if mask is not None:
        axes[1, 1].imshow(norm_image[:, coronal_idx, :].transpose(), cmap='gray')
        axes[1, 1].imshow(binary_mask[:, coronal_idx, :].transpose(), cmap='Reds', alpha=0.5)
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
    
    if prediction is not None:
        axes[1, 2].imshow(norm_image[:, coronal_idx, :].transpose(), cmap='gray')
        axes[1, 2].imshow(binary_pred[:, coronal_idx, :].transpose(), cmap='Blues', alpha=0.5)
        axes[1, 2].set_title('Prediction')
        axes[1, 2].axis('off')
    
    # Plot sagittal view
    axes[2, 0].imshow(norm_image[sagittal_idx, :, :].transpose(), cmap='gray')
    axes[2, 0].set_title('Sagittal View')
    axes[2, 0].axis('off')
    
    if mask is not None:
        axes[2, 1].imshow(norm_image[sagittal_idx, :, :].transpose(), cmap='gray')
        axes[2, 1].imshow(binary_mask[sagittal_idx, :, :].transpose(), cmap='Reds', alpha=0.5)
        axes[2, 1].set_title('Ground Truth')
        axes[2, 1].axis('off')
    
    if prediction is not None:
        axes[2, 2].imshow(norm_image[sagittal_idx, :, :].transpose(), cmap='gray')
        axes[2, 2].imshow(binary_pred[sagittal_idx, :, :].transpose(), cmap='Blues', alpha=0.5)
        axes[2, 2].set_title('Prediction')
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return fig

def compare_methods(images, ground_truths, predictions_dict, metrics_dict, save_dir=None):
    """
    Compare multiple segmentation methods on the same data
    
    Args:
        images: List of image volumes
        ground_truths: List of ground truth masks
        predictions_dict: Dictionary of method_name -> list of predictions
        metrics_dict: Dictionary of method_name -> list of metric dicts
        save_dir: Directory to save visualizations
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # For each test case
    for i, (image, gt) in enumerate(zip(images, ground_truths)):
        # Create comparison visualization
        plt.figure(figsize=(15, 5 * (len(predictions_dict) + 1)))
        
        # Show original image with ground truth
        plt.subplot(len(predictions_dict) + 1, 1, 1)
        axial_slice = image.shape[2] // 2
        plt.imshow(normalize_image(image[:, :, axial_slice]), cmap='gray')
        plt.imshow((gt[:, :, axial_slice] > 0.5).astype(np.float32), cmap='Reds', alpha=0.5)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Show each method's prediction
        for j, (method_name, predictions) in enumerate(predictions_dict.items()):
            plt.subplot(len(predictions_dict) + 1, 1, j + 2)
            plt.imshow(normalize_image(image[:, :, axial_slice]), cmap='gray')
            plt.imshow((predictions[i][:, :, axial_slice] > 0.5).astype(np.float32), 
                       cmap='Blues', alpha=0.5)
            
            # Get metrics for this case and method
            case_metrics = metrics_dict[method_name][i]
            metrics_str = f"Dice: {case_metrics['dice']:.4f}, IoU: {case_metrics['iou']:.4f}"
            
            plt.title(f'{method_name} - {metrics_str}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'comparison_case_{i}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
    
    # Create summary metrics table
    summary = {}
    for method_name, metrics_list in metrics_dict.items():
        method_summary = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            method_summary[f'{metric}_mean'] = np.mean(values)
            method_summary[f'{metric}_std'] = np.std(values)
        summary[method_name] = method_summary
    
    # Create and save summary plot
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['dice', 'iou', 'sensitivity', 'specificity']
    methods = list(summary.keys())
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        means = [summary[method][f'{metric}_mean'] for method in methods]
        stds = [summary[method][f'{metric}_std'] for method in methods]
        
        bars = plt.bar(methods, means, yerr=stds, capsize=10)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'{metric.capitalize()} Comparison')
        plt.ylim([0, 1.1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also save as CSV
        metrics_df = pd.DataFrame()
        for method in methods:
            for metric in metrics_to_plot:
                metrics_df.loc[method, f'{metric}_mean'] = summary[method][f'{metric}_mean']
                metrics_df.loc[method, f'{metric}_std'] = summary[method][f'{metric}_std']
        
        metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'))
    
    return summary

def generate_report_visualizations(trainer, results_dir, method_name):
    """
    Generate comprehensive visualizations for a model and save to results directory
    
    Args:
        trainer: Trainer object with model and validation data
        results_dir: Directory to save results
        method_name: Name of the method for labeling
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Generating visualizations for {method_name}...")
    
    # Get validation data
    val_images = []
    val_masks = []
    val_predictions = []
    
    # Create a directory for slice visualizations
    slices_dir = os.path.join(results_dir, 'slice_visualizations')
    os.makedirs(slices_dir, exist_ok=True)
    
    # Create a directory for 3D visualizations
    viz_3d_dir = os.path.join(results_dir, '3d_visualizations')
    os.makedirs(viz_3d_dir, exist_ok=True)
    
    # Get validation dataset
    for i, (images, masks) in enumerate(trainer.val_dataset.take(10)):  # Take up to 10 validation samples
        # Get predictions
        predictions = trainer.model.predict(images)
        
        # Save each case
        for j in range(images.shape[0]):
            image = images[j].numpy()
            mask = masks[j].numpy()
            prediction = predictions[j]
            
            # Store for metrics calculation
            val_images.append(image)
            val_masks.append(mask)
            val_predictions.append(prediction)
            
            # Calculate metrics
            metrics = calculate_metrics(mask, prediction)
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                   if k in ['dice', 'iou', 'sensitivity']])
            
            case_idx = i * images.shape[0] + j
            
            # Create comparison visualization
            visualize_comparison(
                image, mask, prediction,
                save_path=os.path.join(slices_dir, f'case_{case_idx}_comparison.png')
            )
            
            # Create 3D visualization
            visualize_3D(
                image, mask, prediction,
                save_path=os.path.join(viz_3d_dir, f'case_{case_idx}_3d.png')
            )
            
            # Create confusion matrix
            plot_confusion_matrix(
                mask, prediction,
                save_path=os.path.join(results_dir, f'case_{case_idx}_confusion.png')
            )
            
            print(f"Case {case_idx}: {metrics_str}")
    
    # Calculate and save metrics for all validation cases
    all_metrics = []
    for i, (mask, pred) in enumerate(zip(val_masks, val_predictions)):
        metrics = calculate_metrics(mask, pred)
        all_metrics.append(metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'validation_metrics.csv'))
    
    # Plot metrics distribution
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(['dice', 'iou', 'sensitivity', 'specificity']):
        plt.subplot(2, 2, i+1)
        sns.histplot(metrics_df[metric], kde=True)
        plt.title(f'{metric.capitalize()} Distribution')
        plt.axvline(metrics_df[metric].mean(), color='r', linestyle='--', 
                   label=f'Mean: {metrics_df[metric].mean():.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # If training history is available, plot learning curves
    if hasattr(trainer, 'history'):
        plot_metrics_over_time(
            trainer.history,
            save_path=os.path.join(results_dir, 'learning_curves.png')
        )
    
    print(f"Visualizations generated and saved to {results_dir}")
    return all_metrics