from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

# Constants
# DEFAULT_IMG_SIZE_X = 256
# DEFAULT_IMG_SIZE_Y = 256
# DEFAULT_NUM_CHANNELS = 1
# DEFAULT_NUM_CLASSES = 2
# DEFAULT_TARGET_RESOLUTION = (1.0, 1.0)

# @dataclass
# class PancreasSegConfig:
#     """Configuration for pancreas segmentation with SSL"""

#     # Data parameters
#     data_dir: Path = Path('/content/drive/MyDrive/Local_contrastive_loss_data/Task07_Pancreas/cropped')  # Update with your data directory
#     img_size_x: int = DEFAULT_IMG_SIZE_X  # Input image height after resizing
#     img_size_y: int = DEFAULT_IMG_SIZE_Y  # Input image width after resizing
#     num_channels: int = DEFAULT_NUM_CHANNELS  # Grayscale images
#     num_classes: int = DEFAULT_NUM_CLASSES  # Background and pancreas
#     target_resolution: Tuple[float, float] = DEFAULT_TARGET_RESOLUTION  # Target pixel spacing in mm

#     # Model architecture
#     filters: List[int] = (64, 128, 256, 512, 1024)  # Number of filters in each layer
#     dropout_rate: float = 0.2
#     use_batch_norm: bool = True

#     # Training parameters
#     batch_size: int = 16
#     num_epochs: int = 100
#     learning_rate: float = 5e-3
#     min_learning_rate: float = 1e-6  # Minimum learning rate
#     warmup_epochs: int = 3  # Learning rate warmup
    
#     # Enhanced model capacity
#     initial_filters: int = 32
#     # Semi-supervised learning
#     pseudo_label_threshold: float = 0.9  # Confidence threshold for pseudo-labels
#     pseudo_label_update_freq: int = 5  # Update pseudo-labels every N epochs
#     consistency_weight: float = 0.1  # Weight for consistency loss
#     focal_gamma: float = 2.0  # For focal loss
#     dice_weight: float = 0.5
#     bce_weight: float = 0.5
#     focal_weight: float = 0.2
    
#     # Contrastive learning (not currently used in your code)
#     temperature: float = 0.1  # Temperature for contrastive loss
#     projection_dim: int = 128  # Dimension of projection head output
#     num_augmentations: int = 2  # Number of augmented views per image

#     # Data augmentation
#     augmentation_strength: float = 0.5
#     rotation_range: float = 15.0  # Degrees
#     zoom_range: float = 0.1
#     horizontal_flip: bool = True
#     vertical_flip: bool = False
#     random_brightness: float = 0.1
#     random_contrast: float = 0.1
#     scale_range: Tuple[float, float] = (0.9, 1.1)
#     # brightness_range: Tuple[float, float] = (0.9, 1.1)
#     # contrast_range: Tuple[float, float] = (0.9, 1.1)
#     elastic_deform_sigma: float = 20.0
#     elastic_deform_alpha: float = 500.0

#     #GPU
#     use_mixed_precision: bool = False  # Disabled for stability
#     use_xla: bool = True
#     gradient_clip_norm: float = 1.0
#     ema_decay: float = 0.999 
#     weight_decay: float = 1e-5

#     # Validation
#     validation_freq: int = 1  # Validate every N epochs
#     early_stopping_patience: int = 10
#     min_delta: float = 0.001 # Not used in your current code
#     save_best_only: bool = True  # Not used in your current code

#     # Paths
#     checkpoint_dir: Path = Path('checkpoints')
#     log_dir: Path = Path('logs')
#     output_dir: Path = Path('results')

#     def __post_init__(self):
#         """Create necessary directories and validate config after initialization"""
#         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         if self.num_classes < 2:
#             raise ValueError("num_classes must be at least 2 (background + 1 object class)")
# @dataclass
# class PancreasSegConfig:
#     """Configuration for pancreas segmentation with SSL"""
#     data_dir: Path = Path('/content/drive/MyDrive/Local_contrastive_loss_data/Task07_Pancreas/cropped')
#     img_size_x: int = 256
#     img_size_y: int = 256
#     num_channels: int = 1
#     num_classes: int = 2
#     batch_size: int = 8
#     num_epochs: int = 100
#     temperature: float = 0.1
#     learning_rate: float = 0.001
#     consistency_weight: float = 0.1
#     checkpoint_dir: Path = Path('checkpoints')


from dataclasses import dataclass
from pathlib import Path

@dataclass
class StableSSLConfig:
    """Configuration optimized for stable SSL training"""
    
    # Data parameters
    img_size_x: int = 512  # Updated to your desired size
    img_size_y: int = 512  # Updated to your desired size
    num_channels: int = 1
    num_classes: int = 2
    batch_size: int = 4  # Reduced batch size to prevent OOM errors
    
    # Model architecture
    initial_filters: int = 16  # Reduced number of initial filters
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # Training parameters
    num_epochs: int = 100
    initial_learning_rate: float = 0.0001
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    
    # SSL specific parameters
    ema_decay: float = 0.999
    consistency_weight: float = 0.1
    consistency_rampup_epochs: int = 5
    
    # Loss function parameters
    dice_smooth: float = 1e-6
    temperature: float = 0.5
    
    # Training stability
    gradient_clip_norm: float = 2.0
    early_stopping_patience: int = 20
    min_delta: float = 0.001
    
    # Memory optimization
    use_mixed_precision: bool = True  # Enable mixed precision for memory efficiency
    memory_growth: bool = True  # Enable memory growth
    
    # Augmentation parameters
    noise_std: float = 0.1
    rotation_range: float = 15
    zoom_range: float = 0.1
    
    # Paths
    checkpoint_dir: Path = Path('ssl_checkpoints')
    log_dir: Path = Path('ssl_logs')
    output_dir: Path = Path('ssl_results')
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def training_steps_per_epoch(self):
        """Calculate training steps per epoch based on dataset size"""
        return 100  # Adjust based on your dataset size
        
    @property
    def consistency_rampup_steps(self):
        """Calculate total steps for consistency loss ramp-up"""
        return self.consistency_rampup_epochs * self.training_steps_per_epoch

    @property
    def model_config(self):
        """Get model-specific configuration"""
        return {
            'initial_filters': self.initial_filters,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'num_classes': self.num_classes
        }

    @property
    def training_config(self):
        """Get training-specific configuration"""
        return {
            'batch_size': self.batch_size,
            'initial_learning_rate': self.initial_learning_rate,
            'num_epochs': self.num_epochs
        }

    @property
    def augmentation_config(self):
        """Get data augmentation configuration"""
        return {
            'rotation_range': self.rotation_range,
            'zoom_range': self.zoom_range,
            'noise_std': self.noise_std
        }