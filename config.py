from dataclasses import dataclass,field
from pathlib import Path
import time

@dataclass
class ExperimentConfig:
    experiment_name: str
    experiment_type: str  # 'supervised' or 'semi-supervised'
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    results_dir: Path = Path('experiment_results')
    
    def __post_init__(self):
        """Create results directory after initialization"""
        self.results_dir.mkdir(exist_ok=True)
    
    def get_experiment_dir(self):
        """Get and create experiment directory"""
        exp_dir = self.results_dir / f"{self.experiment_type}_{self.experiment_name}_{self.timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

@dataclass
class StableSSLConfig:
    """Configuration optimized for stable SSL training"""
    
    # Data parameters
    img_size_x: int = 512
    img_size_y: int = 512
    num_channels: int = 1
    num_classes: int = 1  # Changed from 2 to 1 for binary segmentation
    batch_size: int = 8 # Reduced batch size for stability
    
    # Model architecture
    initial_filters: int = 32
    n_filters: int = 32  # Added n_filters attribute to match what the model is looking for
    filters: list = field(default_factory=lambda: [32, 64, 128, 256, 512])  # Added filters list
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # Training parameters
    num_epochs: int = 100
    initial_learning_rate: float = 0.00001
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    learning_rate: float = 0.00001  # Added for consistency with training_config
    weight_decay: float = 1e-5  # Added for consistency with training_config
    
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
    
    # Augmentation parameters
    noise_std: float = 0.1
    rotation_range: float = 15
    zoom_range: float = 0.1
    scale_range: tuple = (0.9, 1.1)  # Added for augmentation_config
    brightness_range: tuple = (0.9, 1.1)  # Added for augmentation_config
    contrast_range: tuple = (0.9, 1.1)  # Added for augmentation_config
    elastic_deform_sigma: float = 20.0  # Added for augmentation_config
    elastic_deform_alpha: float = 500.0  # Added for augmentation_config
    
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
            'filters': self.filters,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'num_classes': self.num_classes
        }

    @property
    def training_config(self):
        """Get training-specific configuration"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs
        }

    @property
    def augmentation_config(self):
        """Get data augmentation configuration"""
        return {
            'rotation_range': self.rotation_range,
            'scale_range': self.scale_range,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'elastic_deform_sigma': self.elastic_deform_sigma,
            'elastic_deform_alpha': self.elastic_deform_alpha
        }