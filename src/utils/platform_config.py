import os
import sys
from typing import Dict, Any
import yaml
import torch

def detect_platform() -> str:
    """Detect the current platform (local, colab, or kaggle)"""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif 'COLAB_GPU' in os.environ:
        return 'colab'
    else:
        return 'local'

def setup_kaggle_environment() -> Dict[str, str]:
    """Setup Kaggle-specific paths and configurations"""
    from kaggle_datasets import KaggleDatasets
    
    # Kaggle paths
    paths = {
        'data_dir': '/kaggle/input',
        'output_dir': '/kaggle/working',
        'model_dir': '/kaggle/working/models'
    }
    
    # Create necessary directories
    os.makedirs(paths['model_dir'], exist_ok=True)
    return paths

def setup_colab_environment() -> Dict[str, str]:
    """Setup Colab-specific paths and configurations"""
    from google.colab import drive
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Colab paths
    paths = {
        'data_dir': '/content/data',
        'output_dir': '/content/drive/MyDrive/project_output',
        'model_dir': '/content/drive/MyDrive/project_output/models'
    }
    
    # Create necessary directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def setup_local_environment() -> Dict[str, str]:
    """Setup local paths and configurations"""
    # Local paths
    paths = {
        'data_dir': 'data',
        'output_dir': 'outputs',
        'model_dir': 'outputs/models'
    }
    
    # Create necessary directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def update_config_for_platform(config: Dict[str, Any], platform: str) -> Dict[str, Any]:
    """Update configuration based on platform"""
    if platform == 'kaggle':
        paths = setup_kaggle_environment()
    elif platform == 'colab':
        paths = setup_colab_environment()
    else:
        paths = setup_local_environment()
    
    # Update data paths
    config['data']['train_path'] = os.path.join(paths['data_dir'], 'train')
    config['data']['test_path'] = os.path.join(paths['data_dir'], 'test')
    config['output']['save_dir'] = paths['output_dir']
    
    # Update device configuration
    if platform in ['kaggle', 'colab']:
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update number of workers based on platform
    if platform == 'kaggle':
        config['data']['num_workers'] = 2  # Kaggle has limited resources
    elif platform == 'colab':
        config['data']['num_workers'] = 2  # Colab also has limitations
    
    return config

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and update configuration based on platform"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    platform = detect_platform()
    config = update_config_for_platform(config, platform)
    return config, platform

def setup_environment(platform: str) -> None:
    """Setup necessary environment variables and installations"""
    if platform == 'kaggle':
        # Kaggle-specific setup
        pass
    elif platform == 'colab':
        # Install additional dependencies for Colab
        os.system('pip install -q ultralytics albumentations')
        
        # Install detectron2 in Colab
        os.system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html')
    else:
        # Local setup - assuming dependencies are installed via requirements.txt
        pass 