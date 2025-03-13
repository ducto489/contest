from typing import Dict, Any
import torch
from ultralytics import YOLO
import torchvision

class ModelFactory:
    @staticmethod
    def create_model(config: Dict[str, Any]) -> torch.nn.Module:
        model_name = config['model']['name'].lower()
        num_classes = config['model']['num_classes']
        pretrained = config['model']['pretrained']

        if model_name == 'yolov8':
            model = YOLO('yolov8n.pt' if pretrained else None)
            if not pretrained:
                model.model.num_classes = num_classes
            return model
        
        elif model_name == 'resnet50_fasterrcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model
        
        elif model_name == 'resnet50_retinanet':
            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        optimizer_name = config['training']['optimizer'].lower()
        lr = config['training']['learning_rate']

        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
        scheduler_name = config['training']['scheduler'].lower()
        epochs = config['training']['epochs']

        if scheduler_name == 'cosineannealinglr':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        elif scheduler_name == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}") 