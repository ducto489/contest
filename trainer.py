import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from data.dataset import ObjectDetectionDataset
from models.model_factory import ModelFactory

class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(os.path.join(self.config['output']['save_dir'], 'logs'))
        
        # Create directories
        os.makedirs(self.config['output']['save_dir'], exist_ok=True)
        
        # Initialize model, optimizer, and scheduler
        self.model = ModelFactory.create_model(self.config)
        self.model = self.model.to(self.device)
        self.optimizer = ModelFactory.get_optimizer(self.model, self.config)
        self.scheduler = ModelFactory.get_scheduler(self.optimizer, self.config)
        
        # Initialize datasets and dataloaders
        train_dataset = ObjectDetectionDataset(
            self.config['data']['train_path'],
            transforms=ObjectDetectionDataset._get_default_transforms(self.config)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            
            self.optimizer.step()
            epoch_loss += losses.item()
            
            progress_bar.set_postfix({'loss': losses.item()})
            
        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        return avg_loss

    def train(self):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            avg_loss = self.train_epoch(epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_checkpoint(epoch, avg_loss, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            self.save_checkpoint(epoch, avg_loss, is_best=False)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['output']['save_dir'], 'last_checkpoint.pth')
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.config['output']['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss'] 