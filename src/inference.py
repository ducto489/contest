import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ObjectDetectionDataset
from src.models.model_factory import ModelFactory

class Inferencer:
    def __init__(self, config, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model and load checkpoint
        self.model = ModelFactory.create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize test dataset and dataloader
        test_dataset = ObjectDetectionDataset(
            self.config['data']['test_path'],
            transforms=ObjectDetectionDataset._get_default_transforms(self.config),
            is_test=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def process_predictions(self, predictions, image_ids):
        """Convert model predictions to submission format"""
        results = []
        
        for pred, img_id in zip(predictions, image_ids):
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    results.append({
                        'image_id': img_id,
                        'label': label,
                        'confidence': score,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
        
        return results

    def run_inference(self):
        all_predictions = []
        
        with torch.no_grad():
            for images, image_ids in tqdm(self.test_loader, desc='Running inference'):
                images = images.to(self.device)
                predictions = self.model(images)
                batch_results = self.process_predictions(predictions, image_ids)
                all_predictions.extend(batch_results)
        
        # Create submission DataFrame
        df = pd.DataFrame(all_predictions)
        
        # Save submission
        submission_path = os.path.join(self.config['output']['save_dir'], 
                                     self.config['output']['submission_file'])
        df.to_csv(submission_path, index=False)
        print(f'Submission saved to {submission_path}') 