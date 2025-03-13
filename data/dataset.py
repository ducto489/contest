import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, transforms=None, is_test=False):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.is_test = is_test
        self.transforms = transforms if transforms else self._get_default_transforms()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        if self.is_test:
            return image, self.image_files[idx]
        
        # TODO: Add your annotation loading logic here for training data
        # boxes = ...
        # labels = ...
        return image, boxes, labels

    @staticmethod
    def _get_default_transforms(config=None):
        if config is None:
            return A.Compose([
                A.Resize(640, 640),
                A.Normalize(),
                ToTensorV2()
            ])
        
        return A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            A.HorizontalFlip(p=config.augmentation.horizontal_flip_prob),
            A.VerticalFlip(p=config.augmentation.vertical_flip_prob),
            A.RandomBrightnessContrast(
                p=config.augmentation.brightness_contrast_prob,
                brightness_limit=config.augmentation.brightness_limit,
                contrast_limit=config.augmentation.contrast_limit
            ),
            A.Normalize(),
            ToTensorV2()
        ]) 