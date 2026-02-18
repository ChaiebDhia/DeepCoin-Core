"""
PyTorch Dataset class for DeepCoin coin classification.
Handles loading preprocessed images and applies data augmentation.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CoinDataset(Dataset):
    """
    Custom Dataset for loading preprocessed coin images.
    
    Args:
        data_dir: Path to processed data directory (e.g., 'data/processed')
        split: 'train', 'val', or 'test'
        transform: Albumentations transform pipeline
    """
    
    def __init__(self, data_dir, class_names, image_paths, labels, transform=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def get_train_transforms():
    """Training data augmentation pipeline."""
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms():
    """Validation/test data transforms (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
