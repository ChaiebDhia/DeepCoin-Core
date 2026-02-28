"""
PyTorch Dataset class for DeepCoin coin classification.
Handles loading preprocessed images and applies data augmentation.

This is the "librarian" that teaches the AI how to read your processed images.
"""

import logging
import os

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)
# WHY __name__ (resolves to 'src.core.dataset'):
#   Every log line shows the exact module. Inherited from the root logger
#   configured by the caller (FastAPI, training script, etc.).


class DeepCoinDataset(Dataset):
    """
    Custom Dataset for DeepCoin Archaeological Coin Classification.
    
    🎓 WHAT THIS CLASS DOES:
    -----------------------
    1. Scans the data/processed folder and discovers all coin type folders (1015, 1017, etc.)
    2. Maps folder names to numerical indices (AI only understands numbers!)
    3. Collects all image paths with their corresponding labels
    4. Loads images ON-DEMAND during training (saves RAM - only loads what's needed)
    5. Applies transformations (augmentation for training, normalization for validation)
    
    📚 ENGINEERING ANALOGY:
    ----------------------
    Think of this as a LIBRARY SYSTEM:
    - __init__: The librarian catalogs all books (images) and assigns them shelf numbers (labels)
    - __len__: Tells you how many books are in the library
    - __getitem__: When you ask for book #42, the librarian fetches it and hands it to you
    
    Args:
        root_dir (str): Path to processed data directory (e.g., 'data/processed')
        transform (albumentations.Compose): Augmentation pipeline (None = no transforms)
    
    Example:
        >>> dataset = DeepCoinDataset('data/processed', transform=get_train_transforms())
        >>> print(f"Total images: {len(dataset)}")  # 7677
        >>> print(f"Number of classes: {len(dataset.classes)}")  # 438
        >>> image, label = dataset[0]  # Get first image
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # STEP 1: Map folder names to numerical labels
        # --------------------------------------------
        # Example: ['1015', '1017', '10708', ...] → {1015: 0, 1017: 1, 10708: 2, ...}
        # WHY? Neural networks can't understand "1015" as a category name. 
        # They need 0, 1, 2, 3... (indices for the softmax output layer)
        self.classes = sorted(os.listdir(root_dir))  # Sort for reproducibility
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # STEP 2: Gather all image paths and their labels
        # ------------------------------------------------
        # Instead of loading all images into RAM (would crash!), we just store FILE PATHS.
        # Format: [(path/to/image1.jpg, 0), (path/to/image2.jpg, 0), (path/to/image3.jpg, 1), ...]
        self.samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            
            # Skip if not a directory (e.g., .DS_Store files on Mac)
            if not os.path.isdir(cls_path):
                continue
            
            # Get label index for this class
            label = self.class_to_idx[cls_name]
            
            # Collect all image files in this class folder
            for img_name in os.listdir(cls_path):
                # Only process image files (skip hidden files, etc.)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_path, img_name)
                    self.samples.append((img_path, label))
        
        logger.info(
            "Dataset loaded: root=%s  classes=%d  images=%d",
            root_dir, len(self.classes), len(self.samples),
        )
        # WHY logger not print:
        #   In production (FastAPI, training job) stdout is not monitored.
        #   Logging is captured by the structured log system with timestamps,
        #   severity levels, and module names. The caller controls the level.
        #   print() with emoji is never acceptable in production code.
    
    def __len__(self):
        """
        🎓 REQUIRED PyTorch METHOD #1
        Returns the total number of samples in the dataset.
        
        WHY? The training loop needs to know "how many images do I have?"
        so it can calculate: num_batches = len(dataset) / batch_size
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        🎓 REQUIRED PyTorch METHOD #2
        Loads and returns a single image-label pair at the given index.
        
        WHY? During training, PyTorch's DataLoader calls this method repeatedly:
        - batch[0] = dataset[0]
        - batch[1] = dataset[1]
        - batch[2] = dataset[2]
        - ... and so on
        
        This is LAZY LOADING - we only load images when requested, not all at once!
        
        Args:
            idx (int): Index of the image to retrieve (0 to len(dataset)-1)
        
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: torch.Tensor of shape [3, 299, 299] (C, H, W)
                - label: int (the class index, e.g., 0, 1, 2, ...)
        """
        # Get the image path and label for this index
        img_path, label = self.samples[idx]
        
        # Load the image from disk
        # Using OpenCV (faster) instead of PIL
        image = cv2.imread(img_path)
        
        # Convert BGR (OpenCV default) to RGB (PyTorch/ImageNet standard)
        # WHY? EfficientNet was trained on ImageNet which uses RGB color order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations (augmentation + normalization)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']  # Now it's a torch.Tensor
        
        return image, label
    
    def get_class_name(self, idx):
        """
        Helper method: Convert numerical index back to class name.
        
        Example:
            >>> dataset.get_class_name(0)  # Returns '1015'
            >>> dataset.get_class_name(5)  # Returns '10946'
        """
        return self.idx_to_class[idx]


def get_train_transforms():
    """
    🎨 TRAINING DATA AUGMENTATION PIPELINE
    ======================================
    
    🎓 THE PROBLEM:
    In the real world, users will take photos of coins under different conditions:
    - Different lighting (bright sunlight vs indoor lamp)
    - Different angles (slightly rotated)
    - Different camera quality (noise, blur)
    
    If we train ONLY on your perfect preprocessed images, the AI will FAIL in production!
    
    🛡️ THE SOLUTION: DATA AUGMENTATION
    We artificially create "variations" of each image during training:
    - Rotate the coin slightly (±15 degrees)
    - Change brightness/contrast (simulate different lighting)
    - Add noise (simulate lower quality cameras)
    - Apply elastic transforms (simulate slight warping)
    
    This makes the AI ROBUST - it learns to recognize coins even in imperfect conditions.
    
    🔬 EACH AUGMENTATION EXPLAINED:
    """
    return A.Compose([
        # 1. HORIZONTAL FLIP (50% probability)
        # WHY? Coins are symmetric — a mirrored coin is still the same coin.
        # This is FREE augmentation: doubles effective dataset size at zero cost.
        A.HorizontalFlip(p=0.5),

        # 2. ROTATION (±20 degrees, 60% probability)
        # WHY? Users don't photograph coins perfectly aligned.
        # ±20° is realistic for handheld photography WITHOUT being so extreme
        # that coin features become unrecognizable (was ±30° — too aggressive).
        # ENGINEERING RULE: augmentation should simulate real-world variation,
        # not torture the model with impossible scenarios.
        A.Rotate(limit=20, p=0.6),

        # 3. BRIGHTNESS & CONTRAST (±20%, 50% probability)
        # WHY? Lighting varies between indoor/outdoor/flash photography.
        # ±20% is enough to simulate real variation without destroying features.
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # 4. COARSE DROPOUT (20% probability) — reduced from 40%
        # WHY? Randomly blacks out small square patches — simulates worn coin areas,
        # partial occlusion, corrosion patches. Very effective overfitting reducer.
        # Reduced p=0.4 → p=0.2 because combined with rotation it was too destructive
        # for a dataset with only ~17 images/class average.
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            fill=0,
            p=0.2
        ),

        # 5. RANDOM SHADOW (25% probability)
        # WHY? Simulates uneven lighting when photographing at an angle.
        # A shadow stripe across a coin is very common in real user photos.
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.25
        ),

        # 6. GAUSSIAN NOISE (20% probability)
        # WHY? Smartphone cameras in low light produce sensor noise.
        A.GaussNoise(p=0.2),

        # 7. ELASTIC TRANSFORM (20% probability)
        # WHY? Simulates slight warping (coins aren't always perfectly flat on table).
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # 8. NORMALIZATION — ALWAYS applied, CRITICAL
        # EfficientNet was pretrained on ImageNet with these exact statistics.
        # MUST use same mean/std or model sees "alien" pixel distributions.
        # Converts pixel range [0,255] → approximately [-2, +2]
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # 9. TENSOR CONVERSION — ALWAYS applied
        # OpenCV gives numpy [H,W,C] uint8 → PyTorch needs [C,H,W] float32
        ToTensorV2()
    ])


def get_val_transforms():
    """
    🧪 VALIDATION/TEST TRANSFORMS (NO AUGMENTATION!)
    ================================================
    
    🎓 WHY NO AUGMENTATION HERE?
    During validation/testing, we want to measure the AI's REAL performance.
    If we augment validation data, we're "cheating" - the AI might get lucky
    with a rotation that makes recognition easier.
    
    We ONLY apply:
    1. Normalization (required for EfficientNet compatibility)
    2. Tensor conversion (required for PyTorch)
    
    Think of it like a school exam:
    - Training = practice problems with hints and variations (augmentation)
    - Validation = the REAL exam with standard questions (no augmentation)
    """
    return A.Compose([
        # Same normalization as training (MUST match!)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Convert to tensor
        ToTensorV2()
    ])


# 🎯 PROFESSIONAL TIP: Why Albumentations over torchvision.transforms?
# ====================================================================
# 1. FASTER: Albumentations is built on OpenCV (C++ backend) vs PIL (Python)
# 2. MORE AUGMENTATIONS: 70+ transforms vs torchvision's 30+
# 3. RESEARCH-PROVEN: Used in Kaggle winning solutions
# 4. SPATIAL TRANSFORMS: Handles bounding boxes/masks (useful for future features)
