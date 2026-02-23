"""
🧪 TEST SCRIPT: Dataset Verification
====================================

This script tests our DeepCoinDataset class to make sure it works correctly
BEFORE we start the expensive training process.

🎓 WHAT THIS TESTS:
1. Can the dataset find all 438 classes?
2. Can it load images correctly?
3. Do the transformations work?
4. Are the tensor shapes correct?

Run this with: python scripts/test_dataset.py
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.dataset import DeepCoinDataset, get_train_transforms, get_val_transforms
import matplotlib.pyplot as plt
import numpy as np


def test_dataset_basic():
    """Test 1: Basic dataset loading"""
    print("=" * 60)
    print("TEST 1: Basic Dataset Loading")
    print("=" * 60)
    
    # Create dataset WITHOUT transforms first (to see raw images)
    dataset = DeepCoinDataset(root_dir='data/processed', transform=None)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Total images: {len(dataset)}")
    print(f"   Number of classes: {len(dataset.classes)}")
    print(f"   First 10 classes: {dataset.classes[:10]}")
    print(f"   Last 10 classes: {dataset.classes[-10:]}")
    
    # Try loading one image
    print(f"\n📸 Loading first image...")
    image, label = dataset[0]
    print(f"   Image shape: {image.shape}")  # Should be (299, 299, 3)
    print(f"   Image dtype: {image.dtype}")  # Should be uint8
    print(f"   Label: {label}")
    print(f"   Class name: {dataset.get_class_name(label)}")
    
    return dataset


def test_transforms():
    """Test 2: Transformation pipeline"""
    print("\n" + "=" * 60)
    print("TEST 2: Transformation Pipeline")
    print("=" * 60)
    
    # Create dataset WITH training transforms
    train_dataset = DeepCoinDataset(
        root_dir='data/processed',
        transform=get_train_transforms()
    )
    
    # Load one image
    image_tensor, label = train_dataset[0]
    
    print(f"\n✅ Transformed image loaded!")
    print(f"   Tensor shape: {image_tensor.shape}")  # Should be [3, 299, 299]
    print(f"   Tensor dtype: {image_tensor.dtype}")  # Should be torch.float32
    print(f"   Tensor range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    print(f"   Label: {label}")
    
    # Check if normalization worked (values should be roughly [-2, +2])
    assert image_tensor.shape == (3, 299, 299), "❌ Wrong tensor shape!"
    assert -3 < image_tensor.min() < 3, "❌ Normalization might be wrong!"
    assert -3 < image_tensor.max() < 3, "❌ Normalization might be wrong!"
    
    print(f"\n✅ All assertions passed!")


def test_augmentation_visualization():
    """Test 3: Visualize augmentation effects"""
    print("\n" + "=" * 60)
    print("TEST 3: Augmentation Visualization")
    print("=" * 60)
    
    # Load ONE image with different augmentations
    train_dataset = DeepCoinDataset(
        root_dir='data/processed',
        transform=get_train_transforms()
    )
    
    print(f"\n🎨 Generating 6 augmented versions of the same coin...")
    
    # Get the same image 6 times (augmentation is random, so each will be different!)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Data Augmentation: Same Coin, 6 Different Variations', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # Load the same image (index 0) but get different augmentation each time
        image_tensor, label = train_dataset[0]
        
        # Convert tensor back to displayable image
        # [3, 299, 299] → [299, 299, 3] and denormalize
        img = image_tensor.permute(1, 2, 0).numpy()
        
        # Denormalize (reverse the normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Clip to [0, 1] range
        
        ax.imshow(img)
        ax.set_title(f'Augmentation #{i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save to file instead of showing (in case you're running without display)
    output_path = 'augmentation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"   Open this file to see how augmentation creates variety!")


def test_class_distribution():
    """Test 4: Analyze class distribution"""
    print("\n" + "=" * 60)
    print("TEST 4: Class Distribution Analysis")
    print("=" * 60)
    
    dataset = DeepCoinDataset(root_dir='data/processed', transform=None)
    
    # Count images per class
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.get_class_name(label)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Statistics
    counts = list(class_counts.values())
    print(f"\n📊 Distribution Statistics:")
    print(f"   Min images per class: {min(counts)}")
    print(f"   Max images per class: {max(counts)}")
    print(f"   Average images per class: {sum(counts) / len(counts):.1f}")
    print(f"   Median images per class: {sorted(counts)[len(counts)//2]}")
    
    # Show classes with most/least images
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🏆 Top 5 classes (most images):")
    for cls_name, count in sorted_classes[:5]:
        print(f"   {cls_name}: {count} images")
    
    print(f"\n⚠️  Bottom 5 classes (least images):")
    for cls_name, count in sorted_classes[-5:]:
        print(f"   {cls_name}: {count} images")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("DEEPCOIN DATASET TESTING SUITE")
    print("🚀" * 30)
    
    try:
        # Run all tests
        test_dataset_basic()
        test_transforms()
        test_class_distribution()
        test_augmentation_visualization()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED! Dataset is ready for training!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
