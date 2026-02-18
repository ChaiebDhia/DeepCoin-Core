import torch
import torch.nn as nn
from torchvision import models

def get_deepcoin_model(num_classes):
    # Load a pre-trained EfficientNet-B3 (Transfer Learning)
    # This means our AI already knows how to see 'shapes' from 1 million images
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    
    # We replace the 'Head' (the last layer) to match your coin classes
    # EfficientNet-B3 has 1536 features going into the final layer
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), # Prevents the AI from 'memorizing' (Overfitting)
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":
    # Test if it works
    test_model = get_deepcoin_model(num_classes=500)
    print("DeepCoin Model Initialized Successfully!")