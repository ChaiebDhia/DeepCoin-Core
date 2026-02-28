import logging
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def get_deepcoin_model(num_classes: int) -> nn.Module:
    """
    Build the DeepCoin classification model.

    Architecture: EfficientNet-B3 (ImageNet pretrained) with a custom head.

    WHAT:
        EfficientNet-B3 feature extractor (18 conv layers, 1536-dim output)
        + Dropout(0.4) + Linear(1536, num_classes) classification head.

    WHY EfficientNet-B3:
        Compound scaling — balanced depth/width/resolution simultaneously.
        Gives the best accuracy/parameter tradeoff for our 4.3 GB VRAM budget.
        B4+ would exceed VRAM; B2- loses accuracy on fine-grained coin features.

    WHY Dropout(0.4):
        40% of neurons are zeroed randomly each forward pass during training.
        Forces the remaining neurons to learn robust, non-redundant features.
        Without it: the model memorises the 7,677 training images instead of
        generalising. With it: F1 macro 0.776 on test set.

    WHY NOT retrain from scratch:
        EfficientNet-B3 was pretrained on 1.2M ImageNet images. Those 18 layers
        already know edges, textures, shapes, and metallic surfaces — exactly
        what ancient coins are made of. Transfer learning reduces our training
        data requirement from ~1,000 images/class to ~10 images/class.

    Args:
        num_classes: Number of output classes. 438 for the current training set.

    Returns:
        nn.Module ready for .load_state_dict() or training.
    """
    model = models.efficientnet_b3(weights="IMAGENET1K_V1")

    # Replace the stock 1000-class head with our num_classes head
    in_features = model.classifier[1].in_features   # 1536 for B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    logger.debug("EfficientNet-B3 head replaced: 1536 -> %d classes", num_classes)
    return model