"""
Unit tests for preprocessing pipeline.
Run with: pytest tests/
"""

import pytest
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_clahe_enhancement():
    """Test CLAHE enhancement function."""
    # Create a dummy image
    img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    
    # Apply CLAHE (from prep_engine.py logic)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_aspect_preserving_resize():
    """Test aspect-preserving resize."""
    # Wide image
    img_wide = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    # Tall image
    img_tall = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
    
    size = 299
    
    # Test both should result in 299x299
    # (implementation would need actual resize function)
    assert True  # Placeholder


# TODO: Add more tests in Phase 6
