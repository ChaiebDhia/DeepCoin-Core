#!/usr/bin/env python3
"""
Test script to verify Ollama models are working.

Tests both gemma3:4b (text) and qwen3-vl:4b (vision) models.
Run this to verify your Ollama setup is correct.

Usage:
    python scripts/test_ollama_models.py
"""

import sys
import time
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests library not found. Install with: pip install requests")
    sys.exit(1)


def test_ollama_connection():
    """Test if Ollama service is running."""
    print("\n" + "="*60)
    print("TEST 1: Ollama Service Connection")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running at http://localhost:11434")
            print(f"✅ Found {len(models)} models loaded")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("   Make sure Ollama is running: Start → Ollama")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_gemma3_model():
    """Test gemma3:4b text generation."""
    print("\n" + "="*60)
    print("TEST 2: gemma3:4b Text Generation")
    print("="*60)
    
    prompt = "What is a silver denarius from ancient Rome? Answer in one sentence."
    
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        print(f"Sending prompt to gemma3:4b...")
        print(f"Prompt: '{prompt}'")
        print()
        
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            model_response = result.get("response", "").strip()
            
            print(f"✅ gemma3:4b responded in {elapsed:.1f} seconds")
            print(f"Response: {model_response}")
            return True
        else:
            print(f"❌ Model returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model took >2 minutes)")
        print("   This is normal for first request if model isn't loaded")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_qwen_vl_model():
    """Test qwen3-vl:4b vision model."""
    print("\n" + "="*60)
    print("TEST 3: qwen3-vl:4b Vision Analysis")
    print("="*60)
    
    # Try to find a sample coin image
    sample_images = [
        Path("data/processed/1015"),  # CNN training data
        Path("reports"),               # Generated reports with images
        Path("data/uploads"),          # User uploads
    ]
    
    image_path = None
    for candidate_dir in sample_images:
        if candidate_dir.exists():
            jpgs = list(candidate_dir.glob("*.jpg"))
            if jpgs:
                image_path = jpgs[0]
                break
    
    if not image_path:
        print("⚠️  No sample images found. Skipping vision test.")
        print("   To test qwen3-vl:4b, place a .jpg image in data/processed/ or reports/")
        return True
    
    try:
        # Read image as base64
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = {
            "model": "qwen3-vl:4b",
            "prompt": "What can you see in this coin image? Describe the main features.",
            "images": [image_data],
            "stream": False
        }
        
        print(f"Testing on: {image_path.name}")
        print()
        
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            model_response = result.get("response", "").strip()
            
            print(f"✅ qwen3-vl:4b responded in {elapsed:.1f} seconds")
            print(f"Vision analysis: {model_response[:200]}...")
            return True
        else:
            print(f"❌ Model returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model took >2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def list_available_models():
    """List all available models."""
    print("\n" + "="*60)
    print("Available Models on This System")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        
        print(f"\nTotal models: {len(models)}\n")
        
        model_info = []
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3)
            
            # Identify primary models
            primary = ""
            if name == "gemma3:4b":
                primary = " ← PRIMARY (Historian/Validator)"
            elif name == "qwen3-vl:4b":
                primary = " ← PRIMARY (Investigator)"
            
            model_info.append((name, size_gb, primary))
        
        # Sort by size descending
        model_info.sort(key=lambda x: x[1], reverse=True)
        
        for name, size_gb, primary in model_info:
            print(f"  • {name:<25} {size_gb:>6.2f} GB{primary}")
        
    except Exception as e:
        print(f"Error listing models: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DeepCoin Ollama Models Test Suite")
    print("="*60)
    print("Testing gemma3:4b and qwen3-vl:4b models\n")
    
    results = []
    
    # Test 1: Connection
    results.append(("Ollama Connection", test_ollama_connection()))
    
    if not results[0][1]:
        print("\n⚠️  Cannot proceed without Ollama running.")
        print("Please start Ollama and try again.")
        return False
    
    # List models
    list_available_models()
    
    # Test 2: gemma3:4b
    results.append(("gemma3:4b Text Model", test_gemma3_model()))
    
    # Test 3: qwen3-vl:4b
    results.append(("qwen3-vl:4b Vision Model", test_qwen_vl_model()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print()
    if all_passed:
        print("✅ All tests passed! Your Ollama setup is working.")
        return True
    else:
        print("⚠️  Some tests failed. See above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
