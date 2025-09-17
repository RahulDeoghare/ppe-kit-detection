#!/usr/bin/env python3
"""
Script to test GPU availability and YOLO model performance
"""

import torch
import time
from ultralytics import YOLO
import cv2
import numpy as np

def test_gpu_availability():
    """Test if GPU is available and working"""
    print("=== GPU Availability Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU available. Running on CPU.")
    
    return torch.cuda.is_available()

def benchmark_model_performance(model_path="best.pt", use_gpu=True):
    """Benchmark YOLO model performance on CPU vs GPU"""
    print("\n=== Model Performance Benchmark ===")
    
    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test on CPU
    print("Testing on CPU...")
    model_cpu = YOLO(model_path)
    model_cpu.to('cpu')
    
    start_time = time.time()
    for _ in range(10):  # Run 10 inference iterations
        results = model_cpu(dummy_image, verbose=False)
    cpu_time = (time.time() - start_time) / 10
    print(f"Average CPU inference time: {cpu_time:.4f} seconds")
    
    # Test on GPU if available
    if use_gpu and torch.cuda.is_available():
        print("Testing on GPU...")
        model_gpu = YOLO(model_path)
        model_gpu.to('cuda')
        
        # Warm up GPU
        for _ in range(3):
            results = model_gpu(dummy_image, verbose=False)
        
        start_time = time.time()
        for _ in range(10):  # Run 10 inference iterations
            results = model_gpu(dummy_image, verbose=False)
        gpu_time = (time.time() - start_time) / 10
        print(f"Average GPU inference time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.2f}x faster than CPU")
    else:
        print("GPU not available for testing")

def main():
    """Main function to run all tests"""
    gpu_available = test_gpu_availability()
    
    # Check if model files exist
    model_paths = ["best.pt", "YOLO-Weights/ppe.pt", "yolov8n.pt"]
    available_model = None
    
    for model_path in model_paths:
        try:
            model = YOLO(model_path)
            available_model = model_path
            print(f"\nFound model: {model_path}")
            break
        except:
            continue
    
    if available_model:
        benchmark_model_performance(available_model, gpu_available)
    else:
        print("\nNo YOLO model found. Please ensure model files are present.")
    
    print("\n=== GPU Setup Instructions ===")
    if not gpu_available:
        print("To enable GPU acceleration:")
        print("1. Install CUDA toolkit from NVIDIA")
        print("2. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Or for latest CUDA version:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("4. Restart your application")
    else:
        print("GPU is ready! Your models should now run faster.")

if __name__ == "__main__":
    main()
