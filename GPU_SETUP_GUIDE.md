# GPU Acceleration Setup for PPE Detection System

## Current Status
✅ **Code Updated**: All Python files have been modified to support GPU acceleration
❌ **GPU Not Available**: Current environment only has CPU version of PyTorch

## Modified Files
The following files have been updated to support GPU acceleration:

1. **main.py** - Added GPU support for basic PPE detection
2. **YOLO_Video.py** - Added GPU support for video detection with alerts
3. **trial1.py** - Added GPU support for trial detection script

## Changes Made

### Code Modifications
All files now include:
```python
import torch

# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Move model to GPU if available
model.to(device)

# Run inference with explicit device specification
results = model(img, stream=True, device=device)
```

### Automatic Device Detection
- The code automatically detects if CUDA/GPU is available
- Falls back to CPU if GPU is not available
- Prints the device being used for transparency

## GPU Setup Instructions

### Prerequisites
1. **NVIDIA GPU**: You need a CUDA-compatible NVIDIA graphics card
2. **NVIDIA Drivers**: Install latest NVIDIA drivers for your GPU

### Step 1: Install CUDA Toolkit
Download and install CUDA toolkit from NVIDIA:
- Visit: https://developer.nvidia.com/cuda-downloads
- Choose your OS and follow installation instructions

### Step 2: Install PyTorch with CUDA Support
Replace the current CPU-only PyTorch with GPU-enabled version:

```powershell
# For CUDA 11.8 (recommended for compatibility)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (latest)
pip uninstall torch torchvision torchaudio  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation
Run the test script to verify GPU setup:
```powershell
python test_gpu.py
```

## Expected Performance Improvements

With GPU acceleration, you can expect:
- **2-5x faster inference** for YOLO models
- **Real-time processing** for video streams
- **Better resource utilization** for multiple concurrent detections

## Current Performance (CPU Only)
- Average inference time: ~0.37 seconds per frame
- This will significantly improve with GPU acceleration

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Driver compatibility**: Ensure NVIDIA drivers are up to date
3. **CUDA version mismatch**: Match PyTorch CUDA version with installed CUDA toolkit

### Verification Commands
```powershell
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU details
python -c "import torch; print(torch.cuda.get_device_name(0))" 
```

## Usage After GPU Setup

Once GPU is properly installed, simply run your existing scripts:
```powershell
python main.py          # Basic detection with GPU
python app.py           # Flask web app with GPU acceleration  
python YOLO_Video.py    # Video processing with GPU
```

The code will automatically detect and use GPU when available!

## Monitoring GPU Usage

To monitor GPU usage during detection:
```powershell
# Install nvidia-ml-py3 for GPU monitoring
pip install nvidia-ml-py3

# Monitor GPU usage
nvidia-smi
```

## Notes
- GPU acceleration is most beneficial for video processing and real-time detection
- For single image processing, the improvement may be less noticeable due to overhead
- The modified code maintains full backward compatibility with CPU-only systems
