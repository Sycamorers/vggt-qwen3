#!/bin/bash
# Fix PyTorch/TorchVision installation with CUDA support

echo "=========================================="
echo "Fixing PyTorch Installation"
echo "=========================================="
echo ""

# Check current versions
echo "Current versions:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch import failed"
pip list | grep -E "^torch "
pip list | grep -E "^torchvision"
echo ""

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "WARNING: nvidia-smi not found. Assuming CUDA 12.1"
    CUDA_VERSION="12.1"
fi
echo ""

echo "=========================================="
echo "Recommended Fix:"
echo "=========================================="
echo ""
echo "Uninstall current PyTorch packages:"
echo "  pip uninstall -y torch torchvision torchaudio"
echo ""
echo "Install PyTorch with CUDA 12.1 support:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "Or with CUDA 11.8:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "=========================================="
echo ""

read -p "Do you want to automatically fix this now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstalling current PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    
    echo ""
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo ""
    echo "Verifying installation..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
    
    echo ""
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✅ PyTorch with CUDA installed successfully!"
    else
        echo "❌ CUDA still not available. You may need to install a different version."
        echo ""
        echo "Try CUDA 11.8 instead:"
        echo "  pip uninstall -y torch torchvision torchaudio"
        echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    fi
else
    echo "Skipping automatic fix. Please run the commands manually."
fi
