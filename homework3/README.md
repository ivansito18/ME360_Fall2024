# PyTorch Installation Guide

This guide covers how to install [PyTorch](https://pytorch.org/get-started/locally/) on Windows, macOS, and Linux.

*Google colab has already installed with PyTorch by default.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Windows

1. Open a command prompt or PowerShell.
2. Install PyTorch using pip:
   ```
   pip3 install torch torchvision torchaudio
   ```

## macOS˚

1. Open Terminal.
2. Install PyTorch using pip:
   ```
   pip3 install torch torchvision torchaudio
   ```

## Linux

1. Open a terminal.
2. Install PyTorch using pip:
   ```
   pip3 install torch torchvision torchaudio
   ```

## Verification

To verify the installation, run Python and try importing torch:

```python
import torch
print(torch.__version__)
```

## GPU Support

For GPU support, visit the official PyTorch website (https://pytorch.org/) and use the installation selector to get the appropriate command for your system and CUDA version.

## Troubleshooting

If you encounter issues:
1. Ensure you have the latest pip version: `pip3 install --upgrade pip`
2. Check the official PyTorch documentation for system-specific instructions.
3. For GPU issues, verify your CUDA installation and compatibility.

For more detailed instructions and options, refer to the official PyTorch documentation: https://pytorch.org/get-started/locally/
