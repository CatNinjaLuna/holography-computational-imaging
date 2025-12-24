# Installation Guide

## ✅ Environment Successfully Set Up!

A clean Python 3.11 virtual environment has been created at:
`/Users/carolina1650/holography-computational-imaging/venv`

**Quick start:** See [QUICKSTART.md](QUICKSTART.md) for usage instructions.

---

## What Was Installed

The following packages are now available in your local environment:

-  **numpy** (2.2.6) - Numerical computing
-  **matplotlib** (3.10.8) - Visualization
-  **torch** (2.9.1) - PyTorch for HoloTorch framework
-  **odak** (0.2.6) - Computer-generated holography toolkit
-  **LightPipes** (2.1.5) - Classical Fourier optics simulation

---

## How to Use

### Activate the environment:

```bash
cd /Users/carolina1650/holography-computational-imaging
source venv/bin/activate
```

You'll see `(venv)` in your prompt.

### Run experiments:

```bash
# Validation test
python -m phase1_physics_simulation.experiments.validation

# Multi-framework hologram simulation
python -m phase1_physics_simulation.experiments.simulate_hologram

# Multi-framework back-propagation
python -m phase1_physics_simulation.experiments.backpropagation
```

### Deactivate when done:

```bash
deactivate
```

---

## Background: Why We Used Homebrew Python

Your conda installation had a critical issue (`ModuleNotFoundError: No module named 'uu'`), which is a known problem with some conda/Python 3.10 combinations.

**Solution:** We installed a fresh Python 3.11 via Homebrew and created a clean virtual environment specifically for this project. This is completely independent of your conda setup.

---

## Original Installation Attempts (for reference)

### Issues Encountered

source venv_holography/bin/activate

# Upgrade pip

pip install --upgrade pip

# Install requirements

pip install -r phase1_physics_simulation/requirementx.txt

````

### Option 3: Install Packages Manually (Quick Fix)

Since your conda/pip is broken, manually download and install:

```bash
# Install PyTorch
python3 -m pip install --user torch torchvision torchaudio

# Install Odak
python3 -m pip install --user odak

# Install LightPipes
python3 -m pip install --user LightPipes

# Install numpy and matplotlib (if not already present)
python3 -m pip install --user numpy matplotlib
````

### Option 4: Use Homebrew Python (macOS)

```bash
# Install Python via Homebrew
brew install python@3.11

# Create venv with Homebrew Python
/opt/homebrew/bin/python3.11 -m venv venv_holography

# Activate
source venv_holography/bin/activate

# Install packages
pip install -r phase1_physics_simulation/requirementx.txt
```

## Required Packages

-  **numpy** (≥1.21.0): Numerical computing
-  **matplotlib** (≥3.5.0): Visualization
-  **torch** (≥2.0.0): PyTorch for HoloTorch framework
-  **odak** (≥0.2.0): Computer-generated holography toolkit
-  **LightPipes** (≥2.1.0): Classical Fourier optics simulation

## Verification

After installation, verify packages are available:

```python
import numpy
import matplotlib
import torch
import odak
import LightPipes

print("All packages successfully imported!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Running Simulations

Once packages are installed:

```bash
# Run validation
cd /Users/carolina1650/holography-computational-imaging
python -m phase1_physics_simulation.experiments.validation

# Run hologram simulation (multi-framework)
python -m phase1_physics_simulation.experiments.simulate_hologram

# Run back-propagation (multi-framework)
python -m phase1_physics_simulation.experiments.backpropagation
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'uu'`

This is a known issue with Python 3.10 and certain conda versions. The `uu` module was deprecated and removed in Python 3.11+.

**Solutions:**

1. Update conda: `conda update conda`
2. Use Python 3.9 or 3.11+
3. Use system Python or Homebrew Python instead of conda

### Issue: Conda/pip commands fail

Your conda installation may be corrupted. Consider:

1. Reinstalling miniconda/anaconda
2. Using Python venv with system Python
3. Using `python3 -m pip` instead of `pip` command

### Issue: GPU not detected for PyTorch

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available())  # Should be True if NVIDIA GPU + CUDA installed
```

For macOS with Apple Silicon (M1/M2):

```python
print(torch.backends.mps.is_available())  # Should be True for Metal acceleration
```
