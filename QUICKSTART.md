# Quick Start Guide - Using the Virtual Environment

## ✅ Your virtual environment is set up at:

`/Users/carolina1650/holography-computational-imaging/venv`

## How to Use the Environment

### Option 1: Activate in your current terminal

```bash
cd /Users/carolina1650/holography-computational-imaging
source venv/bin/activate
```

You'll see `(venv)` in your prompt. Now you can run Python with all packages:

```bash
python -m phase1_physics_simulation.experiments.simulate_hologram
```

To deactivate:

```bash
deactivate
```

### Option 2: Use the activation script

```bash
cd /Users/carolina1650/holography-computational-imaging
source activate_env.sh
```

### Option 3: Run Python directly from venv (no activation needed)

```bash
cd /Users/carolina1650/holography-computational-imaging
./venv/bin/python -m phase1_physics_simulation.experiments.simulate_hologram
```

## Verify Installation

After activating, verify packages:

```bash
source venv/bin/activate
python -c "import numpy, matplotlib, torch, odak, LightPipes; print('All packages OK!')"
```

## Run Experiments

```bash
# Activate environment first
source venv/bin/activate

# Run validation
python -m phase1_physics_simulation.experiments.validation

# Run hologram simulation (multi-framework)
python -m phase1_physics_simulation.experiments.simulate_hologram

# Run back-propagation (multi-framework)
python -m phase1_physics_simulation.experiments.backpropagation
```

## Installed Packages

-  ✅ **numpy** (2.2.6)
-  ✅ **matplotlib** (3.10.8)
-  ✅ **torch** (2.9.1) - PyTorch for HoloTorch framework
-  ✅ **odak** (0.2.6) - Computer-generated holography toolkit
-  ✅ **LightPipes** (2.1.5) - Classical Fourier optics

## GPU Acceleration

Your Mac has Apple Silicon (M1/M2/M3). PyTorch can use Metal for GPU acceleration:

```python
import torch
print(f"Metal (GPU) available: {torch.backends.mps.is_available()}")

# Use MPS device in your code
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

## Troubleshooting

### Issue: "ModuleNotFoundError" when running Python

**Solution:** Make sure you activated the environment:

```bash
source venv/bin/activate
```

### Issue: Want to use this in VS Code

1. Open VS Code in the project directory
2. Press `Cmd+Shift+P` → "Python: Select Interpreter"
3. Choose: `./venv/bin/python` (Python 3.11.14)

### Issue: Want to add more packages

```bash
source venv/bin/activate
pip install <package-name>
```

## Why This Works (and conda didn't)

Your conda installation had a corrupted Python environment (missing `uu` module). This fresh Homebrew Python 3.11 with its own venv is completely independent and clean.

You can still use conda for other projects - just use this venv for holography work!
