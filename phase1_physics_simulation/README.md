# Phase 1 — Physics & Simulation

This phase establishes the **physical foundations** of holography by explicitly modeling wave propagation and hologram formation.

No learning is used in this phase.

---

## Objectives

-  Understand holography as a wave-interference phenomenon
-  Implement Fresnel diffraction numerically
-  Generate synthetic holograms with known ground truth
-  Validate numerical propagation against analytical solutions

---

## Key Concepts

-  Complex optical fields (amplitude + phase)
-  Fresnel diffraction integral
-  Sampling conditions and propagation distance
-  Inline holography geometry

---

## Contents

Typical components in this phase include:

-  Fresnel propagation (FFT-based)
-  Hologram intensity formation
-  Back-propagation
-  Phase visualization

---

## Validation

Validation is critical in this phase:

-  Compare numerical propagation with analytical Fresnel kernels
-  Verify energy conservation
-  Confirm phase consistency under forward/back propagation

This phase serves as the **ground truth generator** for later phases.

---

## State-of-the-Art Simulation Tools

### 1. **Odak** ⭐ (Python)

-  **By**: Koç University & META Reality Labs
-  **Features**: Full wave optics toolkit, computer-generated holography (CGH), ray tracing + wave optics
-  **Best for**: Industry-standard CGH algorithms, 3D holography
-  **Repository**: `kaanaksit/odak` on GitHub
-  **Link**: https://github.com/kaanaksit/odak

### 2. **HoloTorch** (PyTorch-based)

-  **By**: Computational Imaging Labs
-  **Features**: GPU-accelerated wave propagation, differentiable operations, neural holography support
-  **Best for**: Learning-based holography, automatic differentiation
-  **Status**: Research-grade, cutting-edge

### 3. **LightPipes** (Python)

-  **Features**: Classical Fourier optics, beam propagation, diffraction modeling
-  **Best for**: Educational purposes, prototyping optical systems
-  **Link**: https://opticspy.github.io/lightpipes/

### 4. **Pyoptica** (Python)

-  **Features**: Scalar and vector diffraction, optical system simulation
-  **Best for**: Academic research, teaching
-  **Type**: Open-source educational toolkit

### 5. **MATLAB Holography Toolboxes**

-  **Features**: Established algorithms (Gerchberg-Saxton, double-phase encoding), comprehensive documentation
-  **Best for**: Academic research with mature implementations
-  **Status**: Widely used in academic publications

**Current Implementation**: This phase uses NumPy + Matplotlib for fundamental wave propagation simulation, providing a solid foundation before moving to GPU-accelerated tools.

---

## Installation

### Prerequisites

-  Python 3.11 (recommended via Homebrew on macOS)
-  Virtual environment support

### Setup Instructions

1. **Install Python 3.11** (if not already installed):

   ```bash
   brew install python@3.11
   ```

2. **Create virtual environment**:

   ```bash
   # From project root
   /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv
   ```

3. **Activate virtual environment**:

   ```bash
   source venv/bin/activate
   ```

4. **Install required packages**:
   ```bash
   pip install numpy matplotlib torch odak LightPipes plotly kaleido
   ```

### Installed Packages

-  **numpy** (2.2.6): Numerical computing foundation
-  **matplotlib** (3.10.8): Visualization framework
-  **torch** (2.9.1): PyTorch for HoloTorch backend (Metal GPU support on Apple Silicon)
-  **odak** (0.2.6): Industry-standard CGH toolkit
-  **LightPipes** (2.1.5): Classical Fourier optics simulation
-  **plotly** (5.24.1): Optional interactive 3D visualizations for Odak
-  **kaleido** (0.2.1): Optional Plotly image export

### Verification

Check that all packages work:

```bash
venv/bin/python -c "import numpy; import matplotlib; import torch; import odak; import LightPipes as lp; print('✓ All packages ready!')"
```

Check GPU support (Apple Silicon):

```bash
venv/bin/python -c "import torch; print(f'Metal GPU available: {torch.backends.mps.is_available()}')"
```
