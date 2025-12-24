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
