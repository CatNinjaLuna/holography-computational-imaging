# Experiments

This folder contains validation and simulation experiments for the physics-based holography implementation.

## Validation Results

### validation.py - Plane Wave Propagation Test

**Purpose**: Validates that a plane wave remains uniform after propagation through free space.

**Parameters**:

-  Wavelength: 532 nm (green laser)
-  Grid size: 1024 × 1024 pixels
-  Pixel pitch: 1.12 µm
-  Propagation distance: 5 mm
-  Field of view: ~1.15 mm

**Results** (December 22, 2025):

```
I mean: 1.0
I std: 0.0
std/mean: 0.0
```

**Interpretation**: Perfect result - the plane wave maintains uniform intensity after propagation, confirming the Fresnel propagation implementation is physically correct.

**Output**: `figures/figure1_plane_wave_validation.png`

---

## Hologram Simulation

### simulate_hologram.py - Multi-Framework Hologram Generation

**Purpose**: Generate synthetic holograms from known ground-truth objects using multiple state-of-the-art frameworks for comparison.

**Frameworks Tested**:

-  NumPy (baseline implementation)
-  Odak (industry-standard CGH toolkit)
-  HoloTorch (PyTorch-based differentiable holography)
-  LightPipes (classical Fourier optics)

**Parameters**:

-  Wavelength: 532 nm (green laser)
-  Grid size: 1024 × 1024 pixels
-  Pixel pitch: 1.12 µm
-  Propagation distance: 5 mm

**Outputs**:

-  Ground truth amplitude and phase arrays (.npy)
-  Hologram intensity patterns per framework
-  Framework comparison visualizations
-  Comprehensive analysis plots (9-panel with cross-sections, frequency spectra, statistics)
-  Energy conservation validation

**Key Visualizations**:

-  `hologram_[framework].png` - Intensity patterns per framework
-  `framework_comparison.png` - Side-by-side comparison
-  `comprehensive_analysis.png` - Detailed 9-panel analysis
-  `frequency_spectrum_comparison.png` - 2D FFT comparison

---

## Back-Propagation

### backpropagation.py - Multi-Framework Naive Reconstruction

**Purpose**: Demonstrate naive back-propagation from hologram intensity to object plane using multiple frameworks. Shows fundamental limitations of intensity-only reconstruction (phase information is lost).

**Frameworks Tested**:

-  NumPy (baseline implementation)
-  Odak
-  HoloTorch (PyTorch with GPU support)
-  LightPipes

**Quantitative Metrics Computed**:

-  **Amplitude MSE** (Mean Squared Error)
-  **Amplitude MAE** (Mean Absolute Error)
-  **Amplitude PSNR** (Peak Signal-to-Noise Ratio in dB)
-  **Phase MAE** (with proper phase wrapping handling)

**Key Findings**:

-  Naive back-propagation cannot recover phase from intensity alone
-  Artifacts are expected and demonstrate the need for advanced algorithms
-  This motivates Phase 2 (iterative algorithms) and Phase 3 (learning-based methods)

**Key Visualizations**:

-  `backprop_amplitude_[framework].png` - Reconstructed amplitudes
-  `backprop_phase_[framework].png` - Reconstructed phases
-  `backprop_framework_comparison.png` - Side-by-side framework comparison
-  `gt_vs_backprop_comparison.png` - 3×3 grid comparing ground truth with reconstructions
-  `backprop_error_analysis.png` - 6-panel comprehensive error analysis including:
   -  Cross-section profiles (amplitude & phase)
   -  Error profiles with proper phase wrapping
   -  2D error maps

**Educational Value**: Clearly demonstrates the inverse problem challenge and why sophisticated reconstruction algorithms are necessary for holographic imaging.

---

## Generated Visualizations

For detailed analysis of all generated figures, see **[`../figures/README.md`](../figures/README.md)**.

**Summary**: The simulation generates 19 PNG visualizations including:

-  Ground truth amplitude/phase
-  Hologram patterns from 4 frameworks (NumPy, Odak, HoloTorch, LightPipes)
-  Framework comparisons and comprehensive analysis plots
-  Complex field visualizations (amplitude/phase at z=5mm)
-  Frequency spectrum analysis
-  Validation results

**Key Results**:

-  ✅ Energy Conservation: NumPy (100%), HoloTorch (100%), Odak (97.8%)
-  ✅ Inter-framework Agreement: <2% difference between NumPy/Odak/HoloTorch
-  ✅ All frameworks correctly simulate Fresnel diffraction
-  ✅ Implementation validated against multiple independent frameworks
