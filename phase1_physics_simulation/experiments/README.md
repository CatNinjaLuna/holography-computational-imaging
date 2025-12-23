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
