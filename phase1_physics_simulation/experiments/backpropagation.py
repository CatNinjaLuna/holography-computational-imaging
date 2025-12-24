"""
Hologram Back-Propagation Experiment - Multi-Framework Comparison

Purpose:
    Demonstrate naive back-propagation from hologram intensity to object plane
    using multiple state-of-the-art frameworks. Shows limitations of simple
    intensity-only reconstruction (phase is lost).

Algorithm Steps:
    1. Loads hologram intensity from saved .npy file
    2. Builds naive field: sqrt(I) with zero phase assumption
    3. Back-propagates to object plane using negative distance (-z)
    4. Plots amplitude/phase: artifacts expected (twin image, interference)

Quantitative Metrics Computed:
    - Amplitude MSE (Mean Squared Error)
    - Amplitude MAE (Mean Absolute Error)
    - Amplitude PSNR (Peak Signal-to-Noise Ratio in dB)
    - Phase MAE (with proper phase wrapping handling)

Expected Results:
    - Twin image artifacts due to missing phase information
    - Interference patterns from conjugate reconstruction
    - Demonstrates fundamental limitation of intensity-only holography

This is the inverse problem: Hologram → Object
Later phases will implement advanced reconstruction algorithms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from phase1_physics_simulation.optics.fresnel import fresnel_propagate_fft, intensity

# Optional imports for advanced frameworks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. HoloTorch back-propagation will be skipped.")

try:
    import odak
    ODAK_AVAILABLE = True
except ImportError:
    ODAK_AVAILABLE = False
    warnings.warn("Odak not available. Odak back-propagation will be skipped.")

try:
    import LightPipes as lp
    LIGHTPIPES_AVAILABLE = True
except ImportError:
    LIGHTPIPES_AVAILABLE = False
    warnings.warn("LightPipes not available. LightPipes back-propagation will be skipped.")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def backprop_numpy(holo, wavelength, z, dx, N):
    """NumPy baseline back-propagation"""
    # Naive field at sensor: sqrt intensity, zero phase
    u_sensor = np.sqrt(np.maximum(holo, 0.0)).astype(np.float32) + 0j
    
    # Back-propagate to object plane
    u_back = fresnel_propagate_fft(u_sensor, wavelength=wavelength, z=-z, dx=dx)
    
    amp_rec = np.abs(u_back)
    phase_rec = np.angle(u_back)
    
    return amp_rec, phase_rec

def backprop_odak(holo, wavelength, z, dx, N):
    """Odak back-propagation"""
    if not ODAK_AVAILABLE:
        return None, None
    
    try:
        # Create field from intensity (naive: assume zero phase)
        u_sensor = np.sqrt(np.maximum(holo, 0.0)).astype(np.complex64)
        
        # Back-propagate using Odak
        u_back = odak.learn.wave.propagate_beam(
            field=u_sensor,
            k=2 * np.pi / wavelength,
            distance=-z,
            dx=dx,
            wavelength=wavelength
        )
        
        amp_rec = np.abs(u_back)
        phase_rec = np.angle(u_back)
        
        return amp_rec, phase_rec
    except Exception as e:
        warnings.warn(f"Odak back-propagation failed: {e}")
        return None, None

def backprop_holotorch(holo, wavelength, z, dx, N):
    """HoloTorch (PyTorch) back-propagation"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create field from intensity (naive: assume zero phase)
        u_sensor = torch.from_numpy(np.sqrt(np.maximum(holo, 0.0))).float().to(device)
        u_sensor = torch.complex(u_sensor, torch.zeros_like(u_sensor))
        
        # Fresnel propagation kernel (back-propagation: negative z)
        k = 2 * np.pi / wavelength
        fx = torch.fft.fftfreq(N, dx).to(device)
        fy = torch.fft.fftfreq(N, dx).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        H = torch.exp(1j * k * (-z) * torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2 + 0j))
        
        # Apply propagation
        U_sensor = torch.fft.fft2(u_sensor)
        U_back = torch.fft.ifft2(U_sensor * H)
        
        u_back = U_back.cpu().numpy()
        amp_rec = np.abs(u_back)
        phase_rec = np.angle(u_back)
        
        return amp_rec, phase_rec
    except Exception as e:
        warnings.warn(f"HoloTorch back-propagation failed: {e}")
        return None, None

def backprop_lightpipes(holo, wavelength, z, dx, N):
    """LightPipes back-propagation"""
    if not LIGHTPIPES_AVAILABLE:
        return None, None
    
    try:
        size = N * dx
        
        # Initialize field with intensity
        F = lp.Begin(size, wavelength, N)
        
        # Set intensity (sqrt for amplitude)
        amp = np.sqrt(np.maximum(holo, 0.0))
        F = lp.SubIntensity(F, 1.0 - amp)
        
        # Back-propagate (negative distance)
        F = lp.Fresnel(F, -z)
        
        # Extract results
        intensity_back = lp.Intensity(F)
        phase_back = lp.Phase(F)
        
        amp_rec = np.sqrt(intensity_back)
        phase_rec = phase_back
        
        return amp_rec, phase_rec
    except Exception as e:
        warnings.warn(f"LightPipes back-propagation failed: {e}")
        return None, None

def main():
    wavelength = 532e-9
    dx = 1.12e-6
    z = 5e-3

    fig_dir = os.path.join("phase1_physics_simulation", "figures")
    ensure_dir(fig_dir)

    # Load hologram and ground truth
    holo = np.load(os.path.join(fig_dir, "hologram_numpy.npy"))  # Use NumPy hologram as reference
    amp_gt = np.load(os.path.join(fig_dir, "gt_amp.npy"))
    phase_gt = np.load(os.path.join(fig_dir, "gt_phase.npy"))
    
    N = holo.shape[0]

    print("\n" + "="*60)
    print("HOLOGRAM BACK-PROPAGATION - MULTI-FRAMEWORK COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. NumPy (baseline)
    print("\n[1/4] Running NumPy back-propagation...")
    amp_numpy, phase_numpy = backprop_numpy(holo, wavelength, z, dx, N)
    results['numpy'] = (amp_numpy, phase_numpy)
    print("✓ NumPy back-propagation complete")
    
    # 2. Odak
    print("\n[2/4] Running Odak back-propagation...")
    amp_odak, phase_odak = backprop_odak(holo, wavelength, z, dx, N)
    if amp_odak is not None:
        results['odak'] = (amp_odak, phase_odak)
        print("✓ Odak back-propagation complete")
    else:
        print("✗ Odak back-propagation skipped")
    
    # 3. HoloTorch
    print("\n[3/4] Running HoloTorch (PyTorch) back-propagation...")
    amp_torch, phase_torch = backprop_holotorch(holo, wavelength, z, dx, N)
    if amp_torch is not None:
        results['holotorch'] = (amp_torch, phase_torch)
        print("✓ HoloTorch back-propagation complete")
    else:
        print("✗ HoloTorch back-propagation skipped")
    
    # 4. LightPipes
    print("\n[4/4] Running LightPipes back-propagation...")
    amp_lp, phase_lp = backprop_lightpipes(holo, wavelength, z, dx, N)
    if amp_lp is not None:
        results['lightpipes'] = (amp_lp, phase_lp)
        print("✓ LightPipes back-propagation complete")
    else:
        print("✗ LightPipes back-propagation skipped")
    
    # ===== VISUALIZATIONS =====
    
    # Individual framework visualizations
    for name, (amp_rec, phase_rec) in results.items():
        # Normalize for visualization
        amp_rec_vis = amp_rec / (amp_rec.max() + 1e-12)
        
        # Reconstructed amplitude
        plt.figure()
        plt.imshow(amp_rec_vis, cmap="gray")
        plt.title(f"Back-propagated Amplitude - {name.capitalize()}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"backprop_amplitude_{name}.png"), dpi=200)
        plt.close()
        
        # Reconstructed phase
        plt.figure()
        plt.imshow(phase_rec, cmap="twilight")
        plt.title(f"Back-propagated Phase - {name.capitalize()}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"backprop_phase_{name}.png"), dpi=200)
        plt.close()
    
    # Framework comparison: Amplitude
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    if n_results == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (name, (amp_rec, phase_rec)) in enumerate(results.items()):
        amp_rec_vis = amp_rec / (amp_rec.max() + 1e-12)
        
        im0 = axes[0, idx].imshow(amp_rec_vis, cmap="gray")
        axes[0, idx].set_title(f"{name.capitalize()}\nAmplitude")
        plt.colorbar(im0, ax=axes[0, idx])
        
        im1 = axes[1, idx].imshow(phase_rec, cmap="twilight")
        axes[1, idx].set_title(f"{name.capitalize()}\nPhase")
        plt.colorbar(im1, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "backprop_framework_comparison.png"), dpi=200)
    plt.close()
    
    # Ground truth vs reconstructions comparison (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Ground truth
    im0 = axes[0, 0].imshow(amp_gt, cmap="gray")
    axes[0, 0].set_title("Ground Truth Amplitude")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(phase_gt, cmap="twilight")
    axes[0, 1].set_title("Ground Truth Phase")
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(holo / holo.max(), cmap="gray")
    axes[0, 2].set_title("Hologram Intensity")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Rows 2-3: Reconstructions from different frameworks
    framework_list = list(results.items())
    for idx in range(min(6, len(framework_list))):
        row = 1 + idx // 3
        col = idx % 3
        name, (amp_rec, phase_rec) = framework_list[idx]
        
        amp_rec_vis = amp_rec / (amp_rec.max() + 1e-12)
        
        if idx % 2 == 0:  # Show amplitude
            im = axes[row, col].imshow(amp_rec_vis, cmap="gray")
            axes[row, col].set_title(f"{name.capitalize()}\nReconstructed Amplitude")
        else:  # Show phase
            im = axes[row, col].imshow(phase_rec, cmap="twilight")
            axes[row, col].set_title(f"{name.capitalize()}\nReconstructed Phase")
        plt.colorbar(im, ax=axes[row, col])
    
    # Hide unused subplots
    for idx in range(len(framework_list), 6):
        row = 1 + idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gt_vs_backprop_comparison.png"), dpi=200)
    plt.close()
    
    # Side-by-side GT vs best reconstruction (NumPy)
    amp_numpy, phase_numpy = results['numpy']
    amp_numpy_vis = amp_numpy / (amp_numpy.max() + 1e-12)
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(amp_gt, cmap="gray")
    plt.title("GT Amplitude")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(phase_gt, cmap="twilight")
    plt.title("GT Phase")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(amp_numpy_vis, cmap="gray")
    plt.title("Naive Backprop Amplitude")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(phase_numpy, cmap="twilight")
    plt.title("Naive Backprop Phase")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gt_vs_naive_backprop.png"), dpi=200)
    plt.close()
    
    # Cross-section analysis
    center = N // 2
    
    plt.figure(figsize=(14, 10))
    
    # Amplitude profiles
    plt.subplot(3, 2, 1)
    plt.plot(amp_gt[center, :], 'k-', linewidth=2, label='Ground Truth')
    for name, (amp_rec, _) in results.items():
        plt.plot(amp_rec[center, :], '--', alpha=0.7, label=name.capitalize())
    plt.title("Amplitude Profiles (center row)")
    plt.xlabel("Pixel")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Phase profiles
    plt.subplot(3, 2, 2)
    plt.plot(phase_gt[center, :], 'k-', linewidth=2, label='Ground Truth')
    for name, (_, phase_rec) in results.items():
        plt.plot(phase_rec[center, :], '--', alpha=0.7, label=name.capitalize())
    plt.title("Phase Profiles (center row)")
    plt.xlabel("Pixel")
    plt.ylabel("Phase [rad]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Amplitude error profiles
    plt.subplot(3, 2, 3)
    for name, (amp_rec, _) in results.items():
        # Normalize both for fair comparison
        amp_gt_norm = amp_gt / (amp_gt.max() + 1e-12)
        amp_rec_norm = amp_rec / (amp_rec.max() + 1e-12)
        error = np.abs(amp_gt_norm - amp_rec_norm)
        plt.plot(error[center, :], label=name.capitalize(), alpha=0.7)
    plt.title("Amplitude Error Profiles")
    plt.xlabel("Pixel")
    plt.ylabel("Absolute Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Phase error profiles
    plt.subplot(3, 2, 4)
    for name, (_, phase_rec) in results.items():
        error = np.abs(phase_gt - phase_rec)
        # Handle phase wrapping
        error = np.minimum(error, 2*np.pi - error)
        plt.plot(error[center, :], label=name.capitalize(), alpha=0.7)
    plt.title("Phase Error Profiles (wrapped)")
    plt.xlabel("Pixel")
    plt.ylabel("Absolute Error [rad]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2D Error maps (amplitude)
    plt.subplot(3, 2, 5)
    amp_gt_norm = amp_gt / (amp_gt.max() + 1e-12)
    amp_numpy_norm = amp_numpy / (amp_numpy.max() + 1e-12)
    error_2d = np.abs(amp_gt_norm - amp_numpy_norm)
    im = plt.imshow(error_2d, cmap="hot")
    plt.title("Amplitude Error Map (NumPy)")
    plt.colorbar(im)
    
    # 2D Error maps (phase)
    plt.subplot(3, 2, 6)
    error_phase_2d = np.abs(phase_gt - phase_numpy)
    error_phase_2d = np.minimum(error_phase_2d, 2*np.pi - error_phase_2d)
    im = plt.imshow(error_phase_2d, cmap="hot")
    plt.title("Phase Error Map (NumPy)")
    plt.colorbar(im, label="rad")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "backprop_error_analysis.png"), dpi=200)
    plt.close()
    
    # Quantitative metrics
    print("\n" + "="*60)
    print("RECONSTRUCTION METRICS")
    print("="*60)
    
    for name, (amp_rec, phase_rec) in results.items():
        # Normalize for fair comparison
        amp_gt_norm = amp_gt / (amp_gt.max() + 1e-12)
        amp_rec_norm = amp_rec / (amp_rec.max() + 1e-12)
        
        # Amplitude metrics
        amp_mse = np.mean((amp_gt_norm - amp_rec_norm)**2)
        amp_mae = np.mean(np.abs(amp_gt_norm - amp_rec_norm))
        amp_psnr = 10 * np.log10(1.0 / (amp_mse + 1e-10))
        
        # Phase metrics (handle wrapping)
        phase_error = np.abs(phase_gt - phase_rec)
        phase_error = np.minimum(phase_error, 2*np.pi - phase_error)
        phase_mae = np.mean(phase_error)
        
        print(f"\n{name.capitalize()}:")
        print(f"  Amplitude MSE: {amp_mse:.6f}")
        print(f"  Amplitude MAE: {amp_mae:.6f}")
        print(f"  Amplitude PSNR: {amp_psnr:.2f} dB")
        print(f"  Phase MAE: {phase_mae:.4f} rad ({np.degrees(phase_mae):.2f}°)")
    
    # Final summary
    print("\n" + "="*60)
    print("BACK-PROPAGATION COMPLETE")
    print("="*60)
    print("\nFrameworks tested:")
    for name in results.keys():
        print(f"  ✓ {name.capitalize()}")
    
    print("\nSaved visualizations:")
    print("  - backprop_amplitude_[framework].png (per framework)")
    print("  - backprop_phase_[framework].png (per framework)")
    print("  - backprop_framework_comparison.png")
    print("  - gt_vs_backprop_comparison.png")
    print("  - gt_vs_naive_backprop.png")
    print("  - backprop_error_analysis.png")
    
    print(f"\nFigures directory: {fig_dir}")
    print("\nNote: Artifacts are expected - naive back-propagation cannot")
    print("      recover phase from intensity alone. This motivates Phase 2")
    print("      (iterative algorithms) and Phase 3 (learning-based methods).")
    print("="*60)

if __name__ == "__main__":
    main()
