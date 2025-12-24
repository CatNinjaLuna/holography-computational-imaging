"""
Hologram Simulation Experiment - Multi-Framework Comparison

Purpose:
    Generate synthetic holograms from a known ground-truth object using 
    multiple state-of-the-art frameworks:
    1. Custom NumPy implementation (baseline)
    2. Odak - Industry-standard CGH toolkit
    3. HoloTorch - PyTorch-based differentiable holography
    4. LightPipes - Classical Fourier optics
    
    This creates reference data with known amplitude and phase for testing 
    reconstruction algorithms and comparing framework implementations.

Main Parameters:
    wavelength : 532 nm (green laser)
    N          : 1024x1024 pixels
    dx         : 1.12 µm pixel pitch
    z          : 5 mm propagation distance (object to hologram plane)

Outputs:
    - Ground truth amplitude and phase (NPY arrays + PNG visualizations)
    - Simulated hologram intensity patterns from each framework
    - Comparison visualizations
    - All saved to phase1_physics_simulation/figures/

This serves as the forward model: Object → Hologram
Later phases will solve the inverse: Hologram → Object
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from phase1_physics_simulation.optics.utils import complex_from_amp_phase
from phase1_physics_simulation.optics.fresnel import fresnel_propagate_fft, intensity
from phase1_physics_simulation.scenes.objects import make_synthetic_object

# Optional imports for advanced frameworks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. HoloTorch simulation will be skipped.")

try:
    import odak
    ODAK_AVAILABLE = True
except ImportError:
    ODAK_AVAILABLE = False
    warnings.warn("Odak not available. Odak simulation will be skipped.")

try:
    import LightPipes as lp
    LIGHTPIPES_AVAILABLE = True
except ImportError:
    LIGHTPIPES_AVAILABLE = False
    warnings.warn("LightPipes not available. LightPipes simulation will be skipped.")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def simulate_with_numpy(amp, phase, wavelength, z, dx, N):
    """Baseline NumPy implementation"""
    u0 = complex_from_amp_phase(amp, phase)
    uz = fresnel_propagate_fft(u0, wavelength=wavelength, z=z, dx=dx)
    holo = intensity(uz)
    return uz, holo

def simulate_with_odak(amp, phase, wavelength, z, dx, N):
    """Odak framework implementation"""
    if not ODAK_AVAILABLE:
        return None, None
    
    try:
        # Create complex field
        u0 = amp * np.exp(1j * phase)
        
        # Use Odak's propagation
        uz = odak.learn.wave.propagate_beam(
            field=u0,
            k=2 * np.pi / wavelength,
            distance=z,
            dx=dx,
            wavelength=wavelength
        )
        
        holo = np.abs(uz) ** 2
        return uz, holo
    except Exception as e:
        warnings.warn(f"Odak simulation failed: {e}")
        return None, None

def simulate_with_holotorch(amp, phase, wavelength, z, dx, N):
    """HoloTorch (PyTorch) implementation"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        # Convert to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        u0_real = torch.from_numpy(amp * np.cos(phase)).float().to(device)
        u0_imag = torch.from_numpy(amp * np.sin(phase)).float().to(device)
        u0 = torch.complex(u0_real, u0_imag)
        
        # Fresnel propagation using PyTorch FFT
        k = 2 * np.pi / wavelength
        fx = torch.fft.fftfreq(N, dx).to(device)
        fy = torch.fft.fftfreq(N, dx).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2 + 0j))
        
        U0 = torch.fft.fft2(u0)
        UZ = torch.fft.ifft2(U0 * H)
        
        uz = UZ.cpu().numpy()
        holo = np.abs(uz) ** 2
        return uz, holo
    except Exception as e:
        warnings.warn(f"HoloTorch simulation failed: {e}")
        return None, None

def simulate_with_lightpipes(amp, phase, wavelength, z, dx, N):
    """LightPipes implementation"""
    if not LIGHTPIPES_AVAILABLE:
        return None, None
    
    try:
        # Initialize LightPipes field
        size = N * dx
        F = lp.Begin(size, wavelength, N)
        
        # Apply amplitude and phase
        # LightPipes works differently - need to build the field
        F = lp.SubIntensity(F, 1.0 - amp)  # Set amplitude
        F = lp.SubPhase(F, -phase)  # Set phase
        
        # Propagate using Fresnel diffraction
        F = lp.Fresnel(F, z)
        
        # Extract intensity
        holo = lp.Intensity(F)
        
        # Extract complex field (if possible)
        phase_out = lp.Phase(F)
        uz = np.sqrt(holo) * np.exp(1j * phase_out)
        
        return uz, holo
    except Exception as e:
        warnings.warn(f"LightPipes simulation failed: {e}")
        return None, None

def main():
    # Parameters (same as validation)
    wavelength = 532e-9
    N = 1024
    dx = 1.12e-6
    z = 5e-3

    out_dir = os.path.join("phase1_physics_simulation", "figures")
    ensure_dir(out_dir)

    # Generate synthetic object
    amp, phase = make_synthetic_object(N, dx)
    
    # Save ground truth arrays
    np.save(os.path.join(out_dir, "gt_amp.npy"), amp)
    np.save(os.path.join(out_dir, "gt_phase.npy"), phase)

    # Run simulations with all available frameworks
    print("\n" + "="*60)
    print("HOLOGRAM SIMULATION - MULTI-FRAMEWORK COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. NumPy (baseline)
    print("\n[1/4] Running NumPy simulation...")
    uz_numpy, holo_numpy = simulate_with_numpy(amp, phase, wavelength, z, dx, N)
    results['numpy'] = (uz_numpy, holo_numpy)
    np.save(os.path.join(out_dir, "hologram_numpy.npy"), holo_numpy)
    print("✓ NumPy simulation complete")
    
    # 2. Odak
    print("\n[2/4] Running Odak simulation...")
    uz_odak, holo_odak = simulate_with_odak(amp, phase, wavelength, z, dx, N)
    if holo_odak is not None:
        results['odak'] = (uz_odak, holo_odak)
        np.save(os.path.join(out_dir, "hologram_odak.npy"), holo_odak)
        print("✓ Odak simulation complete")
    else:
        print("✗ Odak simulation skipped")
    
    # 3. HoloTorch
    print("\n[3/4] Running HoloTorch (PyTorch) simulation...")
    uz_torch, holo_torch = simulate_with_holotorch(amp, phase, wavelength, z, dx, N)
    if holo_torch is not None:
        results['holotorch'] = (uz_torch, holo_torch)
        np.save(os.path.join(out_dir, "hologram_holotorch.npy"), holo_torch)
        print("✓ HoloTorch simulation complete")
    else:
        print("✗ HoloTorch simulation skipped")
    
    # 4. LightPipes
    print("\n[4/4] Running LightPipes simulation...")
    uz_lp, holo_lp = simulate_with_lightpipes(amp, phase, wavelength, z, dx, N)
    if holo_lp is not None:
        results['lightpipes'] = (uz_lp, holo_lp)
        np.save(os.path.join(out_dir, "hologram_lightpipes.npy"), holo_lp)
        print("✓ LightPipes simulation complete")
    else:
        print("✗ LightPipes simulation skipped")
    
    # ===== VISUALIZATIONS =====
    
    # Ground truth visualizations
    plt.figure()
    plt.imshow(amp, cmap="gray")
    plt.title("Ground Truth Amplitude")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gt_amplitude.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(phase, cmap="twilight")
    plt.title("Ground Truth Phase [radians]")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gt_phase.png"), dpi=200)
    plt.close()
    
    # Individual framework visualizations
    for name, (uz, holo) in results.items():
        if holo is None:
            continue
            
        holo_vis = holo / (holo.max() + 1e-12)
        
        # Hologram intensity
        plt.figure()
        plt.imshow(holo_vis, cmap="gray")
        plt.title(f"Hologram Intensity - {name.capitalize()} (z={z*1e3:.1f}mm)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hologram_{name}.png"), dpi=200)
        plt.close()
        
        # Propagated field amplitude
        plt.figure()
        plt.imshow(np.abs(uz), cmap="viridis")
        plt.title(f"Propagated Field Amplitude - {name.capitalize()}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"field_amplitude_{name}.png"), dpi=200)
        plt.close()
        
        # Propagated field phase
        plt.figure()
        plt.imshow(np.angle(uz), cmap="twilight")
        plt.title(f"Propagated Field Phase - {name.capitalize()}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"field_phase_{name}.png"), dpi=200)
        plt.close()
    
    # Framework comparison visualization
    n_results = len(results)
    if n_results > 1:
        fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
        if n_results == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, (uz, holo)) in enumerate(results.items()):
            holo_vis = holo / (holo.max() + 1e-12)
            
            im0 = axes[0, idx].imshow(holo_vis, cmap="gray")
            axes[0, idx].set_title(f"{name.capitalize()}\nHologram Intensity")
            plt.colorbar(im0, ax=axes[0, idx])
            
            im1 = axes[1, idx].imshow(np.abs(uz), cmap="viridis")
            axes[1, idx].set_title(f"{name.capitalize()}\nField Amplitude")
            plt.colorbar(im1, ax=axes[1, idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "framework_comparison.png"), dpi=200)
        plt.close()
    
    # Comprehensive 4-panel summary (using NumPy results)
    uz_numpy, holo_numpy = results['numpy']
    holo_vis = holo_numpy / (holo_numpy.max() + 1e-12)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im0 = axes[0, 0].imshow(amp, cmap="gray")
    axes[0, 0].set_title("Ground Truth Amplitude")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(phase, cmap="twilight")
    axes[0, 1].set_title("Ground Truth Phase [rad]")
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(np.abs(uz_numpy), cmap="viridis")
    axes[1, 0].set_title(f"Propagated Amplitude (z={z*1e3:.1f}mm)")
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(holo_vis, cmap="gray")
    axes[1, 1].set_title("Hologram Intensity (normalized)")
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hologram_summary_4panel.png"), dpi=200)
    plt.close()

    # Cross-section analysis
    center = N // 2
    
    plt.figure(figsize=(14, 10))
    
    # Amplitude profile
    plt.subplot(3, 3, 1)
    plt.plot(amp[center, :], label='Ground Truth')
    plt.title("Amplitude Profile (center row)")
    plt.xlabel("Pixel")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Phase profile
    plt.subplot(3, 3, 2)
    plt.plot(phase[center, :], label='Ground Truth')
    plt.title("Phase Profile (center row)")
    plt.xlabel("Pixel")
    plt.ylabel("Phase [rad]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Hologram intensity profiles - comparison
    plt.subplot(3, 3, 3)
    for name, (_, holo) in results.items():
        plt.plot(holo[center, :], label=name.capitalize(), alpha=0.7)
    plt.title("Hologram Intensity Profiles")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Propagated amplitude profiles
    plt.subplot(3, 3, 4)
    for name, (uz, _) in results.items():
        plt.plot(np.abs(uz[center, :]), label=name.capitalize(), alpha=0.7)
    plt.title("Propagated Amplitude Profiles")
    plt.xlabel("Pixel")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Propagated phase profiles
    plt.subplot(3, 3, 5)
    for name, (uz, _) in results.items():
        plt.plot(np.angle(uz[center, :]), label=name.capitalize(), alpha=0.7)
    plt.title("Propagated Phase Profiles")
    plt.xlabel("Pixel")
    plt.ylabel("Phase [rad]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Difference maps (if multiple frameworks available)
    if len(results) > 1:
        frameworks = list(results.keys())
        ref_name = frameworks[0]
        _, holo_ref = results[ref_name]
        
        plt.subplot(3, 3, 6)
        if len(frameworks) > 1:
            _, holo_comp = results[frameworks[1]]
            diff = np.abs(holo_ref - holo_comp)
            plt.plot(diff[center, :], label=f'{frameworks[1]} - {ref_name}')
            plt.title("Intensity Difference (center row)")
            plt.xlabel("Pixel")
            plt.ylabel("Absolute Difference")
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    # Frequency spectrum comparison
    plt.subplot(3, 3, 7)
    for name, (_, holo) in results.items():
        holo_fft = np.fft.fftshift(np.fft.fft2(holo))
        holo_fft_1d = np.abs(holo_fft[center, :])
        plt.semilogy(holo_fft_1d, label=name.capitalize(), alpha=0.7)
    plt.title("Frequency Spectrum (center row)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude (log)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Energy conservation
    plt.subplot(3, 3, 8)
    u0 = complex_from_amp_phase(amp, phase)
    energy_object = np.sum(np.abs(u0)**2)
    energy_ratios = []
    framework_names = []
    
    for name, (uz, _) in results.items():
        energy_prop = np.sum(np.abs(uz)**2)
        ratio = energy_prop / energy_object
        energy_ratios.append(ratio)
        framework_names.append(name.capitalize())
    
    plt.bar(framework_names, energy_ratios)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal (1.0)')
    plt.title("Energy Conservation")
    plt.ylabel("Energy Ratio")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Statistical comparison
    plt.subplot(3, 3, 9)
    stats = []
    for name, (_, holo) in results.items():
        stats.append({
            'mean': holo.mean(),
            'std': holo.std(),
            'max': holo.max(),
            'min': holo.min()
        })
    
    x = np.arange(len(framework_names))
    width = 0.2
    plt.bar(x - width*1.5, [s['mean'] for s in stats], width, label='Mean')
    plt.bar(x - width*0.5, [s['std'] for s in stats], width, label='Std')
    plt.bar(x + width*0.5, [s['max'] for s in stats], width, label='Max')
    plt.bar(x + width*1.5, [s['min'] for s in stats], width, label='Min')
    plt.title("Statistical Comparison")
    plt.ylabel("Value")
    plt.xticks(x, framework_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comprehensive_analysis.png"), dpi=200)
    plt.close()

    # Frequency domain visualization (2D for each framework)
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, (_, holo)) in enumerate(results.items()):
        holo_fft = np.fft.fftshift(np.fft.fft2(holo))
        holo_fft_mag = np.log10(np.abs(holo_fft) + 1e-10)
        
        im = axes[idx].imshow(holo_fft_mag, cmap="hot")
        axes[idx].set_title(f"{name.capitalize()}\nFrequency Spectrum")
        plt.colorbar(im, ax=axes[idx], label="log10(magnitude)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "frequency_spectrum_comparison.png"), dpi=200)
    plt.close()

    # Final summary printout
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nFrameworks tested:")
    for name in results.keys():
        print(f"  ✓ {name.capitalize()}")
    
    print("\nSaved NPY arrays:")
    print("  - gt_amp.npy, gt_phase.npy")
    for name in results.keys():
        print(f"  - hologram_{name}.npy")
    
    print("\nSaved visualizations:")
    print("  - gt_amplitude.png, gt_phase.png")
    print("  - hologram_[framework].png (per framework)")
    print("  - field_amplitude_[framework].png (per framework)")
    print("  - field_phase_[framework].png (per framework)")
    print("  - framework_comparison.png")
    print("  - hologram_summary_4panel.png")
    print("  - comprehensive_analysis.png")
    print("  - frequency_spectrum_comparison.png")
    
    print(f"\nOutput directory: {out_dir}")
    
    print("\nEnergy conservation:")
    for name, ratio in zip(framework_names, energy_ratios):
        print(f"  {name}: {ratio:.6f} (should be ~1.0)")
    
    print("="*60)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
