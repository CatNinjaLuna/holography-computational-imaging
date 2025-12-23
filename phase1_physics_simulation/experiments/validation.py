import numpy as np
import matplotlib.pyplot as plt

from optics.fresnel import fresnel_propagate_fft, intensity
from optics.utils import make_xy_grid, complex_from_amp_phase

def main():
    wavelength = 532e-9 #green laser, 532 nm
    N = 1024
    dx = 1.12e-6 # 1.12 µm pixel pitch; common-ish CMOS scale
    z = 5e-3 # 5 mm propagation distance
    '''
      This gives you:
      Field of view: N*dx ≈ 1.15 mm
      Enough fringes to see interference clearly
      FFT sampling that usually behaves well
    '''
    # Plane wave input
    amp = np.ones((N, N), dtype=np.float32)
    phase = np.zeros((N, N), dtype=np.float32)
    u0 = complex_from_amp_phase(amp, phase)

    uz = fresnel_propagate_fft(u0, wavelength=wavelength, z=z, dx=dx)
    I = intensity(uz)

    # Sanity: intensity should be nearly constant
    print("I mean:", float(I.mean()))
    print("I std:", float(I.std()))
    print("std/mean:", float(I.std() / (I.mean() + 1e-12)))

    # Visualize
    plt.figure()
    plt.imshow(I, cmap="gray")
    plt.title("Plane wave propagated intensity (should be ~uniform)")
    plt.colorbar()
    plt.tight_layout()
    
    # Save figure
    import os
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'figure1_plane_wave_validation.png'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {os.path.join(figures_dir, 'figure1_plane_wave_validation.png')}")
    
    plt.show()

if __name__ == "__main__":
    main()
