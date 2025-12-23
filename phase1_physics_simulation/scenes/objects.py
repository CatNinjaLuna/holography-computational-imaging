'''
phase1_physics_simulation.scenes.objects

Create a mixed object:
- amplitude: soft circular mask (with optional absorption)
- phase: Gaussian bump (or multiple bumps)
'''

import numpy as np
from ..optics.utils import make_xy_grid

def gaussian_phase_bump(X, Y, phi0=2.0, sigma=60e-6, x0=0.0, y0=0.0):
    """
    Gaussian phase bump:
        phi(x,y) = phi0 * exp(-((x-x0)^2 + (y-y0)^2)/(2*sigma^2))
    phi0 in radians, sigma in meters.
    """
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return phi0 * np.exp(-r2 / (2.0 * sigma**2))

def circular_amplitude_mask(X, Y, radius=220e-6, center=(0.0, 0.0), inside_amp=0.6, outside_amp=1.0):
    """
    Soft amplitude mask: inside a radius has lower amplitude (absorption).
    """
    x0, y0 = center
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    amp = np.full_like(X, outside_amp, dtype=np.float32)
    amp[r <= radius] = inside_amp
    return amp

def make_synthetic_object(n: int, dx: float):
    """
    Returns (amp, phase) on an n x n grid.
    """
    X, Y = make_xy_grid(n, dx)

    # Phase: one main bump + a smaller offset bump (helps reveal artifacts)
    phase = gaussian_phase_bump(X, Y, phi0=2.2, sigma=65e-6, x0=0.0, y0=0.0)
    phase += gaussian_phase_bump(X, Y, phi0=1.2, sigma=40e-6, x0=140e-6, y0=-80e-6)

    # Amplitude: absorption disk + mild vignette (optional)
    amp = circular_amplitude_mask(X, Y, radius=240e-6, inside_amp=0.55, outside_amp=1.0)

    # Optional: mild smooth attenuation to avoid overly sharp edges
    # (keeps things visually nice and more "physical")
    vignette = 0.98 + 0.02 * np.exp(-((X**2 + Y**2) / (2 * (350e-6)**2)))
    amp = (amp * vignette).astype(np.float32)

    return amp, phase
