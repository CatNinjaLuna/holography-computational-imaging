import numpy as np
from .utils import make_fxfy_grid

def fresnel_propagate_fft(
    u0: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
) -> np.ndarray:
    """
    Fresnel propagation using the transfer-function method:

        Uz = ifft2( fft2(U0) * H )
        H(fx,fy) = exp( -i*pi*lambda*z*(fx^2 + fy^2) )

    Notes:
    - Assumes u0 is sampled on an N x N grid with pitch dx (meters).
    - Uses fftshifted frequency grids so that Uz is centered in the output.
    """
    if u0.ndim != 2 or u0.shape[0] != u0.shape[1]:
        raise ValueError("u0 must be a square 2D array (N x N).")

    n = u0.shape[0]
    FX, FY = make_fxfy_grid(n, dx)

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    U0 = np.fft.fftshift(u0)
    F0 = np.fft.fft2(U0)
    F0 = np.fft.fftshift(F0)

    Fz = F0 * H

    Fz = np.fft.ifftshift(Fz)
    Uz = np.fft.ifft2(Fz)
    Uz = np.fft.ifftshift(Uz)
    return Uz

def intensity(u: np.ndarray) -> np.ndarray:
    return (u.real**2 + u.imag**2)
