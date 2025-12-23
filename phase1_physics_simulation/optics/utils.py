import numpy as np

def make_xy_grid(n: int, dx: float):
    """
    Returns x, y grids (meters) centered at zero.
    """
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    return X, Y

def make_fxfy_grid(n: int, dx: float):
    """
    Returns fx, fy grids (cycles/meter), centered (fftshifted ordering).
    """
    fx = np.fft.fftfreq(n, d=dx)  # cycles/m
    fx = np.fft.fftshift(fx)
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return FX, FY

def complex_from_amp_phase(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return amp * np.exp(1j * phase)
