"""
Postprocessing Module
====================

Frequency domain postprocessing operations including line broadening and baseline correction.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Optional, Tuple

from ..core.data_io import NMRData


def gaussian_broadening(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    fwhm_hz: float
) -> np.ndarray:
    """
    Apply Gaussian line broadening to spectrum.
    
    This smooths spectral lines by convolution with a Gaussian function.
    Useful for improving SNR at the cost of resolution.
    
    Args:
        freq_axis: Frequency axis in Hz
        spec_data: Complex spectrum data
        fwhm_hz: Full Width at Half Maximum in Hz (0 = no broadening)
    
    Returns:
        Broadened spectrum (complex)
    
    Effect:
        - Increases linewidth by ~fwhm_hz
        - Reduces noise
        - Smooths spectrum
    
    Example:
        >>> broadened = gaussian_broadening(xf, yf, fwhm_hz=2.0)
    """
    if fwhm_hz <= 0:
        return spec_data
    
    # Calculate frequency step
    df = freq_axis[1] - freq_axis[0] if len(freq_axis) > 1 else 1.0
    
    # Convert FWHM to sigma
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma = fwhm_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Number of points for Gaussian kernel (±3 sigma)
    kernel_half_width = int(3.0 * sigma / df)
    
    if kernel_half_width < 1:
        return spec_data
    
    # Create Gaussian kernel
    x = np.arange(-kernel_half_width, kernel_half_width + 1) * df
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    
    # Convolve with spectrum (handle complex data)
    broadened_real = np.convolve(np.real(spec_data), gaussian_kernel, mode='same')
    broadened_imag = np.convolve(np.imag(spec_data), gaussian_kernel, mode='same')
    
    return broadened_real + 1j * broadened_imag


def lorentzian_broadening(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    fwhm_hz: float
) -> np.ndarray:
    """
    Apply Lorentzian line broadening to spectrum.
    
    Lorentzian broadening is more physically realistic for NMR than Gaussian.
    
    Args:
        freq_axis: Frequency axis in Hz
        spec_data: Complex spectrum data
        fwhm_hz: Full Width at Half Maximum in Hz
    
    Returns:
        Broadened spectrum (complex)
    """
    if fwhm_hz <= 0:
        return spec_data
    
    # Calculate frequency step
    df = freq_axis[1] - freq_axis[0] if len(freq_axis) > 1 else 1.0
    
    # Lorentzian HWHM (half width at half maximum)
    gamma = fwhm_hz / 2.0
    
    # Number of points for kernel
    kernel_half_width = int(5.0 * gamma / df)
    
    if kernel_half_width < 1:
        return spec_data
    
    # Create Lorentzian kernel
    x = np.arange(-kernel_half_width, kernel_half_width + 1) * df
    lorentzian_kernel = gamma / (np.pi * (x**2 + gamma**2))
    lorentzian_kernel /= lorentzian_kernel.sum()  # Normalize
    
    # Convolve with spectrum
    broadened_real = np.convolve(np.real(spec_data), lorentzian_kernel, mode='same')
    broadened_imag = np.convolve(np.imag(spec_data), lorentzian_kernel, mode='same')
    
    return broadened_real + 1j * broadened_imag


def apply_broadening(
    data: NMRData,
    fwhm_hz: float,
    broadening_type: str = 'gaussian',
    update_data: bool = True
) -> np.ndarray:
    """
    Apply line broadening to NMRData spectrum.
    
    Args:
        data: NMRData object (must have freq_data)
        fwhm_hz: Line broadening FWHM in Hz
        broadening_type: 'gaussian' or 'lorentzian'
        update_data: Update the NMRData object
    
    Returns:
        Broadened spectrum
    """
    if data.freq_data is None or data.freq_axis is None:
        raise ValueError("Frequency data not available. Run FFT first.")
    
    if broadening_type == 'gaussian':
        broadened = gaussian_broadening(data.freq_axis, data.freq_data, fwhm_hz)
    elif broadening_type == 'lorentzian':
        broadened = lorentzian_broadening(data.freq_axis, data.freq_data, fwhm_hz)
    else:
        raise ValueError(f"Unknown broadening type: {broadening_type}")
    
    if update_data:
        data.freq_data = broadened
        data.add_processing_step(
            f"Broadening: {broadening_type}, FWHM={fwhm_hz:.2f} Hz"
        )
    
    return broadened


def baseline_correction(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    method: str = 'polynomial',
    order: int = 1,
    noise_regions: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct baseline in frequency domain spectrum.
    
    Args:
        freq_axis: Frequency axis
        spec_data: Spectrum data (real or magnitude)
        method: Baseline correction method
            - 'polynomial': Polynomial fit
            - 'median': Median-based rolling baseline
            - 'regions': Fit baseline from specified noise regions
        order: Polynomial order (for polynomial method)
        noise_regions: List of (freq_min, freq_max) tuples for noise regions
    
    Returns:
        Tuple of (corrected_spectrum, baseline)
    
    Example:
        >>> # Polynomial baseline
        >>> corrected, baseline = baseline_correction(xf, yf, method='polynomial', order=2)
        >>> 
        >>> # Using noise regions
        >>> noise_regions = [(0, 50), (350, 400)]
        >>> corrected, baseline = baseline_correction(xf, yf, method='regions', 
        ...                                          noise_regions=noise_regions)
    """
    # Ensure we're working with real/magnitude data
    if np.iscomplexobj(spec_data):
        spec_data = np.abs(spec_data)
    
    if method == 'polynomial':
        # Fit polynomial to entire spectrum
        coeffs = np.polyfit(freq_axis, spec_data, order)
        baseline = np.polyval(coeffs, freq_axis)
    
    elif method == 'median':
        # Rolling median baseline
        from scipy.signal import medfilt
        window_size = max(3, len(spec_data) // 100)
        if window_size % 2 == 0:
            window_size += 1
        baseline = medfilt(spec_data, kernel_size=window_size)
    
    elif method == 'regions':
        if noise_regions is None:
            raise ValueError("noise_regions required for 'regions' method")
        
        # Extract points from noise regions
        noise_freq = []
        noise_spec = []
        
        for freq_min, freq_max in noise_regions:
            mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
            noise_freq.extend(freq_axis[mask])
            noise_spec.extend(spec_data[mask])
        
        # Fit polynomial to noise regions
        coeffs = np.polyfit(noise_freq, noise_spec, order)
        baseline = np.polyval(coeffs, freq_axis)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Subtract baseline
    corrected = spec_data - baseline
    
    return corrected, baseline


def whittaker_smoother(
    data: np.ndarray,
    lam: float = 1e5,
    differences: int = 2
) -> np.ndarray:
    """
    Apply Whittaker smoothing for baseline correction.
    
    This is an advanced baseline correction method that works well for
    spectra with many peaks.
    
    Args:
        data: Input spectrum (real/magnitude)
        lam: Smoothing parameter (larger = smoother baseline)
        differences: Order of differences (typically 1 or 2)
    
    Returns:
        Smoothed baseline
    
    Reference:
        Eilers, P. H. C. (2003). A perfect smoother.
        Analytical Chemistry, 75(14), 3631-3636.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    n = len(data)
    
    # Construct difference matrix
    E = sparse.eye(n, format='csc')
    D = E[1:] - E[:-1]  # First difference
    
    for _ in range(differences - 1):
        D = D[1:] - D[:-1]
    
    # Construct weights (all equal initially)
    w = np.ones(n)
    W = sparse.diags(w, 0, shape=(n, n))
    
    # Solve for baseline
    baseline = spsolve(W + lam * D.T @ D, w * data)
    
    return baseline


def asymmetric_least_squares_baseline(
    data: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    
    This method preferentially fits the baseline below peaks, making it
    ideal for NMR spectra.
    
    Args:
        data: Input spectrum (real/magnitude)
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (0.001 - 0.1, smaller = more asymmetric)
        niter: Number of iterations
    
    Returns:
        Baseline
    
    Reference:
        Eilers, P. H. C., & Boelens, H. F. M. (2005).
        Baseline correction with asymmetric least squares smoothing.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    n = len(data)
    
    # Construct difference matrix
    D = sparse.eye(n, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]  # Second difference
    
    # Initial weights
    w = np.ones(n)
    
    # Iterate
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n))
        Z = W + lam * D.T @ D
        baseline = spsolve(Z, w * data)
        
        # Update weights (asymmetric)
        w = p * (data > baseline) + (1 - p) * (data <= baseline)
    
    return baseline


def normalize_spectrum(
    spec_data: np.ndarray,
    method: str = 'max'
) -> np.ndarray:
    """
    Normalize spectrum.
    
    Args:
        spec_data: Spectrum data
        method: Normalization method
            - 'max': Divide by maximum value
            - 'sum': Divide by sum (total integral)
            - 'rms': Divide by RMS value
    
    Returns:
        Normalized spectrum
    """
    if method == 'max':
        return spec_data / np.max(np.abs(spec_data))
    
    elif method == 'sum':
        return spec_data / np.sum(np.abs(spec_data))
    
    elif method == 'rms':
        rms = np.sqrt(np.mean(np.abs(spec_data) ** 2))
        return spec_data / rms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def derivative_spectrum(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Calculate derivative of spectrum.
    
    Useful for peak detection and resolution enhancement.
    
    Args:
        freq_axis: Frequency axis
        spec_data: Spectrum data
        order: Derivative order (1 or 2)
    
    Returns:
        Derivative spectrum
    """
    if order == 1:
        return np.gradient(spec_data, freq_axis)
    elif order == 2:
        first_deriv = np.gradient(spec_data, freq_axis)
        return np.gradient(first_deriv, freq_axis)
    else:
        raise ValueError("order must be 1 or 2")


def deconvolution(
    spec_data: np.ndarray,
    line_width: float,
    target_width: float
) -> np.ndarray:
    """
    Deconvolution for resolution enhancement.
    
    WARNING: This can amplify noise. Use with caution.
    
    Args:
        spec_data: Input spectrum
        line_width: Current linewidth estimate
        target_width: Target linewidth
    
    Returns:
        Deconvolved spectrum
    """
    # This is a placeholder for deconvolution
    # Full implementation would require careful handling of noise
    # and stability issues
    
    # TODO: Implement proper deconvolution
    # Could use Wiener filtering or iterative methods
    
    return spec_data
