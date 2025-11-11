"""
SNR Calculation Module
======================

Signal-to-noise ratio calculation for NMR spectra.
"""

import numpy as np
from scipy.stats import linregress
from typing import Tuple, Optional, Union


def find_peak_in_range(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    freq_min: float,
    freq_max: float,
    return_index: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, int]]:
    """
    Find maximum peak in specified frequency range.
    
    Args:
        freq_axis: Frequency axis in Hz
        spec_data: Spectrum data (real or magnitude)
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        return_index: Also return the index of the peak
    
    Returns:
        Tuple of (peak_frequency, peak_height) or (peak_frequency, peak_height, index)
    """
    # Ensure magnitude spectrum
    if np.iscomplexobj(spec_data):
        spec_data = np.abs(spec_data)
    
    # Find indices in range
    mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
    
    if not np.any(mask):
        raise ValueError(f"No data in range [{freq_min}, {freq_max}] Hz")
    
    # Extract range
    freq_range = freq_axis[mask]
    spec_range = spec_data[mask]
    
    # Find maximum
    max_idx_local = np.argmax(spec_range)
    peak_freq = freq_range[max_idx_local]
    peak_height = spec_range[max_idx_local]
    
    if return_index:
        # Find global index
        global_idx = np.where(mask)[0][max_idx_local]
        return peak_freq, peak_height, global_idx
    else:
        return peak_freq, peak_height


def estimate_noise(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    noise_freq_min: float,
    noise_freq_max: float,
    baseline_correction: bool = True
) -> Tuple[float, float]:
    """
    Estimate noise level from a noise region.
    
    Args:
        freq_axis: Frequency axis in Hz
        spec_data: Spectrum data (real or magnitude)
        noise_freq_min: Start of noise region (Hz)
        noise_freq_max: End of noise region (Hz)
        baseline_correction: Apply linear baseline correction to noise region
    
    Returns:
        Tuple of (noise_rms, noise_mean)
    """
    # Ensure magnitude spectrum
    if np.iscomplexobj(spec_data):
        spec_data = np.abs(spec_data)
    
    # Extract noise region
    mask = (freq_axis >= noise_freq_min) & (freq_axis <= noise_freq_max)
    
    if not np.any(mask):
        raise ValueError(f"No data in noise range [{noise_freq_min}, {noise_freq_max}] Hz")
    
    noise_freq = freq_axis[mask]
    noise_spec = spec_data[mask]
    
    # Baseline correction (remove linear drift in noise region)
    if baseline_correction:
        slope, intercept, _, _, _ = linregress(noise_freq, noise_spec)
        baseline = slope * noise_freq + intercept
        noise_corrected = noise_spec - baseline
    else:
        noise_corrected = noise_spec
    
    # Calculate noise statistics
    noise_mean = np.mean(noise_corrected)
    noise_rms = np.sqrt(np.mean(noise_corrected ** 2))
    
    return noise_rms, noise_mean


def calculate_snr(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    peak_freq_range: Tuple[float, float],
    noise_freq_range: Tuple[float, float],
    baseline_correction: bool = True,
    return_details: bool = False
) -> Union[float, dict]:
    """
    Calculate signal-to-noise ratio.
    
    SNR = peak_height / noise_rms
    
    Args:
        freq_axis: Frequency axis in Hz
        spec_data: Spectrum data (real or magnitude)
        peak_freq_range: (min, max) frequency range for peak search
        noise_freq_range: (min, max) frequency range for noise estimation
        baseline_correction: Apply baseline correction to noise region
        return_details: Return detailed information
    
    Returns:
        SNR value (float), or dict with detailed information if return_details=True
    
    Example:
        >>> snr = calculate_snr(xf, yf, peak_freq_range=(140, 160),
        ...                     noise_freq_range=(350, 400))
        >>> print(f"SNR: {snr:.1f}")
    """
    # Ensure magnitude spectrum
    if np.iscomplexobj(spec_data):
        spec_data = np.abs(spec_data)
    
    # Find peak
    peak_freq, peak_height = find_peak_in_range(
        freq_axis, spec_data,
        peak_freq_range[0], peak_freq_range[1]
    )
    
    # Estimate noise
    noise_rms, noise_mean = estimate_noise(
        freq_axis, spec_data,
        noise_freq_range[0], noise_freq_range[1],
        baseline_correction=baseline_correction
    )
    
    # Calculate baseline offset at peak region (for peak correction)
    peak_mask = (freq_axis >= peak_freq_range[0]) & (freq_axis <= peak_freq_range[1])
    noise_mask = (freq_axis >= noise_freq_range[0]) & (freq_axis <= noise_freq_range[1])
    
    # Use noise mean as baseline offset
    peak_corrected = peak_height - noise_mean
    
    # Calculate SNR
    if noise_rms > 0:
        snr = peak_corrected / noise_rms
    else:
        snr = float('inf')
    
    if return_details:
        return {
            'snr': snr,
            'peak_frequency': peak_freq,
            'peak_height': peak_height,
            'peak_corrected': peak_corrected,
            'noise_rms': noise_rms,
            'noise_mean': noise_mean,
            'peak_range': peak_freq_range,
            'noise_range': noise_freq_range
        }
    else:
        return snr


def calculate_snr_per_scan(
    snr_total: float,
    num_scans: int
) -> float:
    """
    Calculate SNR for a single scan from averaged SNR.
    
    SNR improves with square root of number of scans:
    SNR_total = SNR_single * sqrt(N)
    
    Args:
        snr_total: SNR of averaged spectrum
        num_scans: Number of scans averaged
    
    Returns:
        Estimated SNR for single scan
    
    Example:
        >>> snr_avg = 100.0  # SNR of average of 100 scans
        >>> snr_single = calculate_snr_per_scan(snr_avg, 100)
        >>> print(f"Single scan SNR: {snr_single:.1f}")  # Should be ~10
    """
    if num_scans < 1:
        raise ValueError("num_scans must be >= 1")
    
    return snr_total / np.sqrt(num_scans)


def estimate_required_scans(
    current_snr: float,
    target_snr: float,
    current_scans: int = 1
) -> int:
    """
    Estimate number of scans required to achieve target SNR.
    
    Args:
        current_snr: Current measured SNR
        target_snr: Desired target SNR
        current_scans: Number of scans in current measurement
    
    Returns:
        Required number of total scans
    
    Example:
        >>> # I have SNR=50 with 10 scans, want SNR=100
        >>> required = estimate_required_scans(50, 100, current_scans=10)
        >>> print(f"Need {required} total scans")  # Should be 40
    """
    if current_snr <= 0 or target_snr <= 0:
        raise ValueError("SNR values must be positive")
    
    # SNR ∝ sqrt(N)
    # target_snr / current_snr = sqrt(N_target / N_current)
    # N_target = N_current * (target_snr / current_snr)^2
    
    ratio_squared = (target_snr / current_snr) ** 2
    required_scans = int(np.ceil(current_scans * ratio_squared))
    
    return required_scans


def dynamic_snr_monitor(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    peak_freq_range: Tuple[float, float],
    noise_freq_range: Tuple[float, float],
    target_snr: float,
    current_scans: int = 1
) -> dict:
    """
    Monitor SNR and estimate progress toward target.
    
    Useful for real-time acquisition monitoring.
    
    Args:
        freq_axis: Frequency axis
        spec_data: Current spectrum
        peak_freq_range: Peak frequency range
        noise_freq_range: Noise frequency range
        target_snr: Target SNR to achieve
        current_scans: Number of scans so far
    
    Returns:
        Dictionary with SNR info and acquisition recommendations
    """
    # Calculate current SNR
    snr_details = calculate_snr(
        freq_axis, spec_data,
        peak_freq_range, noise_freq_range,
        return_details=True
    )
    
    current_snr = snr_details['snr']
    
    # Estimate single-scan SNR
    snr_per_scan = calculate_snr_per_scan(current_snr, current_scans)
    
    # Estimate scans needed
    if current_snr < target_snr:
        scans_needed = estimate_required_scans(current_snr, target_snr, current_scans)
        scans_remaining = scans_needed - current_scans
        progress = (current_scans / scans_needed) * 100
    else:
        scans_needed = current_scans
        scans_remaining = 0
        progress = 100.0
    
    return {
        'current_snr': current_snr,
        'target_snr': target_snr,
        'snr_per_scan': snr_per_scan,
        'current_scans': current_scans,
        'scans_needed': scans_needed,
        'scans_remaining': scans_remaining,
        'progress_percent': progress,
        'target_achieved': current_snr >= target_snr,
        'peak_info': {
            'frequency': snr_details['peak_frequency'],
            'height': snr_details['peak_height']
        },
        'noise_info': {
            'rms': snr_details['noise_rms'],
            'mean': snr_details['noise_mean']
        }
    }


def compare_snr(
    freq_axis_1: np.ndarray,
    spec_data_1: np.ndarray,
    freq_axis_2: np.ndarray,
    spec_data_2: np.ndarray,
    peak_freq_range: Tuple[float, float],
    noise_freq_range: Tuple[float, float],
    labels: Tuple[str, str] = ("Spectrum 1", "Spectrum 2")
) -> dict:
    """
    Compare SNR between two spectra.
    
    Useful for comparing simulated vs experimental data, or different processing methods.
    
    Args:
        freq_axis_1: Frequency axis for spectrum 1
        spec_data_1: Spectrum 1 data
        freq_axis_2: Frequency axis for spectrum 2
        spec_data_2: Spectrum 2 data
        peak_freq_range: Peak frequency range (common for both)
        noise_freq_range: Noise frequency range (common for both)
        labels: Labels for the two spectra
    
    Returns:
        Dictionary with comparison results
    """
    snr_1 = calculate_snr(
        freq_axis_1, spec_data_1,
        peak_freq_range, noise_freq_range,
        return_details=True
    )
    
    snr_2 = calculate_snr(
        freq_axis_2, spec_data_2,
        peak_freq_range, noise_freq_range,
        return_details=True
    )
    
    return {
        labels[0]: snr_1,
        labels[1]: snr_2,
        'snr_ratio': snr_1['snr'] / snr_2['snr'] if snr_2['snr'] > 0 else float('inf'),
        'snr_difference': snr_1['snr'] - snr_2['snr'],
        'better_spectrum': labels[0] if snr_1['snr'] > snr_2['snr'] else labels[1]
    }
