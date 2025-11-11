"""
Transform Module
================

Fourier transforms and phase corrections.
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple, Optional, Union

from .data_io import NMRData


def apply_fft(
    data: Union[NMRData, np.ndarray], 
    sampling_rate: Optional[float] = None,
    update_data: bool = True,
    zero_padding: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FFT to time domain data.
    
    Args:
        data: NMRData object or numpy array
        sampling_rate: Sampling rate (required if data is numpy array)
        update_data: Update the NMRData object with results (only if data is NMRData)
        zero_padding: Zero padding factor (0 = no padding)
    
    Returns:
        Tuple of (frequency_axis, frequency_data)
    
    Example:
        >>> # From NMRData
        >>> freq_axis, spectrum = apply_fft(nmr_data)
        >>> 
        >>> # From numpy array
        >>> freq_axis, spectrum = apply_fft(time_array, sampling_rate=5000)
    """
    # Handle input type
    if isinstance(data, NMRData):
        time_data = data.time_data
        sr = data.sampling_rate
        is_nmr_data = True
    else:
        time_data = data
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when data is a numpy array")
        sr = sampling_rate
        is_nmr_data = False
    
    # Apply zero padding if requested
    if zero_padding > 0:
        n_zeros = int(len(time_data) * zero_padding)
        time_data_padded = np.concatenate([time_data, np.zeros(n_zeros, dtype=time_data.dtype)])
    else:
        time_data_padded = time_data
    
    # Apply FFT
    freq_data = fft(time_data_padded)
    
    # Generate frequency axis
    freq_axis = frequency_axis(len(freq_data), sr)
    
    # Update data object if requested and possible
    if is_nmr_data and update_data:
        data.freq_data = freq_data
        data.freq_axis = freq_axis
        data.add_processing_step(f"FFT (zero_padding={zero_padding})")
    
    return freq_axis, freq_data


def apply_ifft(freq_data: np.ndarray) -> np.ndarray:
    """
    Apply inverse FFT to frequency domain data.
    
    Args:
        freq_data: Frequency domain data (complex)
    
    Returns:
        Time domain data
    """
    return ifft(freq_data)


def frequency_axis(num_points: int, sampling_rate: float) -> np.ndarray:
    """
    Generate frequency axis for FFT output.
    
    Args:
        num_points: Number of data points
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Frequency axis array
    """
    return np.linspace(0, sampling_rate, num_points)


def apply_phase_correction(
    data: NMRData,
    linear_phase: float = 0.0,
    constant_phase: float = 0.0,
    update_data: bool = True
) -> np.ndarray:
    """
    Apply phase correction to frequency domain data.
    
    Phase correction is applied as:
    corrected = spectrum * exp(i * linear_phase(f)) * exp(i * constant_phase)
    
    Args:
        data: NMRData object (must have freq_data)
        linear_phase: Linear phase in radians (applied across frequency)
        constant_phase: Constant phase offset in radians
        update_data: Update the NMRData object
    
    Returns:
        Phase-corrected spectrum
    """
    if data.freq_data is None:
        raise ValueError("Frequency data not available. Run FFT first.")
    
    # Generate linear phase ramp
    num_points = len(data.freq_data)
    lin_phase_ramp = np.linspace(np.pi * linear_phase, -np.pi * linear_phase, num_points)
    
    # Apply phase correction
    corrected = data.freq_data * np.exp(1j * lin_phase_ramp) * np.exp(-1j * constant_phase)
    
    if update_data:
        data.freq_data = corrected
        data.add_processing_step(f"Phase: linear={linear_phase:.3f}, const={constant_phase:.3f}")
    
    return corrected


def extract_frequency_range(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    freq_min: float,
    freq_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract data within a frequency range.
    
    Args:
        freq_axis: Frequency axis
        spec_data: Spectrum data
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
    
    Returns:
        Tuple of (freq_axis_subset, spec_data_subset)
    """
    mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
    return freq_axis[mask], spec_data[mask]


def bandpass_filter(
    data: NMRData,
    freq_min: float,
    freq_max: float,
    return_time_domain: bool = True
) -> np.ndarray:
    """
    Apply bandpass filter in frequency domain.
    
    Args:
        data: NMRData object
        freq_min: Lower frequency bound (Hz)
        freq_max: Upper frequency bound (Hz)
        return_time_domain: If True, return filtered time domain data
    
    Returns:
        Filtered data (time or frequency domain)
    """
    if data.freq_data is None:
        # Compute FFT first
        apply_fft(data)
    
    # Find frequency indices
    idx_min = np.where(data.freq_axis >= freq_min)[0][0]
    idx_max = np.where(data.freq_axis >= freq_max)[0][0]
    
    # Apply filter (zero outside range)
    filtered_freq = data.freq_data.copy()
    filtered_freq[:idx_min] = 0
    filtered_freq[idx_max:] = 0
    
    if return_time_domain:
        return apply_ifft(filtered_freq)
    else:
        return filtered_freq


def shift_spectrum(
    freq_axis: np.ndarray,
    spec_data: np.ndarray,
    shift_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift spectrum by a frequency offset.
    
    Args:
        freq_axis: Frequency axis
        spec_data: Spectrum data
        shift_hz: Frequency shift in Hz
    
    Returns:
        Tuple of (shifted_freq_axis, spec_data)
    """
    return freq_axis + shift_hz, spec_data


def combine_spectra(
    freq_axes: list,
    spec_data_list: list,
    weights: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple spectra with optional weighting.
    
    Useful for multi-system combinations (like your Spinach UI).
    
    Args:
        freq_axes: List of frequency axes
        spec_data_list: List of spectrum data arrays
        weights: Optional list of weights (auto-normalized)
    
    Returns:
        Tuple of (common_freq_axis, combined_spectrum)
    """
    if not freq_axes or not spec_data_list:
        raise ValueError("Empty input lists")
    
    if len(freq_axes) != len(spec_data_list):
        raise ValueError("Mismatched list lengths")
    
    # Use first axis as reference (assume all are similar)
    common_axis = freq_axes[0]
    
    # Set weights
    if weights is None:
        weights = [1.0 / len(spec_data_list)] * len(spec_data_list)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Combine spectra
    combined = np.zeros_like(spec_data_list[0], dtype=complex)
    
    for spec, weight in zip(spec_data_list, weights):
        combined += spec * weight
    
    return common_axis, combined
