"""
Filtering Module
================

Signal filtering functions including Savgol and window functions.
"""

import numpy as np
from scipy.signal import savgol_filter
from enum import Enum
from typing import Optional

from ..core.data_io import NMRData


class WindowType(Enum):
    """Window function types"""
    NONE = "none"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BARTLETT = "bartlett"
    KAISER = "kaiser"


def savgol_filter_nmr(
    data: np.ndarray,
    window_length: int,
    polyorder: int = 2,
    mode: str = 'mirror'
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to NMR data.
    
    This removes baseline drift and low-frequency noise while preserving
    the signal shape.
    
    Args:
        data: Input time domain data
        window_length: Length of filter window (must be odd, >= polyorder+1)
        polyorder: Order of polynomial fit
        mode: Padding mode ('mirror', 'nearest', 'constant', 'wrap')
    
    Returns:
        Baseline (smooth component) - subtract this from data to get corrected signal
    
    Example:
        >>> smooth_baseline = savgol_filter_nmr(halp, 301, 2)
        >>> corrected_signal = halp - smooth_baseline
    """
    # Validate parameters
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    
    if window_length < polyorder + 1:
        raise ValueError("window_length must be >= polyorder + 1")
    
    if window_length > len(data):
        raise ValueError(f"window_length ({window_length}) must be <= data length ({len(data)})")
    
    # Apply Savgol filter to get smooth baseline
    smooth_baseline = savgol_filter(data, window_length, polyorder, mode=mode)
    
    return smooth_baseline


def apply_savgol_correction(
    data: NMRData,
    window_length: int,
    polyorder: int,
    update_data: bool = True
) -> np.ndarray:
    """
    Apply Savgol filtering and subtract baseline from NMRData.
    
    Args:
        data: NMRData object
        window_length: Savgol window length (must be odd)
        polyorder: Polynomial order
        update_data: Update the NMRData object
    
    Returns:
        Baseline-corrected time domain data
    """
    smooth_baseline = savgol_filter_nmr(
        data.time_data,
        window_length,
        polyorder
    )
    
    corrected = data.time_data - smooth_baseline
    
    if update_data:
        data.time_data = corrected
        data.add_processing_step(
            f"Savgol: window={window_length}, order={polyorder}"
        )
    
    return corrected


def apply_window_function(
    data: np.ndarray,
    window_type: WindowType = WindowType.HANNING,
    alpha: Optional[float] = None
) -> np.ndarray:
    """
    Apply window function to time domain data.
    
    Window functions reduce spectral leakage at the cost of resolution.
    
    Args:
        data: Input time domain data
        window_type: Type of window function
        alpha: Parameter for Kaiser window (only used if window_type is KAISER)
    
    Returns:
        Windowed data
    
    Window characteristics:
        - NONE: No window (boxcar)
        - HANNING: Good general-purpose, smooth rolloff
        - HAMMING: Similar to Hanning, slightly better sidelobe suppression
        - BLACKMAN: Excellent sidelobe suppression, wider mainlobe
        - BARTLETT: Triangular window
        - KAISER: Adjustable (alpha controls tradeoff)
    """
    n = len(data)
    
    if window_type == WindowType.NONE:
        return data
    
    elif window_type == WindowType.HANNING:
        window = np.hanning(n)
    
    elif window_type == WindowType.HAMMING:
        window = np.hamming(n)
    
    elif window_type == WindowType.BLACKMAN:
        window = np.blackman(n)
    
    elif window_type == WindowType.BARTLETT:
        window = np.bartlett(n)
    
    elif window_type == WindowType.KAISER:
        if alpha is None:
            alpha = 5.0  # Default beta value
        window = np.kaiser(n, alpha)
    
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return data * window


def moving_average_filter(
    data: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Apply simple moving average filter.
    
    Args:
        data: Input data
        window_size: Size of averaging window
    
    Returns:
        Smoothed data
    """
    if window_size < 2:
        return data
    
    # Use convolution for moving average
    kernel = np.ones(window_size) / window_size
    
    # Use 'same' mode to preserve length
    smoothed = np.convolve(data, kernel, mode='same')
    
    return smoothed


def median_filter(
    data: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply median filter (good for spike removal).
    
    Args:
        data: Input data
        kernel_size: Size of median window
    
    Returns:
        Filtered data
    """
    from scipy.signal import medfilt
    
    return medfilt(data, kernel_size=kernel_size)


def lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter.
    
    Args:
        data: Input time domain data
        cutoff_freq: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order (higher = sharper cutoff)
    
    Returns:
        Filtered data
    """
    from scipy.signal import butter, filtfilt
    
    # Normalize cutoff frequency
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design filter
    b, a = butter(order, normalized_cutoff, btype='low')
    
    # Apply filter (filtfilt for zero-phase)
    filtered = filtfilt(b, a, data)
    
    return filtered


def highpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth highpass filter.
    
    Args:
        data: Input time domain data
        cutoff_freq: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
    
    Returns:
        Filtered data
    """
    from scipy.signal import butter, filtfilt
    
    # Normalize cutoff frequency
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design filter
    b, a = butter(order, normalized_cutoff, btype='high')
    
    # Apply filter
    filtered = filtfilt(b, a, data)
    
    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    sampling_rate: float,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency.
    
    Useful for removing powerline interference (50/60 Hz).
    
    Args:
        data: Input time domain data
        notch_freq: Frequency to remove (Hz)
        sampling_rate: Sampling rate (Hz)
        quality_factor: Q factor (higher = narrower notch)
    
    Returns:
        Filtered data
    """
    from scipy.signal import iirnotch, filtfilt
    
    # Normalize frequency
    nyquist = sampling_rate / 2
    normalized_freq = notch_freq / nyquist
    
    # Design notch filter
    b, a = iirnotch(normalized_freq, quality_factor)
    
    # Apply filter
    filtered = filtfilt(b, a, data)
    
    return filtered
