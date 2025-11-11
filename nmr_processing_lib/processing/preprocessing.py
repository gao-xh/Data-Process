"""
Preprocessing Module
===================

Time domain preprocessing operations including truncation, apodization, and zero filling.
"""

import numpy as np
from typing import Optional

from ..core.data_io import NMRData


def truncate_time_domain(
    data: np.ndarray,
    trunc_start: int = 0,
    trunc_end: int = 0
) -> np.ndarray:
    """
    Truncate time domain data from both ends.
    
    Removes artifacts or noise from the beginning and/or end of acquisition.
    
    Args:
        data: Input time domain data
        trunc_start: Number of points to remove from start
        trunc_end: Number of points to remove from end
    
    Returns:
        Truncated data
    
    Example:
        >>> truncated = truncate_time_domain(halp, trunc_start=100, trunc_end=100)
        >>> # Removes first 100 and last 100 points
    """
    if trunc_start < 0 or trunc_end < 0:
        raise ValueError("Truncation values must be >= 0")
    
    if trunc_start + trunc_end >= len(data):
        raise ValueError("Truncation removes all data points")
    
    if trunc_end == 0:
        return data[trunc_start:]
    else:
        return data[trunc_start:-trunc_end]


def apply_truncation(
    data: NMRData,
    trunc_start: int = 0,
    trunc_end: int = 0,
    update_data: bool = True
) -> np.ndarray:
    """
    Apply truncation to NMRData object.
    
    Also updates acquisition time accordingly.
    
    Args:
        data: NMRData object
        trunc_start: Points to remove from start
        trunc_end: Points to remove from end
        update_data: Update the NMRData object
    
    Returns:
        Truncated time domain data
    """
    truncated = truncate_time_domain(data.time_data, trunc_start, trunc_end)
    
    if update_data:
        # Update acquisition time
        new_length = len(truncated)
        old_length = len(data.time_data)
        data.acquisition_time = data.acquisition_time * (new_length / old_length)
        
        data.time_data = truncated
        data.add_processing_step(
            f"Truncation: start={trunc_start}, end={trunc_end}"
        )
    
    return truncated


def apply_apodization(
    data: np.ndarray,
    t2_star: float,
    acquisition_time: Optional[float] = None,
    apodization_type: str = 'exponential'
) -> np.ndarray:
    """
    Apply apodization (exponential decay) to time domain data.
    
    Apodization improves SNR by suppressing noise at the end of the FID,
    at the cost of line broadening.
    
    Args:
        data: Input time domain data
        t2_star: Decay time constant (larger = less decay)
                 Typical values: 0.5 - 2.0 for line broadening
                 Set to 0 to disable
        acquisition_time: Total acquisition time in seconds (optional, auto-calculated if None)
        apodization_type: Type of apodization ('exponential', 'gaussian', 'lorentzian')
    
    Returns:
        Apodized data
    
    Effect on spectrum:
        - Increases SNR (noise reduction)
        - Broadens spectral lines (FWHM increases)
        - Trade-off between sensitivity and resolution
    
    Example:
        >>> apodized = apply_apodization(halp, t2_star=0.75, acquisition_time=0.2)
        >>> # or simple: apodized = apply_apodization(halp, 0.05)  # auto acq_time
    """
    if t2_star <= 0:
        return data  # No apodization
    
    # Auto-calculate acquisition_time if not provided (assume 1 second per 10000 points as default)
    if acquisition_time is None:
        acquisition_time = len(data) / 10000.0  # Rough estimate
    
    # Generate time axis
    t = np.linspace(0, acquisition_time, len(data))
    
    if apodization_type == 'exponential':
        # Exponential decay: exp(-t/T2*)
        window = np.exp(-t / t2_star)
    
    elif apodization_type == 'gaussian':
        # Gaussian: exp(-(t/T2*)^2)
        window = np.exp(-(t / t2_star) ** 2)
    
    elif apodization_type == 'lorentzian':
        # Lorentzian-like: 1 / (1 + (t/T2*)^2)
        window = 1.0 / (1.0 + (t / t2_star) ** 2)
    
    else:
        raise ValueError(f"Unknown apodization type: {apodization_type}")
    
    return data * window


def apply_apodization_to_data(
    data: NMRData,
    t2_star: float,
    apodization_type: str = 'exponential',
    update_data: bool = True
) -> np.ndarray:
    """
    Apply apodization to NMRData object.
    
    Args:
        data: NMRData object
        t2_star: Decay time constant
        apodization_type: Type of apodization
        update_data: Update the NMRData object
    
    Returns:
        Apodized time domain data
    """
    apodized = apply_apodization(
        data.time_data,
        data.acquisition_time,
        t2_star,
        apodization_type
    )
    
    if update_data:
        data.time_data = apodized
        data.add_processing_step(
            f"Apodization: T2*={t2_star:.3f}, type={apodization_type}"
        )
    
    return apodized


def zero_filling(
    data: np.ndarray,
    fill_factor: float = 2.0,
    fill_value: Optional[float] = None
) -> np.ndarray:
    """
    Apply zero filling to increase spectral resolution.
    
    Zero filling pads the FID with zeros (or a constant value) before FFT,
    which interpolates the spectrum without adding new information.
    
    Args:
        data: Input time domain data
        fill_factor: Factor to increase data length (e.g., 2.0 doubles length)
        fill_value: Value to fill with (default: mean of last few points)
    
    Returns:
        Zero-filled data
    
    Example:
        >>> # Double the data length
        >>> filled = zero_filling(halp, fill_factor=2.0)
        >>> 
        >>> # Fill to specific length
        >>> filled = zero_filling(halp, fill_factor=3.5)
    """
    if fill_factor < 0:
        raise ValueError("fill_factor must be >= 0")
    
    if fill_factor == 0:
        return data
    
    # Calculate new length
    new_length = int(len(data) * fill_factor)
    
    if new_length <= len(data):
        return data
    
    # Determine fill value
    if fill_value is None:
        # Use mean of last 10% of points as fill value
        tail_length = max(1, len(data) // 10)
        fill_value = np.mean(data[-tail_length:])
    
    # Create filled array
    filled = np.ones(new_length) * fill_value
    filled[:len(data)] = data
    
    return filled


def apply_zero_filling(
    data: NMRData,
    fill_factor: float = 2.0,
    update_data: bool = True
) -> np.ndarray:
    """
    Apply zero filling to NMRData object.
    
    Also updates acquisition time accordingly.
    
    Args:
        data: NMRData object
        fill_factor: Zero filling factor
        update_data: Update the NMRData object
    
    Returns:
        Zero-filled time domain data
    """
    filled = zero_filling(data.time_data, fill_factor)
    
    if update_data:
        # Update acquisition time proportionally
        data.acquisition_time = data.acquisition_time * (1 + fill_factor)
        data.time_data = filled
        data.add_processing_step(f"Zero filling: factor={fill_factor:.2f}")
    
    return filled


def remove_dc_offset(
    data: np.ndarray,
    offset_points: Optional[int] = None
) -> np.ndarray:
    """
    Remove DC offset from time domain data.
    
    Args:
        data: Input time domain data
        offset_points: Number of initial points to use for offset calculation
                       If None, uses first 10% of data
    
    Returns:
        DC-corrected data
    """
    if offset_points is None:
        offset_points = max(1, len(data) // 10)
    
    # Calculate DC offset from initial points
    dc_offset = np.mean(data[:offset_points])
    
    # Subtract offset
    return data - dc_offset


def apply_first_point_correction(
    data: np.ndarray,
    correction_type: str = 'half'
) -> np.ndarray:
    """
    Apply first point correction to reduce baseline artifacts.
    
    The first point of an NMR FID often contains instrumental artifacts.
    
    Args:
        data: Input time domain data
        correction_type: Type of correction
            - 'half': Multiply first point by 0.5
            - 'zero': Set first point to 0
            - 'interpolate': Interpolate from second point
    
    Returns:
        Corrected data
    """
    corrected = data.copy()
    
    if correction_type == 'half':
        corrected[0] = corrected[0] * 0.5
    
    elif correction_type == 'zero':
        corrected[0] = 0
    
    elif correction_type == 'interpolate':
        if len(data) >= 2:
            corrected[0] = corrected[1]
    
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")
    
    return corrected


def linear_prediction(
    data: np.ndarray,
    num_points: int,
    order: int = 10,
    direction: str = 'forward'
) -> np.ndarray:
    """
    Extend FID using linear prediction.
    
    Linear prediction can extend the FID beyond the acquisition time,
    potentially improving resolution.
    
    Args:
        data: Input time domain data
        num_points: Number of points to predict
        order: Order of linear prediction
        direction: 'forward' or 'backward'
    
    Returns:
        Extended data
    
    Note: This is an advanced technique and requires careful validation.
    """
    # This is a placeholder for linear prediction
    # Full implementation would use algorithms like LPSVD
    # For now, just return original data
    
    # TODO: Implement linear prediction (complex algorithm)
    # Could use scipy.signal.lfilter or custom LP implementation
    
    return data


def resample_data(
    data: np.ndarray,
    original_rate: float,
    new_rate: float
) -> np.ndarray:
    """
    Resample data to a different sampling rate.
    
    Args:
        data: Input data
        original_rate: Original sampling rate (Hz)
        new_rate: Target sampling rate (Hz)
    
    Returns:
        Resampled data
    """
    from scipy import signal
    
    # Calculate number of output points
    num_samples = int(len(data) * new_rate / original_rate)
    
    # Resample using polyphase filtering
    resampled = signal.resample(data, num_samples)
    
    return resampled
