"""
Processing module for NMR data.

Contains signal processing functions for filtering, preprocessing, and postprocessing.
"""

from .filtering import savgol_filter_nmr, apply_window_function, WindowType
from .preprocessing import truncate_time_domain, apply_apodization, zero_filling
from .postprocessing import gaussian_broadening, baseline_correction

__all__ = [
    'savgol_filter_nmr',
    'apply_window_function',
    'WindowType',
    'truncate_time_domain',
    'apply_apodization',
    'zero_filling',
    'gaussian_broadening',
    'baseline_correction',
]
