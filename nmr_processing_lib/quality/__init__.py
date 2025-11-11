"""
Quality control module for NMR data.

Contains SNR calculation and scan selection tools.
"""

from .snr import calculate_snr, find_peak_in_range, estimate_noise
from .scan_selection import (
    ScanSelector,
    calculate_scan_residuals,
    filter_scans_by_threshold,
    auto_threshold_suggestion
)

__all__ = [
    'calculate_snr',
    'find_peak_in_range',
    'estimate_noise',
    'ScanSelector',
    'calculate_scan_residuals',
    'filter_scans_by_threshold',
    'auto_threshold_suggestion',
]
