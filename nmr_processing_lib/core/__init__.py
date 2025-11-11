"""
Core module for NMR data processing.

Contains fundamental data structures and I/O operations.
"""

from .data_io import DataInterface, NMRData, DataSource
from .parameters import ProcessingParameters, AcquisitionParameters, ParameterManager
from .transforms import apply_fft, apply_ifft, apply_phase_correction

__all__ = [
    'DataInterface',
    'NMRData',
    'DataSource',
    'ProcessingParameters',
    'AcquisitionParameters',
    'ParameterManager',
    'apply_fft',
    'apply_ifft',
    'apply_phase_correction',
]
