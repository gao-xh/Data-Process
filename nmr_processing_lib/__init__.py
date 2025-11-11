"""
NMR Data Processing Library
============================

A comprehensive library for NMR data processing with UI integration support.

Main Components:
- DataInterface: Unified data input/output interface
- ProcessingPipeline: Sequential processing workflow
- ParameterManager: Parameter storage and management
- QualityControl: SNR calculation and scan selection

Author: NMR Processing Team
Date: October 2025
Version: 1.0.0
"""

from .core.data_io import (
    DataInterface, 
    NMRData,
    DataSource,
    load_nmrduino_data,
    save_spectrum
)

from .core.parameters import (
    ProcessingParameters,
    AcquisitionParameters,
    ParameterManager
)

from .core.transforms import (
    apply_fft,
    apply_ifft,
    apply_phase_correction,
    frequency_axis
)

from .processing.filtering import (
    savgol_filter_nmr,
    apply_window_function,
    WindowType
)

from .processing.preprocessing import (
    truncate_time_domain,
    apply_apodization,
    zero_filling
)

from .utils.realtime_monitor import (
    RealtimeDataMonitor,
    MonitorState,
    quick_monitor_start
)

from .processing.postprocessing import (
    gaussian_broadening,
    baseline_correction
)

from .quality.snr import (
    calculate_snr,
    find_peak_in_range
)

from .quality.scan_selection import (
    ScanSelector,
    calculate_scan_residuals,
    filter_scans_by_threshold
)

__version__ = "1.0.0"

__all__ = [
    # Data I/O
    'DataInterface',
    'NMRData',
    'DataSource',
    'load_nmrduino_data',
    'save_spectrum',
    
    # Parameters
    'ProcessingParameters',
    'AcquisitionParameters',
    'ParameterManager',
    
    # Transforms
    'apply_fft',
    'apply_ifft',
    'apply_phase_correction',
    'frequency_axis',
    
    # Filtering
    'savgol_filter_nmr',
    'apply_window_function',
    'WindowType',
    
    # Preprocessing
    'truncate_time_domain',
    'apply_apodization',
    'zero_filling',
    
    # Postprocessing
    'gaussian_broadening',
    'baseline_correction',
    
    # Quality
    'calculate_snr',
    'find_peak_in_range',
    'ScanSelector',
    'calculate_scan_residuals',
    'filter_scans_by_threshold',
    
    # Realtime Monitoring
    'RealtimeDataMonitor',
    'MonitorState',
    'quick_monitor_start',
]
