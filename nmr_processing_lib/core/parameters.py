"""
Parameter Management Module
===========================

Manages processing and acquisition parameters with:
1. Dataclass-based parameter storage (similar to your Spinach UI)
2. JSON serialization for save/load
3. Parameter validation
4. Default presets
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum


class WindowType(Enum):
    """Window function types"""
    NONE = "none"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"


@dataclass
class AcquisitionParameters:
    """
    Acquisition-related parameters.
    
    These are typically read from the experiment or set during acquisition.
    """
    sampling_rate: float = 8333.0      # Hz
    num_points: int = 65515            # Number of acquisition points
    num_scans: int = 1                 # Number of scans to average
    magnet_field: float = 0.0          # Tesla (0 for ZULF)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AcquisitionParameters':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    @property
    def acquisition_time(self) -> float:
        """Calculate acquisition time"""
        return self.num_points / self.sampling_rate
    
    @property
    def nyquist_frequency(self) -> float:
        """Calculate Nyquist frequency"""
        return self.sampling_rate / 2


@dataclass
class ProcessingParameters:
    """
    Processing parameters for NMR data.
    
    This matches the structure from your notebook workflow.
    Compatible with UI spinboxes/sliders.
    """
    # Savgol filtering
    savgol_window: int = 300           # Window length (must be odd)
    savgol_order: int = 2              # Polynomial order
    savgol_enabled: bool = True        # Enable Savgol filtering
    
    # Time domain truncation
    trunc_start: int = 10              # Points to remove from start
    trunc_end: int = 10                # Points to remove from end
    truncation_start: int = 10         # Alias for trunc_start
    truncation_end: int = 10           # Alias for trunc_end
    
    # Apodization
    apodization_t2: float = 0.0        # Exponential decay factor (0 = disabled)
    
    # Window function
    window_type: str = "none"          # Window function type
    window_enabled: bool = False       # Apply window (e.g., Hanning)
    
    # Zero filling
    zero_fill_factor: float = 0.0      # Zero filling factor (0 = no filling)
    
    # Frequency domain processing
    freq_range_min: float = 0.0        # Hz (for display/analysis)
    freq_range_max: float = 300.0      # Hz
    
    # Line broadening (Gaussian)
    gaussian_fwhm: float = 0.0         # Hz (0 = disabled)
    broadening_hz: float = 0.0         # Alias for gaussian_fwhm
    
    # Phase correction
    phase_linear: float = 0.0          # Linear phase (radians)
    phase_constant: float = 0.0        # Constant phase (radians)
    phase0: float = 0.0                # Alias for phase_constant
    phase1: float = 0.0                # Alias for phase_linear
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingParameters':
        """Create from dictionary"""
        # Filter only known fields
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def validate(self) -> List[str]:
        """
        Validate parameters and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Savgol validation
        if self.savgol_window < 2:
            errors.append("Savgol window must be >= 2")
        if self.savgol_window % 2 == 0:
            errors.append("Savgol window must be odd")
        if self.savgol_order >= self.savgol_window:
            errors.append("Savgol order must be < window length")
        if self.savgol_order < 0:
            errors.append("Savgol order must be >= 0")
        
        # Truncation validation
        if self.trunc_start < 0:
            errors.append("Truncation start must be >= 0")
        if self.trunc_end < 0:
            errors.append("Truncation end must be >= 0")
        
        # Apodization validation
        if self.apodization_t2 < 0:
            errors.append("Apodization T2* must be >= 0")
        
        # Zero filling validation
        if self.zero_fill_factor < 0:
            errors.append("Zero fill factor must be >= 0")
        
        # Frequency range validation
        if self.freq_range_min < 0:
            errors.append("Frequency range min must be >= 0")
        if self.freq_range_max <= self.freq_range_min:
            errors.append("Frequency range max must be > min")
        
        # Gaussian broadening validation
        if self.gaussian_fwhm < 0:
            errors.append("Gaussian FWHM must be >= 0")
        
        return errors
    
    def copy(self) -> 'ProcessingParameters':
        """Create a deep copy"""
        return ProcessingParameters(**self.to_dict())


class ParameterManager:
    """
    Manages parameter save/load with presets.
    
    Similar to your SaveLoad class but specialized for processing parameters.
    """
    
    DEFAULT_PRESETS = {
        "high_resolution": ProcessingParameters(
            savgol_window=301,
            savgol_order=2,
            trunc_start=100,
            trunc_end=100,
            apodization_t2=0.5,
            zero_fill_factor=2.0
        ),
        "high_sensitivity": ProcessingParameters(
            savgol_window=1001,
            savgol_order=4,
            trunc_start=50,
            trunc_end=50,
            apodization_t2=1.0,
            zero_fill_factor=4.0
        ),
        "fast_preview": ProcessingParameters(
            savgol_window=101,
            savgol_order=2,
            trunc_start=10,
            trunc_end=10,
            zero_fill_factor=0.0
        )
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize parameter manager.
        
        Args:
            config_dir: Directory to store saved parameters (default: current dir)
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Current parameters
        self.processing = ProcessingParameters()
        self.acquisition = AcquisitionParameters()
    
    def save_processing_params(self, filepath: str, params: Optional[ProcessingParameters] = None):
        """
        Save processing parameters to JSON file.
        
        Args:
            filepath: Output file path
            params: Parameters to save (uses current if None)
        """
        params = params or self.processing
        
        with open(filepath, 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def load_processing_params(self, filepath: str) -> ProcessingParameters:
        """
        Load processing parameters from JSON file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Loaded parameters
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        params = ProcessingParameters.from_dict(data)
        self.processing = params
        return params
    
    def save_all(self, filepath: str):
        """
        Save both processing and acquisition parameters.
        
        Args:
            filepath: Output file path
        """
        data = {
            'processing': self.processing.to_dict(),
            'acquisition': self.acquisition.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_all(self, filepath: str):
        """
        Load both processing and acquisition parameters.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'processing' in data:
            self.processing = ProcessingParameters.from_dict(data['processing'])
        if 'acquisition' in data:
            self.acquisition = AcquisitionParameters.from_dict(data['acquisition'])
    
    def load_preset(self, preset_name: str) -> ProcessingParameters:
        """
        Load a default preset.
        
        Args:
            preset_name: Name of preset ('high_resolution', 'high_sensitivity', 'fast_preview')
        
        Returns:
            Preset parameters
        """
        if preset_name not in self.DEFAULT_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available: {list(self.DEFAULT_PRESETS.keys())}")
        
        params = self.DEFAULT_PRESETS[preset_name].copy()
        self.processing = params
        return params
    
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names"""
        return list(self.DEFAULT_PRESETS.keys())
    
    def validate_current(self) -> List[str]:
        """Validate current processing parameters"""
        return self.processing.validate()


# For backward compatibility with your notebook code
def load_stored_parameters(path: str) -> ProcessingParameters:
    """
    Load parameters from folder (legacy notebook format).
    
    Args:
        path: Experiment folder path
    
    Returns:
        ProcessingParameters object
    """
    params = ProcessingParameters()
    
    save_dir = Path(path) / "savgol_filter_save"
    
    if not save_dir.exists():
        return params
    
    try:
        params.zero_fill_factor = float(np.load(save_dir / "zf_factor_stored.npy"))
        params.window_enabled = bool(np.load(save_dir / "hanning_stored.npy"))
        params.savgol_window = int(np.load(save_dir / "conv_points_stored.npy"))
        params.savgol_order = int(np.load(save_dir / "poly_order_stored.npy"))
        params.trunc_start = int(np.load(save_dir / "trunc_stored.npy"))
        params.trunc_end = int(np.load(save_dir / "trunc_f_stored.npy"))
        params.apodization_t2 = float(np.load(save_dir / "apod_stored.npy"))
    except Exception as e:
        print(f"Warning: Could not load all parameters: {e}")
    
    return params


def save_stored_parameters(path: str, params: ProcessingParameters):
    """
    Save parameters to folder (legacy notebook format).
    
    Args:
        path: Experiment folder path
        params: Parameters to save
    """
    save_dir = Path(path) / "savgol_filter_save"
    save_dir.mkdir(exist_ok=True)
    
    np.save(save_dir / "zf_factor_stored.npy", params.zero_fill_factor)
    np.save(save_dir / "hanning_stored.npy", int(params.window_enabled))
    np.save(save_dir / "conv_points_stored.npy", params.savgol_window)
    np.save(save_dir / "poly_order_stored.npy", params.savgol_order)
    np.save(save_dir / "trunc_stored.npy", params.trunc_start)
    np.save(save_dir / "trunc_f_stored.npy", params.trunc_end)
    np.save(save_dir / "apod_stored.npy", params.apodization_t2)
