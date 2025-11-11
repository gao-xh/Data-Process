"""
Data I/O Module
===============

Unified interface for NMR data input/output with support for:
1. File-based input (NMRduino .dat files)
2. Direct data input (from acquisition setup/live data)
3. Cached data loading
4. Spectrum export

This module provides abstraction layer for UI integration.
"""

import os
import struct
import numpy as np
from pathlib import Path
from glob import glob
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
from enum import Enum


class DataSource(Enum):
    """Enumeration of data sources"""
    FILE = "file"              # From .dat files
    LIVE = "live"              # From live acquisition
    MEMORY = "memory"          # From memory (already loaded)
    CACHE = "cache"            # From cached .npy files


@dataclass
class NMRData:
    """
    Container for NMR data with metadata.
    
    This is the main data structure passed between processing functions.
    Compatible with UI data binding.
    """
    # Time domain data
    time_data: np.ndarray                    # Time domain signal
    sampling_rate: float                      # Sampling rate in Hz
    acquisition_time: float                   # Acquisition time in seconds
    
    # Frequency domain data (computed after FFT)
    freq_data: Optional[np.ndarray] = None   # Frequency domain signal (complex)
    freq_axis: Optional[np.ndarray] = None   # Frequency axis
    
    # Metadata
    source: DataSource = DataSource.FILE
    source_path: Optional[str] = None        # File path or identifier
    scan_numbers: Optional[List[int]] = None # Which scans were loaded
    num_scans: int = 0                       # Total number of scans
    
    # Processing state
    processed: bool = False                   # Has been processed?
    processing_steps: List[str] = field(default_factory=list)  # Processing history
    
    def copy(self) -> 'NMRData':
        """Create a deep copy of the data"""
        return NMRData(
            time_data=self.time_data.copy(),
            sampling_rate=self.sampling_rate,
            acquisition_time=self.acquisition_time,
            freq_data=self.freq_data.copy() if self.freq_data is not None else None,
            freq_axis=self.freq_axis.copy() if self.freq_axis is not None else None,
            source=self.source,
            source_path=self.source_path,
            scan_numbers=self.scan_numbers.copy() if self.scan_numbers else None,
            num_scans=self.num_scans,
            processed=self.processed,
            processing_steps=self.processing_steps.copy()
        )
    
    def add_processing_step(self, step: str):
        """Add a processing step to history"""
        self.processing_steps.append(step)
        self.processed = True
    
    @property
    def time_axis(self) -> np.ndarray:
        """Generate time axis"""
        return np.linspace(0, self.acquisition_time, len(self.time_data))
    
    @property
    def nyquist_frequency(self) -> float:
        """Calculate Nyquist frequency"""
        return self.sampling_rate / 2


class DataInterface:
    """
    Unified interface for loading NMR data from various sources.
    
    This class provides a consistent API for the UI to load data
    regardless of the source (files, live acquisition, etc.)
    """
    
    @staticmethod
    def from_nmrduino_folder(
        folder_path: str,
        scans: Union[int, List[int]] = 0,
        use_cache: bool = True
    ) -> NMRData:
        """
        Load data from NMRduino folder.
        
        Args:
            folder_path: Path to experiment folder
            scans: 0 for all, int for single scan, [start, end] for range
            use_cache: Use cached compiled data if available
        
        Returns:
            NMRData object
        """
        # Check for cached data first
        if use_cache and scans == 0:
            cache_path = os.path.join(folder_path, "halp_compiled.npy")
            if os.path.exists(cache_path):
                return DataInterface._load_from_cache(folder_path)
        
        # Load from .dat files
        time_data, sampling_rate, acq_time, scan_nums = load_nmrduino_data(
            folder_path, scans
        )
        
        return NMRData(
            time_data=time_data,
            sampling_rate=sampling_rate,
            acquisition_time=acq_time,
            source=DataSource.FILE,
            source_path=folder_path,
            scan_numbers=scan_nums,
            num_scans=len(scan_nums) if scan_nums else 1
        )
    
    @staticmethod
    def from_live_acquisition(
        time_data: np.ndarray,
        sampling_rate: float,
        acquisition_time: float,
        scan_number: Optional[int] = None
    ) -> NMRData:
        """
        Create NMRData from live acquisition.
        
        This is the interface for real-time data from acquisition setup.
        
        Args:
            time_data: Raw time domain data
            sampling_rate: Sampling rate in Hz
            acquisition_time: Acquisition time in seconds
            scan_number: Optional scan identifier
        
        Returns:
            NMRData object
        """
        return NMRData(
            time_data=time_data,
            sampling_rate=sampling_rate,
            acquisition_time=acquisition_time,
            source=DataSource.LIVE,
            scan_numbers=[scan_number] if scan_number else None,
            num_scans=1
        )
    
    @staticmethod
    def from_arrays(
        time_data: np.ndarray,
        sampling_rate: float,
        acquisition_time: Optional[float] = None
    ) -> NMRData:
        """
        Create NMRData from numpy arrays (memory).
        
        Args:
            time_data: Time domain data
            sampling_rate: Sampling rate in Hz
            acquisition_time: Acquisition time (auto-calculated if None)
        
        Returns:
            NMRData object
        """
        if acquisition_time is None:
            acquisition_time = len(time_data) / sampling_rate
        
        return NMRData(
            time_data=time_data,
            sampling_rate=sampling_rate,
            acquisition_time=acquisition_time,
            source=DataSource.MEMORY
        )
    
    @staticmethod
    def _load_from_cache(folder_path: str) -> NMRData:
        """Load from cached .npy files"""
        time_data = np.load(os.path.join(folder_path, "halp_compiled.npy"))
        sampling_rate = float(np.load(os.path.join(folder_path, "sampling_rate_compiled.npy")))
        acq_time = float(np.load(os.path.join(folder_path, "acq_time_compiled.npy")))
        
        return NMRData(
            time_data=time_data,
            sampling_rate=sampling_rate,
            acquisition_time=acq_time,
            source=DataSource.CACHE,
            source_path=folder_path
        )
    
    @staticmethod
    def save_cache(data: NMRData, folder_path: str):
        """
        Save data to cache for faster loading.
        
        Args:
            data: NMRData to save
            folder_path: Destination folder
        """
        np.save(os.path.join(folder_path, "halp_compiled.npy"), data.time_data)
        np.save(os.path.join(folder_path, "sampling_rate_compiled.npy"), data.sampling_rate)
        np.save(os.path.join(folder_path, "acq_time_compiled.npy"), data.acquisition_time)


def load_nmrduino_data(
    path: str,
    scans: Union[int, List[int]] = 0,
    nowarn: bool = False
) -> Tuple[np.ndarray, float, float, List[int]]:
    """
    Load NMRduino .dat files (optimized version).
    
    This is the low-level loading function extracted and optimized 
    from nmrduino_util.py.
    
    Args:
        path: Path to experiment folder
        scans: 0 for all, int for single scan, [start, end] for range
        nowarn: Suppress warnings
    
    Returns:
        Tuple of (time_data, sampling_rate, acq_time, scan_numbers)
    """
    # Normalize path
    path_to_exp = Path(path).resolve()
    
    # Get sampling rate from 0.ini
    sampling_rate = 6000  # Default
    ini_path = path_to_exp / '0.ini'
    
    if ini_path.exists():
        with open(ini_path, 'r') as f:
            in_nmr_block = False
            for line in f:
                line = line.strip()
                if '[NMRduino]' in line:
                    in_nmr_block = True
                if in_nmr_block and 'SampleRate' in line:
                    sampling_rate = int(float(line.split('=')[1]))
                    break
    
    # Find all .dat files
    dat_files = sorted(path_to_exp.glob('*.dat'))
    num_scans = len(dat_files)
    
    if num_scans == 0:
        raise ValueError(f"No .dat files found in {path}")
    
    # Parse scan specification
    if isinstance(scans, int):
        if scans == 0:
            # All scans
            scan_range = list(range(num_scans))
        elif 0 < scans <= num_scans:
            # Single scan (1-indexed)
            scan_range = [scans - 1]
        else:
            raise ValueError(f"Scan {scans} doesn't exist (max: {num_scans})")
    elif isinstance(scans, list) and len(scans) == 2:
        # Range (1-indexed, inclusive)
        start, end = scans
        if not (1 <= start <= end <= num_scans):
            raise ValueError(f"Invalid scan range: {scans}")
        scan_range = list(range(start - 1, end))
    else:
        raise ValueError("scans must be int or [start, end]")
    
    # Load and average scans
    halp_sum = None
    
    for scan_idx in scan_range:
        filename = path_to_exp / f"{scan_idx}.dat"
        
        with open(filename, 'rb') as f:
            data = f.read()
        
        # Reverse byte order
        data_reversed = bytearray(data)
        data_reversed.reverse()
        
        # Unpack as int16
        num_values = len(data_reversed) // 2
        values = struct.unpack(f'<{num_values}h', data_reversed)
        
        # Skip header (first 20) and footer (last 2)
        signal = np.array(values[20:-2], dtype=float)
        
        # Accumulate
        if halp_sum is None:
            halp_sum = signal
        else:
            try:
                halp_sum += signal
            except:
                if not nowarn:
                    print(f"Warning: Error adding scan {scan_idx}")
    
    # Average and flip
    halp = np.flip(halp_sum / len(scan_range))
    
    # Remove first point (artifact)
    halp = halp[1:]
    
    # Calculate acquisition time
    acq_time = len(halp) / sampling_rate
    
    # Return 1-indexed scan numbers
    scan_numbers = [s + 1 for s in scan_range]
    
    return halp, sampling_rate, acq_time, scan_numbers


def save_spectrum(
    data: NMRData,
    output_path: str,
    save_time_domain: bool = True,
    save_freq_domain: bool = True,
    save_metadata: bool = True
):
    """
    Save spectrum data to files.
    
    Args:
        data: NMRData object
        output_path: Base path for output files
        save_time_domain: Save time domain data as CSV
        save_freq_domain: Save frequency domain data as CSV/NPY
        save_metadata: Save metadata as JSON
    """
    base_path = Path(output_path)
    base_name = base_path.stem
    base_dir = base_path.parent
    
    # Time domain CSV
    if save_time_domain:
        time_axis = data.time_axis
        time_csv = np.vstack((time_axis, data.time_data))
        csv_path = base_dir / f"{base_name}_time.csv"
        np.savetxt(csv_path, time_csv.T, delimiter=",", 
                   header="Time(s),Signal", comments='')
    
    # Frequency domain
    if save_freq_domain and data.freq_data is not None:
        # Save as NPY (preserves complex data)
        np.save(base_dir / f"{base_name}_freq.npy", data.freq_axis)
        np.save(base_dir / f"{base_name}_spec.npy", data.freq_data)
        
        # Also save magnitude as CSV for easy viewing
        freq_csv = np.vstack((data.freq_axis, np.abs(data.freq_data)))
        csv_path = base_dir / f"{base_name}_spectrum.csv"
        np.savetxt(csv_path, freq_csv.T, delimiter=",",
                   header="Frequency(Hz),Magnitude", comments='')
    
    # Metadata
    if save_metadata:
        import json
        metadata = {
            'sampling_rate': data.sampling_rate,
            'acquisition_time': data.acquisition_time,
            'num_points': len(data.time_data),
            'num_scans': data.num_scans,
            'source': data.source.value,
            'processing_steps': data.processing_steps
        }
        
        json_path = base_dir / f"{base_name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def get_available_scans(folder_path: str) -> List[int]:
    """
    Get list of available scan numbers in a folder.
    
    Args:
        folder_path: Path to experiment folder
    
    Returns:
        List of scan numbers (as named in files, e.g., 1.dat -> 1)
    """
    path = Path(folder_path)
    dat_files = sorted(path.glob('*.dat'))
    
    scan_numbers = []
    for f in dat_files:
        try:
            scan_num = int(f.stem)
            scan_numbers.append(scan_num)  # Use filename as-is
        except ValueError:
            continue
    
    return sorted(scan_numbers)


def select_folder_dialog() -> Optional[str]:
    """
    Simple folder selection using tkinter.
    
    Returns:
        Selected folder path or None
    """
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    folder_path = filedialog.askdirectory(title="Select NMR Data Folder")
    root.destroy()
    
    return folder_path if folder_path else None
