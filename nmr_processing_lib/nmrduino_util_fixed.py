"""
Fixed NMRduino Utilities
=========================
Simplified version with proper Windows path handling
Only includes functions used by the UI
"""

import numpy as np
import struct
import os
from pathlib import Path
from scipy.stats import linregress


def scan_number_extraction(path):
    """
    Determines the number of scans/averages in an experiment folder.
    
    Args:
        path (string): path to the folder containing scan files
        
    Returns:
        int: number of scans found
    """
    path_to_exp = Path(path).resolve()
    
    # Count .dat files with numeric names (0.dat, 1.dat, ...)
    scan_count = 0
    while True:
        filename = path_to_exp / f"{scan_count}.dat"
        if not filename.exists():
            break
        scan_count += 1
    
    return scan_count


def nmrduino_dat_interp(path, scans, nowarn=False):
    """
    Loads NMRduino data from a folder
    
    Args:
        path (string): path to the folder containing scan files
        scans (int or list[2]): Number of scans or range of scans. 
                                0 for all scans
                                
    Returns:
        tuple: (halp, sampling_rate, acq_time)
               - halp: Time domain data (numpy array)
               - sampling_rate: Sampling rate in Hz
               - acq_time: Acquisition time in seconds
    """
    path_to_exp = Path(path).resolve()
    
    # Get sampling rate from 0.ini
    sampling_rate = 6000  # Default fallback
    ini_file = path_to_exp / '0.ini'
    
    if ini_file.exists():
        with open(ini_file, 'r') as f:
            found_nmrduino = False
            for line in f:
                line = line.strip()
                if '[NMRduino]' in line:
                    found_nmrduino = True
                elif found_nmrduino and 'SampleRate' in line:
                    sampling_rate = int(float(line.split('=')[1]))
                    break
    
    # Count total scans
    total_scans = scan_number_extraction(path)
    
    if total_scans == 0:
        raise ValueError(f"No scan files found in {path}")
    
    def read_scan(scan_index):
        """Reads and processes a single .dat scan."""
        filename = path_to_exp / f"{scan_index}.dat"
        
        if not filename.exists():
            raise FileNotFoundError(f"Scan file {filename} not found")
        
        with open(filename, 'rb') as file:
            byte_data = bytearray(file.read())
        
        # Reverse byte order
        byte_data.reverse()
        
        # Unpack as 16-bit integers
        int16_data = struct.unpack(f'<{len(byte_data)//2}h', byte_data)
        
        # Skip first 20 and last 2 values
        np_data = np.array(int16_data[20:-2], dtype=np.int16)
        
        return np_data
    
    # Determine which scans to read
    if isinstance(scans, int):
        if scans == 0:
            # All scans
            scan_indices = range(total_scans)
        elif scans > 0 and scans <= total_scans:
            # Single scan (1-indexed)
            if not nowarn:
                print(f"Scan number {scans} only")
            scan = read_scan(scans - 1)
            halp = np.flip(scan)
            return halp[1:], sampling_rate, len(halp[1:]) / sampling_rate
        else:
            raise ValueError(f"Invalid scan number: {scans} (max: {total_scans})")
    
    elif isinstance(scans, list) and len(scans) == 2:
        # Range of scans [start, end] (1-indexed, inclusive)
        start, end = scans
        if start <= 0 or end <= 0 or start > end or end > total_scans:
            raise ValueError(f"Invalid scan range: [{start}, {end}]")
        scan_indices = range(start - 1, end)
    
    else:
        raise ValueError("scans must be int or list[2]")
    
    # Process multiple scans (average)
    summed_data = None
    for idx in scan_indices:
        try:
            scan = read_scan(idx)
            if summed_data is None:
                summed_data = scan.astype(np.float64)
            else:
                summed_data += scan
        except Exception as e:
            if not nowarn:
                print(f"Error reading scan {idx}: {e}")
    
    if summed_data is None:
        raise ValueError("No valid scans could be read")
    
    # Average and flip
    averaged = summed_data / len(scan_indices)
    halp = np.flip(averaged)
    
    return halp[1:], sampling_rate, len(halp[1:]) / sampling_rate


def snr_calc(xf, yf, peak_freq_range, noise_freq_range, nowarn=False):
    """
    Calculates Signal-to-Noise Ratio (SNR) from a spectrum
    
    Args:
        xf (numpy array): Frequency axis in Hz
        yf (numpy array): Spectrum values (magnitude)
        peak_freq_range (list[2]): [min_freq, max_freq] for signal region (Hz)
        noise_freq_range (list[2]): [min_freq, max_freq] for noise region (Hz)
        nowarn (bool): If True, suppress print output
        
    Returns:
        float: SNR value
    """
    # Use frequency ranges directly (already in Hz)
    peak_freq_range_hz = peak_freq_range
    noise_freq_range_hz = noise_freq_range
    
    # Extract noise region
    noise_mask = (xf >= noise_freq_range_hz[0]) & (xf <= noise_freq_range_hz[1])
    yf_noise = yf[noise_mask]
    xf_noise = xf[noise_mask]
    
    if len(yf_noise) < 2:
        if not nowarn:
            print("Warning: Not enough points in noise region")
        return 0.0
    
    # Baseline correction using linear regression
    slope, intercept, r_value, p_value, std_err = linregress(xf_noise, yf_noise)
    yf_baseline = slope * xf_noise + intercept
    yf_noise_corrected = yf_noise - yf_baseline
    
    # Extract peak region
    peak_mask = (xf >= peak_freq_range_hz[0]) & (xf <= peak_freq_range_hz[1])
    peak_range = yf[peak_mask]
    
    if len(peak_range) == 0:
        if not nowarn:
            print("Warning: No points found in peak region")
        return 0.0
    
    # Calculate peak height (with noise offset)
    peak_offset = np.mean(yf_noise)
    max_val = np.max(peak_range) - peak_offset
    
    # Calculate noise power (RMS)
    avg_noise_power = np.mean(np.sqrt(np.square(yf_noise_corrected)))
    
    if avg_noise_power == 0:
        if not nowarn:
            print("Warning: Zero noise power")
        return float('inf') if max_val > 0 else 0.0
    
    # Calculate SNR
    signal_to_noise = max_val / avg_noise_power
    
    return signal_to_noise
