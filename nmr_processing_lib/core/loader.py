import os
import struct
import numpy as np
from pathlib import Path
import glob

class ProgressiveLoader:
    """
    Handles loading of NMR data from multiple folders as a unified dataset.
    Supports:
    - Multiple experiment folders
    - Natural sorting of scan files
    - Metadata extraction (0.ini)
    - Lazy loading of scan data
    - Dynamic averaging
    """
    
    def __init__(self, folder_paths):
        """
        Initialize with a list of folder paths or a single folder path.
        """
        if isinstance(folder_paths, (str, Path)):
            self.folder_paths = [Path(folder_paths).resolve()]
        else:
            self.folder_paths = [Path(p).resolve() for p in folder_paths]
            
        self.scan_files = [] # List of tuples: (file_path, original_folder_index)
        self.sampling_rate = 6000 # Default
        self.acq_time = None
        
        self._scan_and_index()
        self._load_metadata()

    def _scan_and_index(self):
        """
        Scans all folders for .dat files, sorts them naturally, and indexes them.
        """
        self.scan_files = []
        
        for folder_idx, folder in enumerate(self.folder_paths):
            if not folder.exists():
                print(f"Warning: Folder not found: {folder}")
                continue
                
            # Find all .dat files
            # strict check for integer filenames to avoid system files or non-scan data
            folder_files = list(folder.glob("*.dat"))
            valid_files = [f for f in folder_files if f.stem.isdigit()]
            
            # Natural sort (1, 2, ... 10, 11)
            valid_files.sort(key=lambda f: int(f.stem))
            
            for f in valid_files:
                self.scan_files.append((f, folder_idx))
                
        print(f"Indexed {len(self.scan_files)} scans from {len(self.folder_paths)} folders.")

    def _load_metadata(self):
        """
        Tries to read metadata from the first available 0.ini
        """
        for folder in self.folder_paths:
            ini_file = folder / '0.ini'
            if ini_file.exists():
                try:
                    with open(ini_file, 'r') as f:
                        found_nmrduino = False
                        for line in f:
                            line = line.strip()
                            if '[NMRduino]' in line:
                                found_nmrduino = True
                            elif found_nmrduino and 'SampleRate' in line:
                                self.sampling_rate = float(line.split('=')[1])
                                return # Found it
                except Exception as e:
                    print(f"Error reading metadata from {ini_file}: {e}")
        
        # If we couldn't find/read it, stick to default
        print("Using default sampling rate:", self.sampling_rate)

    def get_count(self):
        return len(self.scan_files)

    def read_scan(self, index):
        """
        Reads a single scan by its global index.
        Returns: numpy array of data
        """
        if index < 0 or index >= len(self.scan_files):
            raise IndexError("Scan index out of range")
            
        file_path, _ = self.scan_files[index]
        
        with open(file_path, 'rb') as file:
            byte_data = bytearray(file.read())
        
        # NMRduino format processing
        # Reverse byte order
        byte_data.reverse()
        
        # Unpack as 16-bit integers
        # Assuming standard header/footer length from nmrduino_util_fixed
        try:
            int16_data = struct.unpack(f'<{len(byte_data)//2}h', byte_data)
            # Skip first 20 and last 2 values (based on nmrduino_util)
            if len(int16_data) > 22:
                return np.array(int16_data[20:-2], dtype=np.int16)
            else:
                return np.array([], dtype=np.int16)
        except struct.error:
            return np.array([], dtype=np.int16)

    def get_aggregated_data(self, indices=None):
        """
        Returns the averaged data for the specified indices.
        If indices is None, uses all scans.
        """
        if not self.scan_files:
            return None, self.sampling_rate, 0
            
        if indices is None:
            indices = range(len(self.scan_files))
            
        valid_indices = [i for i in indices if 0 <= i < len(self.scan_files)]
        
        if not valid_indices:
            return None, self.sampling_rate, 0
            
        # Determine data length from the first valid scan
        first_data = self.read_scan(valid_indices[0])
        data_len = len(first_data)
        
        accumulated = np.zeros(data_len, dtype=np.float64)
        count = 0
        
        for idx in valid_indices:
            scan_data = self.read_scan(idx)
            # Handle length mismatch (truncate or pad? Truncate to min is safest, or skip)
            # Simplest approach: Truncate to shortest common length if close, or strict check
            if len(scan_data) != data_len:
                if abs(len(scan_data) - data_len) < 100: # Allow small mismatch
                    min_len = min(len(scan_data), data_len)
                    accumulated[:min_len] += scan_data[:min_len]
                else:
                    # Skip grossly mismatched scans
                    print(f"Skipping scan {idx}: length mismatch ({len(scan_data)} vs {data_len})")
                    continue
            else:
                accumulated += scan_data
            count += 1
            
        if count == 0:
            return None, self.sampling_rate, 0
            
        averaged = accumulated / count
        
        # Calculate acquisition time
        acq_time = data_len / self.sampling_rate
        
        return averaged, self.sampling_rate, acq_time
