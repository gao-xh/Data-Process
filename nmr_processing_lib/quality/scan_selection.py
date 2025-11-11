"""
Scan Selection Module
=====================

Tools for identifying and filtering bad scans based on quality metrics.
Integrated from fid_select.py with enhanced functionality.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from ..core.data_io import load_nmrduino_data, get_available_scans


@dataclass
class ScanQualityMetrics:
    """Container for scan quality metrics"""
    scan_number: int
    residual_sum: float          # Sum of squared residuals
    peak_height: Optional[float] = None
    snr: Optional[float] = None
    is_good: bool = True


class ScanSelector:
    """
    Scan quality analyzer and selector.
    
    Analyzes all scans in an experiment and identifies bad scans
    based on various quality metrics.
    """
    
    def __init__(self, folder_path: str):
        """
        Initialize scan selector.
        
        Args:
            folder_path: Path to experiment folder
        """
        self.folder_path = Path(folder_path)
        self.available_scans = get_available_scans(str(folder_path))
        self.scan_metrics: Dict[int, ScanQualityMetrics] = {}
        self.reference_scan: Optional[int] = None
    
    def calculate_residuals(
        self,
        reference_scan: int,
        method: str = 'squared'
    ) -> Dict[int, float]:
        """
        Calculate residuals between each scan and reference scan.
        
        This is the core method from fid_select.py.
        
        Args:
            reference_scan: Scan number to use as reference (goodscan)
            method: Residual calculation method
                - 'squared': Sum of squared residuals (default)
                - 'absolute': Sum of absolute residuals
                - 'max': Maximum absolute difference
        
        Returns:
            Dictionary mapping scan_number -> residual_value
        """
        self.reference_scan = reference_scan
        
        # Load reference scan
        ref_data, ref_sr, ref_acq, _ = load_nmrduino_data(
            str(self.folder_path),
            reference_scan,
            nowarn=True
        )
        
        print(f"Reference scan {reference_scan}: {len(ref_data)} points")
        
        residuals = {}
        
        for scan_num in self.available_scans:
            # Load scan
            try:
                scan_data, scan_sr, scan_acq, _ = load_nmrduino_data(
                    str(self.folder_path),
                    scan_num,
                    nowarn=True
                )
            except Exception as e:
                print(f"Warning: Could not load scan {scan_num}: {e}")
                residuals[scan_num] = float('inf')
                continue
            
            # Ensure same length
            min_len = min(len(ref_data), len(scan_data))
            ref_trunc = ref_data[:min_len]
            scan_trunc = scan_data[:min_len]
            
            # Calculate residual
            diff = scan_trunc - ref_trunc
            
            if method == 'squared':
                residual = np.sum(diff ** 2)
            elif method == 'absolute':
                residual = np.sum(np.abs(diff))
            elif method == 'max':
                residual = np.max(np.abs(diff))
            else:
                raise ValueError(f"Unknown method: {method}")
            
            residuals[scan_num] = residual
            
            # Store in metrics
            self.scan_metrics[scan_num] = ScanQualityMetrics(
                scan_number=scan_num,
                residual_sum=residual
            )
        
        return residuals
    
    def filter_by_threshold(
        self,
        threshold: float,
        residuals: Optional[Dict[int, float]] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Filter scans based on residual threshold.
        
        Args:
            threshold: Residual threshold (scans above this are rejected)
            residuals: Pre-calculated residuals (uses stored if None)
        
        Returns:
            Tuple of (good_scans, bad_scans)
        """
        if residuals is None:
            residuals = {m.scan_number: m.residual_sum 
                        for m in self.scan_metrics.values()}
        
        good_scans = []
        bad_scans = []
        
        for scan_num, residual in residuals.items():
            if residual <= threshold:
                good_scans.append(scan_num)
                if scan_num in self.scan_metrics:
                    self.scan_metrics[scan_num].is_good = True
            else:
                bad_scans.append(scan_num)
                if scan_num in self.scan_metrics:
                    self.scan_metrics[scan_num].is_good = False
        
        return sorted(good_scans), sorted(bad_scans)
    
    def auto_threshold_suggestion(
        self,
        residuals: Dict[int, float],
        method: str = 'percentile',
        percentile: float = 75.0,
        sigma_factor: float = 3.0
    ) -> float:
        """
        Automatically suggest a threshold for scan filtering.
        
        Args:
            residuals: Calculated residuals
            method: Threshold calculation method
                - 'percentile': Use percentile of residuals
                - 'sigma': Mean + sigma_factor * std
                - 'median': Median + sigma_factor * MAD
            percentile: Percentile to use (for percentile method)
            sigma_factor: Factor for sigma/MAD methods
        
        Returns:
            Suggested threshold value
        """
        residual_values = np.array(list(residuals.values()))
        
        if method == 'percentile':
            threshold = np.percentile(residual_values, percentile)
        
        elif method == 'sigma':
            mean = np.mean(residual_values)
            std = np.std(residual_values)
            threshold = mean + sigma_factor * std
        
        elif method == 'median':
            median = np.median(residual_values)
            mad = np.median(np.abs(residual_values - median))
            threshold = median + sigma_factor * mad
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(threshold)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about scan quality.
        
        Returns:
            Dictionary with statistics
        """
        if not self.scan_metrics:
            return {}
        
        residuals = [m.residual_sum for m in self.scan_metrics.values()]
        good_scans = [m.scan_number for m in self.scan_metrics.values() if m.is_good]
        bad_scans = [m.scan_number for m in self.scan_metrics.values() if not m.is_good]
        
        return {
            'total_scans': len(self.scan_metrics),
            'good_scans': len(good_scans),
            'bad_scans': len(bad_scans),
            'good_scan_numbers': good_scans,
            'bad_scan_numbers': bad_scans,
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals),
            'residual_mean': np.mean(residuals),
            'residual_median': np.median(residuals),
            'residual_std': np.std(residuals),
            'reference_scan': self.reference_scan
        }
    
    def save_selected_scans(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        Save list of good scans to file.
        
        Args:
            output_file: Output file path (default: folder/selected_scans.txt)
        
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = str(self.folder_path / "selected_scans.txt")
        
        good_scans = [m.scan_number for m in self.scan_metrics.values() if m.is_good]
        
        np.savetxt(output_file, good_scans, fmt='%d')
        
        return output_file
    
    def load_selected_scans(
        self,
        input_file: Optional[str] = None
    ) -> List[int]:
        """
        Load previously saved scan selection.
        
        Args:
            input_file: Input file path (default: folder/selected_scans.txt)
        
        Returns:
            List of selected scan numbers
        """
        if input_file is None:
            input_file = str(self.folder_path / "selected_scans.txt")
        
        selected = np.loadtxt(input_file, dtype=int)
        
        # Update metrics
        for scan_num in self.available_scans:
            if scan_num in self.scan_metrics:
                self.scan_metrics[scan_num].is_good = scan_num in selected
        
        return list(selected)


def calculate_scan_residuals(
    folder_path: str,
    reference_scan: int,
    method: str = 'squared'
) -> Tuple[Dict[int, float], List[int]]:
    """
    Calculate residuals for all scans in a folder (standalone function).
    
    Args:
        folder_path: Path to experiment folder
        reference_scan: Reference scan number
        method: Residual calculation method
    
    Returns:
        Tuple of (residuals_dict, scan_numbers)
    """
    selector = ScanSelector(folder_path)
    residuals = selector.calculate_residuals(reference_scan, method)
    return residuals, selector.available_scans


def filter_scans_by_threshold(
    residuals: Dict[int, float],
    threshold: float
) -> Tuple[List[int], List[int]]:
    """
    Filter scans by threshold (standalone function).
    
    Args:
        residuals: Dictionary of scan_number -> residual
        threshold: Threshold value
    
    Returns:
        Tuple of (good_scans, bad_scans)
    """
    good_scans = []
    bad_scans = []
    
    for scan_num, residual in residuals.items():
        if residual <= threshold:
            good_scans.append(scan_num)
        else:
            bad_scans.append(scan_num)
    
    return sorted(good_scans), sorted(bad_scans)


def auto_threshold_suggestion(
    residuals: Dict[int, float],
    method: str = 'percentile',
    percentile: float = 75.0
) -> float:
    """
    Automatically suggest threshold (standalone function).
    
    Args:
        residuals: Dictionary of residuals
        method: 'percentile', 'sigma', or 'median'
        percentile: Percentile for percentile method
    
    Returns:
        Suggested threshold
    """
    residual_values = np.array(list(residuals.values()))
    
    if method == 'percentile':
        return float(np.percentile(residual_values, percentile))
    elif method == 'sigma':
        mean = np.mean(residual_values)
        std = np.std(residual_values)
        return float(mean + 3.0 * std)
    elif method == 'median':
        median = np.median(residual_values)
        mad = np.median(np.abs(residual_values - median))
        return float(median + 3.0 * mad)
    else:
        raise ValueError(f"Unknown method: {method}")
