"""
NMR Scan Selection System
=========================
A systematic approach to select high-quality NMR scans based on reference scan comparison.

This module provides:
- Data loading from NMRduino format
- Quality assessment using squared residual analysis
- Interactive threshold-based scan filtering
- Results visualization and export

Author: Refactored for better maintainability
Date: 2025-10-21
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.fft import fft
import os
import struct
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path
from glob import glob
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import shutil
from datetime import datetime


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class ScanData:
    """Container for NMR scan data"""
    time_data: np.ndarray
    sampling_rate: int
    acquisition_time: float
    scan_number: int


@dataclass
class QualityMetrics:
    """Container for scan quality metrics"""
    scan_numbers: np.ndarray
    residual_sums: np.ndarray
    reference_scan: int
    
    @property
    def min_residual(self) -> float:
        return np.min(self.residual_sums)
    
    @property
    def max_residual(self) -> float:
        return np.max(self.residual_sums)
    
    @property
    def mean_residual(self) -> float:
        return np.mean(self.residual_sums)
    
    @property
    def median_residual(self) -> float:
        return np.median(self.residual_sums)
    
    @property
    def std_residual(self) -> float:
        return np.std(self.residual_sums)
    
    def get_statistics_summary(self) -> Dict[str, float]:
        """Get complete statistics summary"""
        return {
            'min': self.min_residual,
            'max': self.max_residual,
            'mean': self.mean_residual,
            'median': self.median_residual,
            'std': self.std_residual
        }


@dataclass
class SelectionResult:
    """Container for scan selection results"""
    threshold: float
    selected_scans: List[int]
    total_scans: int
    quality_metrics: QualityMetrics
    
    @property
    def selection_count(self) -> int:
        return len(self.selected_scans)
    
    @property
    def selection_rate(self) -> float:
        return (self.selection_count / self.total_scans) * 100 if self.total_scans > 0 else 0.0


# ============================================================================
# Data Loading Module
# ============================================================================

class NMRDataLoader:
    """
    Responsible for loading NMR data from NMRduino format files.
    """
    
    DEFAULT_SAMPLING_RATE = 6000
    
    def __init__(self, data_path: str, verbose: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the folder containing experiment files
            verbose: Whether to print progress messages
        """
        self.data_path = self._normalize_path(data_path)
        self.verbose = verbose
        self.sampling_rate = self._read_sampling_rate()
        self.available_scans = self._discover_scans()
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path for cross-platform compatibility"""
        if os.name == 'nt' and "\\" in path:
            path = path.replace("\\", "/")
        else:
            path = str(Path(path).expanduser().resolve()) + "/"
        return path
    
    def _read_sampling_rate(self) -> int:
        """Read sampling rate from configuration file"""
        config_file = os.path.join(self.data_path, '0.ini')
        
        if not os.path.exists(config_file):
            if self.verbose:
                print(f"Warning: Configuration file not found. Using default SR: {self.DEFAULT_SAMPLING_RATE}")
            return self.DEFAULT_SAMPLING_RATE
        
        try:
            with open(config_file, 'r') as f:
                lines = [line.rstrip() for line in f]
            
            found_nmrduino = False
            for line in lines:
                if "[NMRduino]" in line:
                    found_nmrduino = True
                if "SampleRate" in line and found_nmrduino:
                    sampling_rate = int(float(line.split("=")[1]))
                    if self.verbose:
                        print(f"Sampling rate: {sampling_rate} Hz")
                    return sampling_rate
        except Exception as e:
            if self.verbose:
                print(f"Error reading sampling rate: {e}")
        
        return self.DEFAULT_SAMPLING_RATE
    
    def _discover_scans(self) -> List[int]:
        """Discover available scan files"""
        dat_files = sorted(glob(os.path.join(self.data_path, '*.dat')))
        scan_numbers = []
        
        for dat_file in dat_files:
            try:
                filename = os.path.basename(dat_file)
                scan_num = int(filename.replace('.dat', ''))
                scan_numbers.append(scan_num)
            except ValueError:
                continue
        
        if self.verbose and scan_numbers:
            print(f"Found {len(scan_numbers)} scans: {min(scan_numbers)} - {max(scan_numbers)}")
        
        return sorted(scan_numbers)
    
    def _parse_dat_file(self, filepath: str) -> np.ndarray:
        """Parse a single .dat file into numpy array"""
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
        
        # Reverse byte array
        byte_array = bytearray(raw_bytes)
        byte_array.reverse()
        
        # Unpack as 16-bit integers
        num_int16 = len(byte_array) // 2
        int16_data = struct.unpack(f'<{num_int16}h', byte_array)
        
        # Extract actual data (skip header and footer)
        data = np.array(int16_data[20:-2])
        
        return np.flip(data[1:])  # Flip and remove first point
    
    def load_scan(self, scan_number: int) -> Optional[ScanData]:
        """
        Load a single scan.
        
        Args:
            scan_number: The scan number to load
            
        Returns:
            ScanData object or None if loading fails
        """
        if scan_number not in self.available_scans:
            if self.verbose:
                print(f"Warning: Scan {scan_number} not in available scans")
            return None
        
        try:
            filepath = os.path.join(self.data_path, f'{scan_number}.dat')
            time_data = self._parse_dat_file(filepath)
            acq_time = len(time_data) / self.sampling_rate
            
            return ScanData(
                time_data=time_data,
                sampling_rate=self.sampling_rate,
                acquisition_time=acq_time,
                scan_number=scan_number
            )
        except Exception as e:
            if self.verbose:
                print(f"Error loading scan {scan_number}: {e}")
            return None
    
    def load_multiple_scans(self, scan_numbers: List[int]) -> Dict[int, ScanData]:
        """
        Load multiple scans.
        
        Args:
            scan_numbers: List of scan numbers to load
            
        Returns:
            Dictionary mapping scan number to ScanData
        """
        results = {}
        for scan_num in scan_numbers:
            scan_data = self.load_scan(scan_num)
            if scan_data is not None:
                results[scan_num] = scan_data
        
        return results


# ============================================================================
# Quality Analysis Module
# ============================================================================

class ScanQualityAnalyzer:
    """
    Analyzes scan quality by comparing against a reference scan.
    Uses sum of squared residuals as quality metric.
    """
    
    def __init__(self, data_loader: NMRDataLoader, verbose: bool = True):
        """
        Initialize quality analyzer.
        
        Args:
            data_loader: NMRDataLoader instance
            verbose: Whether to print progress messages
        """
        self.data_loader = data_loader
        self.verbose = verbose
    
    def create_average_reference(self, scan_numbers: List[int]) -> np.ndarray:
        """
        Create an average reference from multiple scans.
        
        Args:
            scan_numbers: List of scan numbers to average
            
        Returns:
            Averaged time-domain data
        """
        if len(scan_numbers) == 0:
            raise ValueError("Must provide at least one scan for averaging")
        
        if self.verbose:
            print(f"\nCreating average reference from {len(scan_numbers)} scans...")
        
        summed_data = None
        valid_count = 0
        min_length = None
        
        for scan_num in scan_numbers:
            scan_data = self.data_loader.load_scan(scan_num)
            if scan_data is None:
                if self.verbose:
                    print(f"  Warning: Could not load scan {scan_num}, skipping...")
                continue
            
            # Track minimum length
            if min_length is None:
                min_length = len(scan_data.time_data)
            else:
                min_length = min(min_length, len(scan_data.time_data))
            
            # Sum up the data
            if summed_data is None:
                summed_data = scan_data.time_data.astype(np.float64)
            else:
                summed_data[:min_length] += scan_data.time_data[:min_length].astype(np.float64)
            
            valid_count += 1
        
        if valid_count == 0:
            raise ValueError("No valid scans found for averaging")
        
        # Calculate average and trim to minimum length
        # Keep as float64 to avoid precision loss and overflow in residual calculation
        averaged_data = summed_data[:min_length] / valid_count
        
        if self.verbose:
            print(f"  Successfully averaged {valid_count} scans")
            print(f"  Data points: {len(averaged_data)}")
        
        return averaged_data
    
    def calculate_quality_metrics(
        self,
        reference_scan: Optional[int] = None,
        reference_scans: Optional[List[int]] = None,
        scan_list: Optional[List[int]] = None
    ) -> QualityMetrics:
        """
        Calculate quality metrics for all scans compared to reference.
        
        Metric: Sum of squared residuals Σ(scan - reference)²
        
        Args:
            reference_scan: Single reference scan number (mutually exclusive with reference_scans)
            reference_scans: List of scans to average as reference (mutually exclusive with reference_scan)
            scan_list: List of scans to analyze (default: all available)
            
        Returns:
            QualityMetrics object
        """
        if scan_list is None:
            scan_list = self.data_loader.available_scans
        
        # Determine reference data
        if reference_scan is not None and reference_scans is not None:
            raise ValueError("Cannot specify both reference_scan and reference_scans")
        
        if reference_scan is None and reference_scans is None:
            raise ValueError("Must specify either reference_scan or reference_scans")
        
        if reference_scans is not None:
            # Use average of multiple scans as reference
            ref_time_data = self.create_average_reference(reference_scans)
            ref_identifier = f"average of {len(reference_scans)} scans"
            ref_value = -1  # Use -1 to indicate averaged reference
            if self.verbose:
                print(f"\nAnalyzing quality using averaged reference ({len(reference_scans)} scans)...")
                print(f"Method: Σ(scan - average_reference)² for each scan")
        else:
            # Use single scan as reference
            ref_data = self.data_loader.load_scan(reference_scan)
            if ref_data is None:
                raise ValueError(f"Failed to load reference scan {reference_scan}")
            ref_time_data = ref_data.time_data
            ref_identifier = f"scan {reference_scan}"
            ref_value = reference_scan
            if self.verbose:
                print(f"\nAnalyzing quality using reference scan {reference_scan}...")
                print(f"Method: Σ(scan - reference)² for each scan")
        
        residual_sums = []
        valid_scans = []
        
        for i, scan_num in enumerate(scan_list):
            # Progress indicator
            if self.verbose and (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(scan_list)} scans...")
            
            # Load current scan
            scan_data = self.data_loader.load_scan(scan_num)
            if scan_data is None:
                continue
            
            # Calculate residual
            min_len = min(len(ref_time_data), len(scan_data.time_data))
            # Convert to float64 to prevent overflow
            residual = scan_data.time_data[:min_len].astype(np.float64) - ref_time_data[:min_len].astype(np.float64)
            residual_sum = np.sum(residual ** 2, dtype=np.float64)
            
            # Check for valid values
            if np.isfinite(residual_sum):
                residual_sums.append(residual_sum)
                valid_scans.append(scan_num)
            elif self.verbose:
                print(f"Warning: Invalid residual for scan {scan_num}, skipping...")
        
        if len(valid_scans) == 0:
            raise ValueError("No valid scans found after quality analysis")
        
        metrics = QualityMetrics(
            scan_numbers=np.array(valid_scans),
            residual_sums=np.array(residual_sums, dtype=np.float64),
            reference_scan=ref_value  # Will be -1 for averaged reference
        )
        
        if self.verbose:
            self._print_statistics(metrics)
        
        return metrics
    
    @staticmethod
    def _print_statistics(metrics: QualityMetrics):
        """Print quality metrics statistics"""
        stats = metrics.get_statistics_summary()
        print(f"\nQuality Metrics Statistics:")
        print(f"  {'Min:':<8} {stats['min']:.2e}")
        print(f"  {'Max:':<8} {stats['max']:.2e}")
        print(f"  {'Mean:':<8} {stats['mean']:.2e}")
        print(f"  {'Median:':<8} {stats['median']:.2e}")
        print(f"  {'Std:':<8} {stats['std']:.2e}")


# ============================================================================
# Scan Filtering Module
# ============================================================================

class ScanFilter:
    """
    Filters scans based on quality metrics and threshold.
    """
    
    @staticmethod
    def filter_by_threshold(
        metrics: QualityMetrics,
        threshold: float
    ) -> SelectionResult:
        """
        Filter scans by threshold value.
        
        Args:
            metrics: QualityMetrics object
            threshold: Threshold value for filtering
            
        Returns:
            SelectionResult object
        """
        selected_indices = np.where(metrics.residual_sums <= threshold)[0]
        selected_scans = metrics.scan_numbers[selected_indices].tolist()
        
        return SelectionResult(
            threshold=threshold,
            selected_scans=selected_scans,
            total_scans=len(metrics.scan_numbers),
            quality_metrics=metrics
        )
    
    @staticmethod
    def suggest_threshold(metrics: QualityMetrics, method: str = 'median') -> float:
        """
        Suggest an initial threshold value.
        
        Args:
            metrics: QualityMetrics object
            method: Method for suggestion ('median', 'mean', 'percentile')
            
        Returns:
            Suggested threshold value
        """
        if method == 'median':
            return metrics.median_residual
        elif method == 'mean':
            return metrics.mean_residual
        elif method == 'percentile':
            return np.percentile(metrics.residual_sums, 75)
        else:
            return metrics.median_residual


# ============================================================================
# Visualization Module
# ============================================================================

class QualityVisualizer:
    """
    Visualizes scan quality metrics and selection results.
    """
    
    @staticmethod
    def plot_quality_overview(
        metrics: QualityMetrics,
        result: Optional[SelectionResult] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Create overview plot of quality metrics.
        
        Args:
            metrics: QualityMetrics object
            result: Optional SelectionResult to highlight selected scans
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot all scans
        plt.scatter(
            metrics.scan_numbers,
            metrics.residual_sums,
            c='gray',
            s=20,
            alpha=0.5,
            label='All scans'
        )
        
        # Highlight selected scans if provided
        if result is not None:
            selected_mask = np.isin(metrics.scan_numbers, result.selected_scans)
            plt.scatter(
                metrics.scan_numbers[selected_mask],
                metrics.residual_sums[selected_mask],
                c='green',
                s=40,
                label=f'Selected scans ({result.selection_count})',
                zorder=5
            )
            
            # Draw threshold line
            plt.axhline(
                y=result.threshold,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Threshold = {result.threshold:.2e}'
            )
        
        plt.xlabel('Scan Number', fontsize=12)
        plt.ylabel('Sum of Squared Residuals: Σ(scan - reference)²', fontsize=12)
        ref_label = f"avg of scans" if metrics.reference_scan == -1 else f"scan {metrics.reference_scan}"
        plt.title(
            f'Scan Quality Analysis (Reference: {ref_label})',
            fontsize=14
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


# ============================================================================
# Interactive Selection Module
# ============================================================================

class InteractiveSelector:
    """
    Provides interactive threshold selection with real-time visualization.
    """
    
    def __init__(self, metrics: QualityMetrics, output_path: str, enable_filtering: bool = True):
        """
        Initialize interactive selector.
        
        Args:
            metrics: QualityMetrics object
            output_path: Path to save results
            enable_filtering: If True, allows interactive filtering. If False, always selects all scans.
        """
        self.metrics = metrics
        self.output_path = output_path
        self.filter = ScanFilter()
        self.enable_filtering = enable_filtering
        
        # Initialize with 100% selection (no filtering) - select all scans by default
        # This allows users to see all data first, then adjust filtering as needed
        self.current_result = self.filter.filter_by_threshold(
            metrics,
            metrics.max_residual  # Set threshold to max, selecting all scans
        )
    
    def get_selected_scans(self) -> List[int]:
        """
        Get list of currently selected scan numbers.
        
        Returns:
            List of scan numbers that pass the current threshold
        """
        if not self.enable_filtering:
            # If filtering is disabled, return all scans
            return list(self.metrics.scan_numbers)
        return self.current_result.selected_scans
    
    def get_selection_info(self) -> Dict[str, any]:
        """
        Get current selection information for integration with other tools.
        
        Returns:
            Dictionary containing:
                - selected_scans: List of selected scan numbers
                - total_scans: Total number of scans
                - selection_count: Number of selected scans
                - selection_rate: Percentage of scans selected
                - threshold: Current threshold value
                - filtering_enabled: Whether filtering is enabled
        """
        return {
            'selected_scans': self.get_selected_scans(),
            'total_scans': self.current_result.total_scans,
            'selection_count': len(self.get_selected_scans()),
            'selection_rate': (len(self.get_selected_scans()) / self.current_result.total_scans * 100) 
                              if self.current_result.total_scans > 0 else 0.0,
            'threshold': self.current_result.threshold,
            'filtering_enabled': self.enable_filtering
        }
    
    def start_interactive_session(self):
        """Start interactive threshold selection session"""
        # Create figure with controls
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.30)
        
        # Initial plot elements
        scatter_all = ax.scatter(
            self.metrics.scan_numbers,
            self.metrics.residual_sums,
            c='gray',
            s=20,
            alpha=0.5,
            label='All scans'
        )
        
        scatter_selected = ax.scatter(
            [], [],
            c='green',
            s=40,
            label='Selected scans',
            zorder=5
        )
        
        threshold_line = ax.axhline(
            y=self.current_result.threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold = {self.current_result.threshold:.2e}'
        )
        
        ax.set_xlabel('Scan Number', fontsize=12)
        ax.set_ylabel('Sum of Squared Residuals: Σ(scan - reference)²', fontsize=12)
        ref_label = "averaged reference" if self.metrics.reference_scan == -1 else f"scan {self.metrics.reference_scan}"
        ax.set_title(
            f'Interactive Scan Selection (Reference: {ref_label})',
            fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Info text box
        info_text = ax.text(
            0.02, 0.98, '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Sort residuals to map selection rate to threshold
        sorted_residuals = np.sort(self.metrics.residual_sums)
        
        # Create threshold slider
        ax_slider_threshold = plt.axes([0.15, 0.18, 0.7, 0.03])
        
        # Calculate safe step value to avoid overflow
        residual_range = self.metrics.max_residual - self.metrics.min_residual
        if residual_range > 0 and np.isfinite(residual_range):
            step_value = residual_range / 1000
        else:
            step_value = None  # Let matplotlib handle it automatically
        
        slider_threshold = Slider(
            ax_slider_threshold,
            'Threshold',
            self.metrics.min_residual,
            self.metrics.max_residual,
            valinit=self.current_result.threshold,
            valstep=step_value,
            color='lightblue'
        )
        
        # Create selection rate slider
        ax_slider_rate = plt.axes([0.15, 0.12, 0.7, 0.03])
        slider_rate = Slider(
            ax_slider_rate,
            'Selection Rate (%)',
            0,
            100,
            valinit=self.current_result.selection_rate,
            valstep=0.5,
            color='lightcoral'
        )
        
        # Create save button
        ax_button = plt.axes([0.15, 0.05, 0.25, 0.04])
        button_save = Button(
            ax_button,
            'Save Selected Scans',
            color='lightgreen',
            hovercolor='green'
        )
        
        # Create input rate button
        ax_button_input = plt.axes([0.45, 0.05, 0.25, 0.04])
        button_input_rate = Button(
            ax_button_input,
            'Input Selection Rate',
            color='lightyellow',
            hovercolor='yellow'
        )
        
        # Flag to prevent recursive updates
        updating = [False]
        
        def update_plot_from_threshold(threshold):
            """Update plot when threshold slider changes"""
            if updating[0]:
                return
            updating[0] = True
            
            self.current_result = self.filter.filter_by_threshold(
                self.metrics,
                threshold
            )
            
            # Update threshold line
            threshold_line.set_ydata([threshold, threshold])
            threshold_line.set_label(f'Threshold = {threshold:.2e}')
            
            # Update selected scans scatter
            if self.current_result.selection_count > 0:
                selected_mask = np.isin(
                    self.metrics.scan_numbers,
                    self.current_result.selected_scans
                )
                scatter_selected.set_offsets(
                    np.c_[
                        self.metrics.scan_numbers[selected_mask],
                        self.metrics.residual_sums[selected_mask]
                    ]
                )
                scatter_selected.set_label(
                    f'Selected scans ({self.current_result.selection_count})'
                )
            else:
                scatter_selected.set_offsets(np.empty((0, 2)))
                scatter_selected.set_label('Selected scans (0)')
            
            # Update info text
            info_text.set_text(
                f'Threshold: {threshold:.2e}\n'
                f'Selected: {self.current_result.selection_count}/{self.current_result.total_scans} scans\n'
                f'Selection rate: {self.current_result.selection_rate:.1f}%'
            )
            
            # Update selection rate slider without triggering callback
            current_rate = slider_rate.val
            new_rate = self.current_result.selection_rate
            if abs(current_rate - new_rate) > 0.1:  # Only update if significantly different
                slider_rate.set_val(new_rate)
            
            ax.legend()
            fig.canvas.draw_idle()
            updating[0] = False
        
        def update_plot_from_rate(rate):
            """Update plot when selection rate slider changes"""
            if updating[0]:
                return
            updating[0] = True
            
            # Calculate threshold from desired selection rate
            if rate <= 0:
                threshold = self.metrics.min_residual - 1  # Select none
            elif rate >= 100:
                threshold = self.metrics.max_residual  # Select all
            else:
                # Find threshold that gives approximately the desired rate
                target_count = int(len(sorted_residuals) * rate / 100)
                target_count = max(0, min(target_count, len(sorted_residuals) - 1))
                threshold = sorted_residuals[target_count]
            
            # Update the result directly
            self.current_result = self.filter.filter_by_threshold(
                self.metrics,
                threshold
            )
            
            # Update threshold slider without triggering its callback
            current_threshold = slider_threshold.val
            if abs(current_threshold - threshold) > 1e-10:  # Only update if different
                slider_threshold.set_val(threshold)
            
            # Update threshold line
            threshold_line.set_ydata([threshold, threshold])
            threshold_line.set_label(f'Threshold = {threshold:.2e}')
            
            # Update selected scans scatter
            if self.current_result.selection_count > 0:
                selected_mask = np.isin(
                    self.metrics.scan_numbers,
                    self.current_result.selected_scans
                )
                scatter_selected.set_offsets(
                    np.c_[
                        self.metrics.scan_numbers[selected_mask],
                        self.metrics.residual_sums[selected_mask]
                    ]
                )
                scatter_selected.set_label(
                    f'Selected scans ({self.current_result.selection_count})'
                )
            else:
                scatter_selected.set_offsets(np.empty((0, 2)))
                scatter_selected.set_label('Selected scans (0)')
            
            # Update info text
            info_text.set_text(
                f'Threshold: {threshold:.2e}\n'
                f'Selected: {self.current_result.selection_count}/{self.current_result.total_scans} scans\n'
                f'Selection rate: {self.current_result.selection_rate:.1f}%'
            )
            
            ax.legend()
            fig.canvas.draw_idle()
            updating[0] = False
        
        def input_selection_rate(event):
            """Allow user to input exact selection rate"""
            # Create a simple input dialog
            root = tk.Tk()
            root.withdraw()
            
            rate_str = tk.simpledialog.askstring(
                "Input Selection Rate",
                f"Enter desired selection rate (0-100%):\n\n"
                f"Current: {self.current_result.selection_rate:.2f}%",
                parent=root
            )
            root.destroy()
            
            if rate_str:
                try:
                    rate = float(rate_str)
                    if 0 <= rate <= 100:
                        slider_rate.set_val(rate)
                    else:
                        print(f"Error: Rate must be between 0 and 100. Got: {rate}")
                except ValueError:
                    print(f"Error: Invalid number format: {rate_str}")
        
        def save_results(event):
            """Save results when button is clicked"""
            if self.current_result.selection_count == 0:
                print("\nWarning: No scans selected! Please adjust threshold.")
                return
            
            self._save_selection_results(self.current_result)
        
        # Connect events
        slider_threshold.on_changed(update_plot_from_threshold)
        slider_rate.on_changed(update_plot_from_rate)
        button_save.on_clicked(save_results)
        button_input_rate.on_clicked(input_selection_rate)
        
        # Initial update
        update_plot_from_threshold(self.current_result.threshold)
        
        print("\n" + "=" * 60)
        print("Interactive Selection Window Opened")
        print("=" * 60)
        print("Instructions:")
        print("  1. Adjust 'Threshold' slider to change absolute threshold value")
        print("  2. Adjust 'Selection Rate' slider to select by percentage")
        print("  3. Click 'Input Selection Rate' to type exact percentage")
        print("  4. Observe selected scans (green points)")
        print("  5. Click 'Save Selected Scans' when satisfied")
        print("=" * 60)
        
        plt.show()
        print("\nWindow closed.")
    
    def _save_selection_results(self, result: SelectionResult):
        """Save selection results by copying selected scan files"""
        # Ask user to select save location
        root = tk.Tk()
        root.withdraw()
        save_location = filedialog.askdirectory(title="Select Save Location for Filtered Scans")
        root.destroy()
        
        if not save_location:
            print("\nSave cancelled - No location selected.")
            return
        
        # Ask user for folder name
        print("\n" + "=" * 60)
        print("Enter folder name for filtered scans")
        print("=" * 60)
        print("(Leave empty to use default: filtered_scans_YYYYMMDD_HHMMSS)")
        folder_name = input("Folder name: ").strip()
        
        if not folder_name:
            # Use default name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"filtered_scans_{timestamp}"
        
        # Create main output folder
        main_folder = os.path.join(save_location, folder_name)
        
        # Check if folder exists
        if os.path.exists(main_folder):
            print(f"\nWarning: Folder '{folder_name}' already exists!")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Save cancelled.")
                return
            # Remove existing folder
            shutil.rmtree(main_folder)
        
        data_folder = os.path.join(main_folder, "data")
        
        # Create directories
        os.makedirs(data_folder, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Saving Selected Scans...")
        print("=" * 60)
        
        # Copy and renumber selected scan files
        selected_scans_sorted = sorted(result.selected_scans)
        file_mapping = []
        
        for new_index, original_scan in enumerate(selected_scans_sorted):
            # Copy .dat file
            original_dat = os.path.join(self.output_path, f"{original_scan}.dat")
            new_dat = os.path.join(data_folder, f"{new_index}.dat")
            
            if os.path.exists(original_dat):
                shutil.copy2(original_dat, new_dat)
                file_mapping.append(f"{new_index}.dat <- {original_scan}.dat")
            
            # Copy .ini file
            original_ini = os.path.join(self.output_path, f"{original_scan}.ini")
            new_ini = os.path.join(data_folder, f"{new_index}.ini")
            
            if os.path.exists(original_ini):
                shutil.copy2(original_ini, new_ini)
                file_mapping.append(f"{new_index}.ini <- {original_scan}.ini")
            
            if (new_index + 1) % 10 == 0:
                print(f"  Copied {new_index + 1}/{len(selected_scans_sorted)} scans...")
        
        # Copy all non-data files (excluding numbered .dat and .ini files)
        print("\nCopying additional files...")
        for filename in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, filename)
            
            # Skip if it's a directory
            if os.path.isdir(file_path):
                continue
            
            # Skip numbered .dat and .ini files (data files)
            base_name = os.path.splitext(filename)[0]
            if base_name.isdigit():
                continue
            
            # Copy all other files
            dest_path = os.path.join(data_folder, filename)
            shutil.copy2(file_path, dest_path)
            print(f"  Copied: {filename}")
        
        # Create documentation file
        doc_file = os.path.join(main_folder, "selection_info.txt")
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("NMR Scan Selection Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source folder: {self.output_path}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Selection Parameters\n")
            f.write("-" * 70 + "\n")
            if self.metrics.reference_scan == -1:
                f.write(f"Reference type: Averaged reference\n")
                f.write(f"Note: Reference was created from multiple scans average\n")
            else:
                f.write(f"Reference scan number: {self.metrics.reference_scan}\n")
            f.write(f"Quality metric: Sum of squared residuals Σ(scan - reference)²\n")
            f.write(f"Threshold value: {result.threshold:.6e}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Selection Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total scans analyzed: {result.total_scans}\n")
            f.write(f"Scans selected (good): {result.selection_count}\n")
            f.write(f"Scans rejected (bad): {result.total_scans - result.selection_count}\n")
            f.write(f"Selection rate: {result.selection_rate:.2f}%\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Quality Metrics Statistics\n")
            f.write("-" * 70 + "\n")
            stats = self.metrics.get_statistics_summary()
            f.write(f"Minimum residual: {stats['min']:.6e}\n")
            f.write(f"Maximum residual: {stats['max']:.6e}\n")
            f.write(f"Mean residual: {stats['mean']:.6e}\n")
            f.write(f"Median residual: {stats['median']:.6e}\n")
            f.write(f"Std deviation: {stats['std']:.6e}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Selected Scan Numbers (Original -> New)\n")
            f.write("-" * 70 + "\n")
            for new_index, original_scan in enumerate(selected_scans_sorted):
                f.write(f"{original_scan:4d} -> {new_index:4d}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("File Mapping\n")
            f.write("-" * 70 + "\n")
            for mapping in file_mapping:
                f.write(f"{mapping}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("End of Report\n")
            f.write("=" * 70 + "\n")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Results Saved Successfully!")
        print("=" * 60)
        print(f"Reference scan:     {self.metrics.reference_scan}")
        print(f"Threshold:          {result.threshold:.2e}")
        print(f"Selected scans:     {result.selection_count}/{result.total_scans}")
        print(f"Selection rate:     {result.selection_rate:.1f}%")
        print("-" * 60)
        print(f"Save location:      {main_folder}")
        print(f"Data folder:        {data_folder}")
        print(f"Documentation:      {doc_file}")
        print("=" * 60)


# ============================================================================
# User Interface Module
# ============================================================================

class UserInterface:
    """
    Handles user interaction and workflow coordination.
    """
    
    @staticmethod
    def select_folder() -> Optional[str]:
        """Open folder selection dialog"""
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(title="Select NMR Data Folder")
        root.destroy()
        
        if folder_path:
            print(f"Selected folder: {folder_path}")
            return folder_path
        else:
            print("No folder selected.")
            return None
    
    @staticmethod
    def get_reference_scan(available_scans: List[int]) -> tuple:
        """
        Prompt user to select reference scan(s).
        
        Returns:
            tuple: (reference_scan, reference_scans) where one is None
        """
        print(f"\nAvailable scans: {len(available_scans)} total")
        if len(available_scans) <= 20:
            print(f"Scan numbers: {available_scans}")
        else:
            print(f"Scan range: {min(available_scans)} - {max(available_scans)}")
            print(f"First 10: {available_scans[:10]}")
            print(f"Last 10:  {available_scans[-10:]}")
        
        print("\nReference Selection Options:")
        print("  1. Use a single scan as reference")
        print("  2. Use average of selected scans as reference")
        print("  3. Use average of ALL scans as reference")
        
        while True:
            try:
                choice = input("\nSelect option (1, 2, or 3): ").strip()
                
                if choice == "1":
                    # Single scan reference
                    while True:
                        try:
                            ref_scan = int(input("Enter reference scan number (good quality scan): "))
                            if ref_scan in available_scans:
                                return (ref_scan, None)
                            else:
                                print(f"Error: Scan {ref_scan} not found. Please try again.")
                        except ValueError:
                            print("Error: Please enter a valid integer.")
                
                elif choice == "2":
                    # Multiple scans average
                    print("\nEnter scan numbers to average (comma-separated):")
                    print("Example: 5,10,15,20")
                    scan_input = input("Scan numbers: ").strip()
                    
                    try:
                        ref_scans = [int(s.strip()) for s in scan_input.split(',')]
                        
                        # Validate all scans exist
                        invalid_scans = [s for s in ref_scans if s not in available_scans]
                        if invalid_scans:
                            print(f"Error: These scans not found: {invalid_scans}")
                            continue
                        
                        if len(ref_scans) < 2:
                            print("Error: Please provide at least 2 scans for averaging.")
                            continue
                        
                        print(f"\nWill use average of {len(ref_scans)} scans: {sorted(ref_scans)}")
                        confirm = input("Confirm? (y/n): ").strip().lower()
                        if confirm == 'y':
                            return (None, ref_scans)
                    except ValueError:
                        print("Error: Invalid format. Please use comma-separated integers.")
                
                elif choice == "3":
                    # All scans average
                    print(f"\nWill use average of ALL {len(available_scans)} scans as reference")
                    confirm = input("Confirm? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return (None, available_scans)
                
                else:
                    print("Error: Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                raise


# ============================================================================
# Main Application Class
# ============================================================================

class NMRScanSelector:
    """
    Main application class that coordinates the entire workflow.
    """
    
    def __init__(self):
        """Initialize the application"""
        self.ui = UserInterface()
        self.data_loader = None
        self.analyzer = None
        self.visualizer = QualityVisualizer()
    
    def run(self):
        """Execute the complete workflow"""
        try:
            print("=" * 60)
            print("NMR Scan Selection System")
            print("=" * 60)
            print("Version: 2.0 (Refactored)")
            print("Method: Reference-based quality assessment")
            print("=" * 60)
            
            # Step 1: Select data folder
            folder_path = self.ui.select_folder()
            if not folder_path:
                return
            
            # Step 2: Initialize data loader
            print("\nInitializing data loader...")
            self.data_loader = NMRDataLoader(folder_path, verbose=True)
            
            if len(self.data_loader.available_scans) == 0:
                print("Error: No scan files found in selected folder!")
                return
            
            # Step 3: Select reference scan
            ref_scan, ref_scans = self.ui.get_reference_scan(self.data_loader.available_scans)
            
            # Step 4: Analyze quality
            print("\nAnalyzing scan quality...")
            self.analyzer = ScanQualityAnalyzer(self.data_loader, verbose=True)
            
            if ref_scans is not None:
                # Use averaged reference
                metrics = self.analyzer.calculate_quality_metrics(
                    reference_scans=ref_scans
                )
            else:
                # Use single scan reference
                metrics = self.analyzer.calculate_quality_metrics(
                    reference_scan=ref_scan
                )
            
            # Step 5: Interactive selection
            print("\nStarting interactive selection...")
            selector = InteractiveSelector(metrics, folder_path)
            selector.start_interactive_session()
            
            print("\nWorkflow completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Integration API for Main UI
# ============================================================================

class ScanSelectionAPI:
    """
    Simplified API for integrating scan selection into other applications.
    This provides a high-level interface for the main UI to use.
    """
    
    def __init__(self, data_folder: str, verbose: bool = False):
        """
        Initialize scan selection API.
        
        Args:
            data_folder: Path to NMR data folder
            verbose: Whether to print progress messages
        """
        self.data_folder = data_folder
        self.verbose = verbose
        self.data_loader = NMRDataLoader(data_folder, verbose=verbose)
        self.analyzer = None
        self.metrics = None
        self.selector = None
        self.filtering_enabled = False
    
    def setup_quality_analysis(self, reference_scan: Optional[int] = None, 
                               reference_scans: Optional[List[int]] = None):
        """
        Setup quality analysis with reference scan(s).
        
        Args:
            reference_scan: Single reference scan number (or None)
            reference_scans: List of scans to average as reference (or None)
        """
        if len(self.data_loader.available_scans) == 0:
            raise ValueError("No scan files found in data folder")
        
        # If no reference provided, use first scan as default
        if reference_scan is None and reference_scans is None:
            reference_scan = self.data_loader.available_scans[0]
            if self.verbose:
                print(f"Using first scan ({reference_scan}) as reference")
        
        self.analyzer = ScanQualityAnalyzer(self.data_loader, verbose=self.verbose)
        
        if reference_scans is not None:
            self.metrics = self.analyzer.calculate_quality_metrics(
                reference_scans=reference_scans
            )
        else:
            self.metrics = self.analyzer.calculate_quality_metrics(
                reference_scan=reference_scan
            )
        
        self.selector = InteractiveSelector(
            self.metrics, 
            self.data_folder,
            enable_filtering=True
        )
    
    def enable_filtering(self, enabled: bool = True):
        """
        Enable or disable scan filtering.
        
        Args:
            enabled: True to enable filtering, False to use all scans
        """
        self.filtering_enabled = enabled
        if self.selector:
            self.selector.enable_filtering = enabled
    
    def get_selected_scans(self) -> List[int]:
        """
        Get list of selected scan numbers.
        
        Returns:
            List of scan numbers (all scans if filtering disabled)
        """
        if not self.filtering_enabled or self.selector is None:
            return self.data_loader.available_scans
        return self.selector.get_selected_scans()
    
    def get_all_scans(self) -> List[int]:
        """
        Get list of all available scan numbers.
        
        Returns:
            List of all scan numbers
        """
        return self.data_loader.available_scans
    
    def set_threshold(self, threshold: float):
        """
        Set filtering threshold manually.
        
        Args:
            threshold: Threshold value for filtering
        """
        if self.selector and self.metrics:
            self.selector.current_result = self.selector.filter.filter_by_threshold(
                self.metrics,
                threshold
            )
    
    def set_selection_rate(self, rate: float):
        """
        Set filtering by selection rate (percentage).
        
        Args:
            rate: Selection rate in percentage (0-100)
        """
        if self.selector and self.metrics:
            # Calculate threshold from desired selection rate
            sorted_residuals = np.sort(self.metrics.residual_sums)
            if rate <= 0:
                threshold = self.metrics.min_residual - 1
            elif rate >= 100:
                threshold = self.metrics.max_residual
            else:
                target_count = int(len(sorted_residuals) * rate / 100)
                target_count = max(0, min(target_count, len(sorted_residuals) - 1))
                threshold = sorted_residuals[target_count]
            
            self.set_threshold(threshold)
    
    def get_selection_info(self) -> Dict[str, any]:
        """
        Get current selection information.
        
        Returns:
            Dictionary with selection details
        """
        if self.selector:
            return self.selector.get_selection_info()
        return {
            'selected_scans': self.get_selected_scans(),
            'total_scans': len(self.data_loader.available_scans),
            'selection_count': len(self.get_selected_scans()),
            'selection_rate': 100.0,
            'threshold': None,
            'filtering_enabled': self.filtering_enabled
        }
    
    def open_interactive_selector(self):
        """Open interactive threshold selection window."""
        if self.selector is None:
            raise ValueError("Must call setup_quality_analysis() first")
        self.selector.start_interactive_session()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    app = NMRScanSelector()
    app.run()


if __name__ == "__main__":
    main()
