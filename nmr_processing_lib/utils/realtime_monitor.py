"""
Real-time Data Monitoring Module
=================================

Monitors experiment folder for new scan files and automatically loads data.
Supports single scan viewing and cumulative averaging.

This module is designed for integration with live acquisition systems.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from threading import Thread, Event

from ..core.data_io import DataInterface, NMRData, get_available_scans


@dataclass
class MonitorState:
    """State of the realtime monitor"""
    is_running: bool = False
    folder_path: Optional[str] = None
    last_scan_number: int = 0
    total_scans_loaded: int = 0
    average_mode: bool = True  # True=average, False=single
    poll_interval: float = 1.0  # seconds
    scan_numbers: List[int] = field(default_factory=list)


class RealtimeDataMonitor:
    """
    Monitor experiment folder for new scans and load them automatically.
    
    Features:
    - Detect new .dat files in folder
    - Single scan view mode
    - Cumulative average mode
    - Callback system for UI updates
    - Thread-safe operation
    
    Example:
        >>> monitor = RealtimeDataMonitor("/path/to/experiment")
        >>> monitor.on_new_scan = lambda data: plot_spectrum(data)
        >>> monitor.start(average_mode=True)
        >>> # ... acquisition running ...
        >>> monitor.stop()
    """
    
    def __init__(self, folder_path: str, poll_interval: float = 1.0):
        """
        Initialize realtime monitor.
        
        Args:
            folder_path: Path to experiment folder to monitor
            poll_interval: How often to check for new files (seconds)
        """
        self.state = MonitorState(
            folder_path=folder_path,
            poll_interval=poll_interval
        )
        
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        
        # Accumulated data for averaging
        self._accumulated_data: Optional[np.ndarray] = None
        self._num_accumulated: int = 0
        
        # Callbacks (UI can set these)
        self.on_new_scan: Optional[Callable[[NMRData, int], None]] = None
        self.on_average_updated: Optional[Callable[[NMRData, int], None]] = None
        self.on_scan_count_changed: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def start(self, average_mode: bool = True):
        """
        Start monitoring folder for new scans.
        
        Args:
            average_mode: True for cumulative average, False for single scans
        """
        if self.state.is_running:
            print("Monitor already running")
            return
        
        self.state.average_mode = average_mode
        self.state.is_running = True
        self._stop_event.clear()
        
        # Get initial scan count
        initial_scans = get_available_scans(self.state.folder_path)
        if initial_scans:
            self.state.last_scan_number = max(initial_scans)
            self.state.scan_numbers = initial_scans
        else:
            self.state.last_scan_number = 0
            self.state.scan_numbers = []
        
        # Reset accumulated data
        self._accumulated_data = None
        self._num_accumulated = 0
        
        # Start monitoring thread
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"Monitor started: {'Average' if average_mode else 'Single'} mode")
        print(f"Initial scans: {len(initial_scans)}")
    
    def stop(self):
        """Stop monitoring."""
        if not self.state.is_running:
            return
        
        self.state.is_running = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        print("Monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                # Check for new scans
                current_scans = get_available_scans(self.state.folder_path)
                
                # Find new scans
                new_scans = [s for s in current_scans if s > self.state.last_scan_number]
                
                if new_scans:
                    # Process each new scan
                    for scan_num in sorted(new_scans):
                        self._process_new_scan(scan_num)
                    
                    # Update state
                    self.state.last_scan_number = max(current_scans)
                    self.state.scan_numbers = current_scans
                    self.state.total_scans_loaded = len(current_scans)
                    
                    # Notify scan count changed
                    if self.on_scan_count_changed:
                        self.on_scan_count_changed(len(current_scans))
                
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Monitor error: {e}")
                else:
                    print(f"Monitor error: {e}")
            
            # Wait for next poll
            self._stop_event.wait(self.state.poll_interval)
    
    def _process_new_scan(self, scan_num: int):
        """
        Process a newly detected scan.
        
        Args:
            scan_num: Scan number to process
        """
        try:
            # Load scan data
            data = DataInterface.from_nmrduino_folder(
                self.state.folder_path,
                scans=scan_num
            )
            
            if self.state.average_mode:
                # Cumulative averaging mode
                self._update_average(data)
                
                # Create averaged data object
                averaged_data = DataInterface.from_arrays(
                    self._accumulated_data / self._num_accumulated,
                    data.sampling_rate,
                    data.acquisition_time
                )
                averaged_data.num_scans = self._num_accumulated
                
                # Callback with averaged data
                if self.on_average_updated:
                    self.on_average_updated(averaged_data, self._num_accumulated)
            
            else:
                # Single scan mode
                if self.on_new_scan:
                    self.on_new_scan(data, scan_num)
        
        except Exception as e:
            if self.on_error:
                self.on_error(f"Error processing scan {scan_num}: {e}")
    
    def _update_average(self, new_data: NMRData):
        """
        Update cumulative average with new scan.
        
        Args:
            new_data: Newly loaded scan data
        """
        if self._accumulated_data is None:
            # First scan
            self._accumulated_data = new_data.time_data.copy()
            self._num_accumulated = 1
        else:
            # Add to accumulation
            # Handle potential length mismatch
            min_len = min(len(self._accumulated_data), len(new_data.time_data))
            self._accumulated_data[:min_len] += new_data.time_data[:min_len]
            self._num_accumulated += 1
    
    def get_current_average(self) -> Optional[NMRData]:
        """
        Get current averaged data.
        
        Returns:
            Current averaged NMRData or None if no data
        """
        if self._accumulated_data is None:
            return None
        
        # Get sampling rate from a scan
        try:
            sample_data = DataInterface.from_nmrduino_folder(
                self.state.folder_path,
                scans=self.state.scan_numbers[0] if self.state.scan_numbers else 1
            )
            
            averaged_data = DataInterface.from_arrays(
                self._accumulated_data / self._num_accumulated,
                sample_data.sampling_rate,
                sample_data.acquisition_time
            )
            averaged_data.num_scans = self._num_accumulated
            
            return averaged_data
        
        except:
            return None
    
    def reset_average(self):
        """Reset accumulated average (start fresh)."""
        self._accumulated_data = None
        self._num_accumulated = 0
        print("Average reset")
    
    def set_mode(self, average_mode: bool):
        """
        Switch between single and average mode.
        
        Args:
            average_mode: True for average, False for single
        """
        was_running = self.state.is_running
        
        if was_running:
            self.stop()
        
        self.state.average_mode = average_mode
        
        if was_running:
            self.start(average_mode)
    
    def get_status(self) -> dict:
        """
        Get current monitor status.
        
        Returns:
            Status dictionary
        """
        return {
            'is_running': self.state.is_running,
            'folder_path': self.state.folder_path,
            'mode': 'average' if self.state.average_mode else 'single',
            'last_scan': self.state.last_scan_number,
            'total_scans': self.state.total_scans_loaded,
            'accumulated_scans': self._num_accumulated,
            'poll_interval': self.state.poll_interval
        }


def quick_monitor_start(
    folder_path: str,
    on_data_callback: Callable[[NMRData, int], None],
    average_mode: bool = True,
    poll_interval: float = 1.0
) -> RealtimeDataMonitor:
    """
    Quick start realtime monitoring (convenience function).
    
    Args:
        folder_path: Experiment folder to monitor
        on_data_callback: Callback function(data, scan_count)
        average_mode: True for average, False for single scans
        poll_interval: Poll interval in seconds
    
    Returns:
        RealtimeDataMonitor instance
    
    Example:
        >>> def update_plot(data, count):
        ...     print(f"New data! Scans: {count}")
        ...     plot(data.freq_axis, abs(data.freq_data))
        >>> 
        >>> monitor = quick_monitor_start("/path/to/exp", update_plot)
        >>> # ... later ...
        >>> monitor.stop()
    """
    monitor = RealtimeDataMonitor(folder_path, poll_interval)
    
    if average_mode:
        monitor.on_average_updated = on_data_callback
    else:
        monitor.on_new_scan = on_data_callback
    
    monitor.start(average_mode)
    
    return monitor
