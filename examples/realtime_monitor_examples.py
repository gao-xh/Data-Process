"""
Real-time Monitoring Example
============================

Demonstrates how to use RealtimeDataMonitor for live acquisition monitoring.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from nmr_processing_lib.utils import RealtimeDataMonitor, quick_monitor_start
from nmr_processing_lib.core import ProcessingParameters
from nmr_processing_lib.processing import (
    savgol_filter_nmr,
    truncate_time_domain,
    apply_apodization,
    zero_filling
)
from nmr_processing_lib.core.transforms import apply_fft
from nmr_processing_lib.processing.postprocessing import gaussian_broadening


# =============================================================================
# Example 1: Simple Realtime Monitor
# =============================================================================

def example_simple_monitor():
    """
    Simplest way to monitor folder for new scans.
    """
    folder = r"C:\NMR_Data\experiment_001"
    
    def on_new_data(nmr_data, scan_count):
        """Callback when new data arrives"""
        print(f"\n=== New data! Total scans: {scan_count} ===")
        print(f"Data points: {len(nmr_data.time_data)}")
        print(f"Sampling rate: {nmr_data.sampling_rate} Hz")
    
    # Start monitoring (average mode)
    monitor = quick_monitor_start(
        folder_path=folder,
        on_data_callback=on_new_data,
        average_mode=True,
        poll_interval=1.0
    )
    
    # Let it run for a while
    print("Monitoring... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()


# =============================================================================
# Example 2: Monitor with Real-time Plotting
# =============================================================================

def example_realtime_plotting():
    """
    Monitor folder and update plot in real-time.
    """
    folder = r"C:\NMR_Data\experiment_001"
    
    # Setup plot
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    line1, = ax1.plot([], [], 'b-', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal (a.u.)')
    ax1.set_title('Time Domain')
    ax1.grid(True, alpha=0.3)
    
    line2, = ax2.plot([], [], 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Intensity (a.u.)')
    ax2.set_title('Frequency Domain')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    def update_plot(nmr_data, scan_count):
        """Update plots with new averaged data"""
        # Process data
        params = ProcessingParameters()
        
        # Time domain processing
        filtered = savgol_filter_nmr(nmr_data.time_data, params.savgol_window)
        truncated = truncate_time_domain(
            filtered,
            params.truncation_start,
            params.truncation_end
        )
        apodized = apply_apodization(truncated, params.apodization_t2)
        zero_filled = zero_filling(apodized, params.zero_fill_factor)
        
        # FFT
        freq_axis, spectrum = apply_fft(
            zero_filled,
            nmr_data.sampling_rate,
            zero_padding=params.zero_fill_factor
        )
        
        # Broadening
        broadened = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
        
        # Update time domain plot
        time_axis = np.arange(len(nmr_data.time_data)) / nmr_data.sampling_rate
        line1.set_data(time_axis, nmr_data.time_data.real)
        ax1.relim()
        ax1.autoscale_view()
        ax1.set_title(f'Time Domain (Scans: {scan_count})')
        
        # Update frequency domain plot
        line2.set_data(freq_axis, np.abs(broadened))
        ax2.relim()
        ax2.autoscale_view()
        ax2.set_title(f'Frequency Domain (Scans: {scan_count})')
        
        # Redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # Start monitoring
    monitor = RealtimeDataMonitor(folder, poll_interval=2.0)
    monitor.on_average_updated = update_plot
    monitor.start(average_mode=True)
    
    print("Monitoring with real-time plotting...")
    print("Close plot window to stop")
    
    try:
        plt.show(block=True)
    except:
        pass
    finally:
        monitor.stop()


# =============================================================================
# Example 3: Advanced Monitor with Manual Control
# =============================================================================

def example_advanced_monitor():
    """
    Full-featured monitor with mode switching and status display.
    """
    folder = r"C:\NMR_Data\experiment_001"
    
    # Create monitor
    monitor = RealtimeDataMonitor(folder, poll_interval=1.0)
    
    # Callbacks
    def on_single_scan(data, scan_num):
        print(f"[SINGLE] Scan #{scan_num}: {len(data.time_data)} points")
    
    def on_average(data, total_scans):
        print(f"[AVERAGE] Total scans: {total_scans}")
        # Can process data here
        if total_scans % 10 == 0:
            print(f"  -> SNR improved by factor: {np.sqrt(total_scans):.1f}")
    
    def on_count_changed(count):
        print(f"Total files detected: {count}")
    
    def on_error(msg):
        print(f"ERROR: {msg}")
    
    # Set callbacks
    monitor.on_new_scan = on_single_scan
    monitor.on_average_updated = on_average
    monitor.on_scan_count_changed = on_count_changed
    monitor.on_error = on_error
    
    # Start in average mode
    monitor.start(average_mode=True)
    
    print("\n=== Monitor Control Demo ===")
    print("Commands:")
    print("  status  - Show status")
    print("  single  - Switch to single scan mode")
    print("  average - Switch to average mode")
    print("  reset   - Reset average")
    print("  stop    - Stop monitoring")
    print("  start   - Start monitoring")
    print("  quit    - Exit")
    
    try:
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'status':
                status = monitor.get_status()
                print("\nCurrent Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
            
            elif cmd == 'single':
                monitor.set_mode(average_mode=False)
                print("Switched to SINGLE scan mode")
            
            elif cmd == 'average':
                monitor.set_mode(average_mode=True)
                print("Switched to AVERAGE mode")
            
            elif cmd == 'reset':
                monitor.reset_average()
                print("Average reset")
            
            elif cmd == 'stop':
                monitor.stop()
                print("Monitoring stopped")
            
            elif cmd == 'start':
                mode = input("Mode (single/average): ").strip().lower()
                monitor.start(average_mode=(mode == 'average'))
                print(f"Monitoring started in {mode} mode")
            
            elif cmd == 'quit':
                break
            
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        monitor.stop()
        print("\nMonitor stopped")


# =============================================================================
# Example 4: Integration with Processing Pipeline
# =============================================================================

def example_monitor_with_pipeline():
    """
    Combine realtime monitoring with full processing pipeline.
    """
    folder = r"C:\NMR_Data\experiment_001"
    
    # Create parameter set
    params = ProcessingParameters(
        savgol_window=51,
        truncation_start=100,
        truncation_end=-100,
        apodization_t2=0.05,
        zero_fill_factor=2,
        broadening_hz=5.0
    )
    
    # Setup results storage
    results = {
        'scan_counts': [],
        'snr_values': []
    }
    
    def process_and_analyze(nmr_data, scan_count):
        """Full processing pipeline"""
        print(f"\n=== Processing {scan_count} averaged scans ===")
        
        # Complete processing chain
        filtered = savgol_filter_nmr(nmr_data.time_data, params.savgol_window)
        truncated = truncate_time_domain(filtered, params.truncation_start, params.truncation_end)
        apodized = apply_apodization(truncated, params.apodization_t2)
        zero_filled = zero_filling(apodized, params.zero_fill_factor)
        
        freq_axis, spectrum = apply_fft(
            zero_filled,
            nmr_data.sampling_rate,
            zero_padding=params.zero_fill_factor
        )
        
        final_spectrum = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
        
        # Calculate SNR (assume peak around 0 Hz)
        from nmr_processing_lib.quality.snr import calculate_snr
        
        snr = calculate_snr(
            freq_axis,
            final_spectrum,
            peak_range=(-50, 50),
            noise_range=(200, 400)
        )
        
        print(f"  SNR: {snr:.1f}")
        print(f"  Expected SNR improvement: {np.sqrt(scan_count):.1f}x")
        
        # Store results
        results['scan_counts'].append(scan_count)
        results['snr_values'].append(snr)
        
        # Check if target SNR reached
        if snr > 100:
            print("\n*** Target SNR reached! ***")
    
    # Start monitoring
    monitor = RealtimeDataMonitor(folder, poll_interval=2.0)
    monitor.on_average_updated = process_and_analyze
    monitor.start(average_mode=True)
    
    print("Monitoring with full processing pipeline...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        
        # Show results
        if results['scan_counts']:
            print("\n=== Final Results ===")
            print(f"Total scans: {results['scan_counts'][-1]}")
            print(f"Final SNR: {results['snr_values'][-1]:.1f}")
            
            # Optional: plot SNR vs scans
            try:
                plt.figure(figsize=(8, 5))
                plt.plot(results['scan_counts'], results['snr_values'], 'bo-', label='Measured')
                plt.plot(results['scan_counts'], 
                        [results['snr_values'][0] * np.sqrt(n/results['scan_counts'][0]) 
                         for n in results['scan_counts']], 
                        'r--', label='Theoretical')
                plt.xlabel('Number of Scans')
                plt.ylabel('SNR')
                plt.title('SNR Improvement During Averaging')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            except:
                pass


# =============================================================================
# Example 5: UI Integration Pattern (for PySide6)
# =============================================================================

class MonitorWidgetExample:
    """
    Example pattern for integrating with PySide6 UI.
    
    This shows how to structure the monitor for UI integration.
    """
    
    def __init__(self, folder_path: str):
        self.monitor = RealtimeDataMonitor(folder_path)
        
        # Connect callbacks (these would be UI update methods)
        self.monitor.on_new_scan = self.update_single_scan_view
        self.monitor.on_average_updated = self.update_average_view
        self.monitor.on_scan_count_changed = self.update_scan_counter
        self.monitor.on_error = self.show_error_message
    
    def update_single_scan_view(self, data, scan_num):
        """Update UI with single scan (would update plot widget)"""
        # In real UI:
        # self.plot_widget.update_data(data.freq_axis, data.freq_data)
        # self.scan_label.setText(f"Scan #{scan_num}")
        print(f"[UI] Update single scan view: #{scan_num}")
    
    def update_average_view(self, data, total_scans):
        """Update UI with averaged data"""
        # In real UI:
        # self.plot_widget.update_data(data.freq_axis, data.freq_data)
        # self.scans_label.setText(f"Averaged: {total_scans} scans")
        # self.snr_improvement_label.setText(f"SNR improvement: {np.sqrt(total_scans):.1f}x")
        print(f"[UI] Update average view: {total_scans} scans")
    
    def update_scan_counter(self, count):
        """Update scan counter display"""
        # In real UI:
        # self.counter_label.setText(f"Total: {count}")
        print(f"[UI] Scan counter: {count}")
    
    def show_error_message(self, msg):
        """Show error in UI"""
        # In real UI:
        # QMessageBox.warning(self, "Monitor Error", msg)
        print(f"[UI ERROR] {msg}")
    
    def start_monitoring(self, average_mode: bool):
        """Start button clicked"""
        self.monitor.start(average_mode)
    
    def stop_monitoring(self):
        """Stop button clicked"""
        self.monitor.stop()
    
    def toggle_mode(self, average_mode: bool):
        """Mode switch toggled"""
        self.monitor.set_mode(average_mode)
    
    def reset_average(self):
        """Reset button clicked"""
        self.monitor.reset_average()


if __name__ == '__main__':
    print("Real-time Monitor Examples")
    print("=" * 50)
    print("\nChoose an example:")
    print("1. Simple monitor")
    print("2. Real-time plotting")
    print("3. Advanced control")
    print("4. Full processing pipeline")
    print("5. UI integration pattern")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    examples = {
        '1': example_simple_monitor,
        '2': example_realtime_plotting,
        '3': example_advanced_monitor,
        '4': example_monitor_with_pipeline,
        '5': lambda: print("\n(See MonitorWidgetExample class for UI pattern)")
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")
