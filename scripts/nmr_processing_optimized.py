"""
NMR Data Processing - Optimized Version
========================================

Features:
1. Savgol filtering
2. Time domain truncation
3. Exponential apodization
4. Hanning window
5. Zero filling
6. FFT frequency domain analysis
7. Interactive parameter adjustment
8. Automatic parameter saving

Author: Optimized Version
Date: 2025-10-06
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import savgol_filter
from matplotlib.widgets import Slider, Button
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
import matplotlib
matplotlib.use('TkAgg')


@dataclass
class ProcessingParameters:
    """Processing parameters class"""
    zf_factor: float = 0.0          # Zero filling factor
    hanning: int = 0                 # Hanning window switch (0 or 1)
    conv_points: int = 300           # Savgol filter window points
    poly_order: int = 2              # Savgol polynomial order
    trunc_start: int = 10            # Time domain start truncation points
    trunc_end: int = 10              # Time domain end truncation points
    apodization: float = 0.0         # Exponential decay factor (T2*)
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, path: str):
        """Save parameters to JSON file"""
        save_dir = os.path.join(path, "processing_params")
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "parameters.json"), 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        # Also save as numpy format for backward compatibility
        np.save(os.path.join(save_dir, "zf_factor.npy"), self.zf_factor)
        np.save(os.path.join(save_dir, "hanning.npy"), self.hanning)
        np.save(os.path.join(save_dir, "conv_points.npy"), self.conv_points)
        np.save(os.path.join(save_dir, "poly_order.npy"), self.poly_order)
        np.save(os.path.join(save_dir, "trunc_start.npy"), self.trunc_start)
        np.save(os.path.join(save_dir, "trunc_end.npy"), self.trunc_end)
        np.save(os.path.join(save_dir, "apodization.npy"), self.apodization)
        
        print(f"Parameters saved to: {save_dir}")
    
    @classmethod
    def load(cls, path: str):
        """Load parameters from file"""
        save_dir = os.path.join(path, "processing_params")
        
        # Try to load JSON first
        json_path = os.path.join(save_dir, "parameters.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            print("Loaded parameters from JSON")
            return cls.from_dict(data)
        
        # Backward compatibility with old numpy format
        old_dir = os.path.join(path, "savgol_filter_save")
        if os.path.exists(old_dir):
            params = cls(
                zf_factor=float(np.load(os.path.join(old_dir, "zf_factor_stored.npy"))),
                hanning=int(np.load(os.path.join(old_dir, "hanning_stored.npy"))),
                conv_points=int(np.load(os.path.join(old_dir, "conv_points_stored.npy"))),
                poly_order=int(np.load(os.path.join(old_dir, "poly_order_stored.npy"))),
                trunc_start=int(np.load(os.path.join(old_dir, "trunc_stored.npy"))),
                trunc_end=int(np.load(os.path.join(old_dir, "trunc_f_stored.npy"))),
                apodization=float(np.load(os.path.join(old_dir, "apod_stored.npy")))
            )
            print("Loaded parameters from old numpy format")
            return params
        
        print("No saved parameters found, using defaults")
        return cls()


class NMRProcessor:
    """NMR data processor class"""
    
    def __init__(self, halp: np.ndarray, sampling_rate: float, acq_time: float):
        """
        Initialize processor
        
        Args:
            halp: Time domain signal data
            sampling_rate: Sampling rate (Hz)
            acq_time: Acquisition time (s)
        """
        self.halp_original = halp.copy()
        self.sampling_rate = sampling_rate
        self.acq_time_original = acq_time
        
        # Cache variables
        self._last_params = None
        self._cached_result = None
    
    def process(self, params: ProcessingParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute complete data processing pipeline
        
        Args:
            params: Processing parameters
        
        Returns:
            (xf, yf, time_domain): Frequency axis, frequency domain signal, processed time domain signal
        """
        # Disable cache for now - check if params changed
        params_dict = params.to_dict()
        cache_key = str(sorted(params_dict.items()))
        
        if self._last_params == cache_key and self._cached_result is not None:
            return self._cached_result
        
        # 1. Savgol filtering
        smooth_signal = savgol_filter(
            self.halp_original, 
            int(params.conv_points), 
            int(params.poly_order), 
            mode="mirror"
        )
        signal_corrected = self.halp_original - smooth_signal
        
        # 2. Time domain truncation
        if params.trunc_start >= len(signal_corrected) or params.trunc_end >= len(signal_corrected):
            print("Warning: Truncation parameters too large, using original signal")
            signal_corrected = signal_corrected.copy()
        else:
            signal_corrected = signal_corrected[int(params.trunc_start):-int(params.trunc_end)]
        
        # Calculate effective acquisition time
        acq_time_effective = self.acq_time_original * (len(signal_corrected) / len(self.halp_original))
        
        # 3. Exponential apodization
        if params.apodization != 0:
            t = np.linspace(0, acq_time_effective, len(signal_corrected))
            apodization_window = np.exp(-params.apodization * t)
            signal_corrected = signal_corrected * apodization_window
        
        # 4. Hanning window
        if int(params.hanning) == 1:
            signal_corrected = np.hanning(len(signal_corrected)) * signal_corrected
        
        # 5. Zero filling
        if params.zf_factor > 0:
            zero_fill_length = int(len(signal_corrected) * params.zf_factor)
            zero_fill = np.ones(zero_fill_length) * np.mean(signal_corrected)
            signal_corrected = np.concatenate((signal_corrected, zero_fill))
        
        # 6. FFT transform
        yf = fft(signal_corrected)
        xf = np.linspace(0, self.sampling_rate, len(yf))
        
        # Cache result
        self._last_params = cache_key
        self._cached_result = (xf, yf, signal_corrected)
        
        return xf, yf, signal_corrected
    
    def calculate_snr(self, xf: np.ndarray, yf: np.ndarray, 
                     signal_range: Tuple[float, float], 
                     noise_range: Tuple[float, float],
                     num_scans: int = 1) -> float:
        """
        Calculate signal-to-noise ratio
        
        Args:
            xf: Frequency axis
            yf: Frequency domain signal
            signal_range: Signal frequency range [Hz]
            noise_range: Noise frequency range [Hz]
            num_scans: Number of scans
        
        Returns:
            SNR value
        """
        try:
            # Signal region
            sig_idx_start = np.where(xf >= signal_range[0])[0][0]
            sig_idx_end = np.where(xf >= signal_range[1])[0][0]
            signal_amplitude = np.max(np.abs(yf[sig_idx_start:sig_idx_end]))
            
            # Noise region
            noise_idx_start = np.where(xf >= noise_range[0])[0][0]
            noise_idx_end = np.where(xf >= noise_range[1])[0][0]
            noise_std = np.std(np.abs(yf[noise_idx_start:noise_idx_end]))
            
            # SNR calculation (considering number of scans)
            snr = (signal_amplitude / noise_std) / np.sqrt(num_scans)
            return snr
        
        except Exception as e:
            print(f"SNR calculation error: {e}")
            return 0.0


class InteractiveNMRViewer:
    """Interactive NMR data viewer"""
    
    def __init__(self, processor: NMRProcessor, params: ProcessingParameters,
                 freq_range1: Tuple[float, float] = (0, 30),
                 freq_range2: Tuple[float, float] = (50, 275)):
        """
        Initialize viewer
        
        Args:
            processor: NMR processor instance
            params: Initial processing parameters
            freq_range1: First spectrum window range
            freq_range2: Second spectrum window range
        """
        self.processor = processor
        self.params = params
        self.freq_range1 = freq_range1
        self.freq_range2 = freq_range2
        
        # Create figure interface
        self._create_figure()
        self._create_sliders()
        
        # Initial plot
        self.update(None)
    
    def _create_figure(self):
        """Create figure window"""
        self.fig = plt.figure(figsize=(12, 10))
        
        # Adjust subplot layout, leave space for sliders
        self.fig.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, hspace=0.3)
        
        # Create three subplots
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        
        self.ax1.set_title("Time Domain Signal (Processed)", fontsize=12, fontweight='bold')
        self.ax1.set_ylabel("Signal Intensity [a.u.]")
        self.ax1.set_xlabel("Time [s]")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title(f"Frequency Domain Signal ({self.freq_range1[0]}-{self.freq_range1[1]} Hz)", 
                          fontsize=12, fontweight='bold')
        self.ax2.set_ylabel("Amplitude [a.u.]")
        self.ax2.set_xlabel("Frequency [Hz]")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title(f"Frequency Domain Signal ({self.freq_range2[0]}-{self.freq_range2[1]} Hz)", 
                          fontsize=12, fontweight='bold')
        self.ax3.set_ylabel("Amplitude [a.u.]")
        self.ax3.set_xlabel("Frequency [Hz]")
        self.ax3.grid(True, alpha=0.3)
        
        # Initialize lines with dummy data (will be updated immediately)
        self.line1, = self.ax1.plot([0, 1], [0, 0], 'k-', linewidth=0.8)
        self.line2, = self.ax2.plot([0, 1], [0, 0], 'b-', linewidth=0.8)
        self.line3, = self.ax3.plot([0, 1], [0, 0], 'b-', linewidth=0.8)
    
    def _create_sliders(self):
        """Create slider controls"""
        slider_color = 'lightgoldenrodyellow'
        
        # Zero filling factor slider
        ax_zf = self.fig.add_axes([0.15, 0.26, 0.7, 0.02], facecolor=slider_color)
        self.slider_zf = Slider(ax_zf, 'Zero Fill', 0, 10, 
                                valinit=self.params.zf_factor, valstep=0.01)
        
        # Hanning window slider
        ax_han = self.fig.add_axes([0.15, 0.23, 0.7, 0.02], facecolor=slider_color)
        self.slider_han = Slider(ax_han, 'Hanning', 0, 1, 
                                 valinit=self.params.hanning, valstep=1)
        
        # Convolution points slider
        ax_conv = self.fig.add_axes([0.15, 0.20, 0.7, 0.02], facecolor=slider_color)
        self.slider_conv = Slider(ax_conv, 'Conv Points', 2, 12000, 
                                  valinit=self.params.conv_points, valstep=1)
        
        # Polynomial order slider
        ax_poly = self.fig.add_axes([0.15, 0.17, 0.7, 0.02], facecolor=slider_color)
        self.slider_poly = Slider(ax_poly, 'Poly Order', 1, 20, 
                                  valinit=self.params.poly_order, valstep=1)
        
        # Start truncation slider
        ax_trunc_s = self.fig.add_axes([0.15, 0.14, 0.7, 0.02], facecolor=slider_color)
        self.slider_trunc_s = Slider(ax_trunc_s, 'Trunc Start', 1, 8000, 
                                     valinit=self.params.trunc_start, valstep=1)
        
        # End truncation slider
        ax_trunc_e = self.fig.add_axes([0.15, 0.11, 0.7, 0.02], facecolor=slider_color)
        self.slider_trunc_e = Slider(ax_trunc_e, 'Trunc End', 1, 60000, 
                                     valinit=self.params.trunc_end, valstep=1)
        
        # Apodization factor slider
        ax_apod = self.fig.add_axes([0.15, 0.08, 0.7, 0.02], facecolor=slider_color)
        self.slider_apod = Slider(ax_apod, 'Apodization', -2, 2, 
                                  valinit=self.params.apodization, valstep=0.01)
        
        # Save button
        ax_save = self.fig.add_axes([0.15, 0.03, 0.15, 0.03])
        self.btn_save = Button(ax_save, 'Save Params', color='lightgreen', hovercolor='green')
        self.btn_save.on_clicked(self.save_params)
        
        # Export button
        ax_export = self.fig.add_axes([0.35, 0.03, 0.15, 0.03])
        self.btn_export = Button(ax_export, 'Export Data', color='lightblue', hovercolor='blue')
        self.btn_export.on_clicked(self.export_data)
        
        # Bind update events
        self.slider_zf.on_changed(self.update)
        self.slider_han.on_changed(self.update)
        self.slider_conv.on_changed(self.update)
        self.slider_poly.on_changed(self.update)
        self.slider_trunc_s.on_changed(self.update)
        self.slider_trunc_e.on_changed(self.update)
        self.slider_apod.on_changed(self.update)
        
        # Store all sliders for keyboard control
        self.sliders = [
            self.slider_zf,
            self.slider_han,
            self.slider_conv,
            self.slider_poly,
            self.slider_trunc_s,
            self.slider_trunc_e,
            self.slider_apod
        ]
        self.active_slider_idx = 0
        
        # Bind keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Mark for initial update
        self._needs_initial_update = True
    
    def update(self, val):
        """Update plots"""
        try:
            # Update parameters
            self.params.zf_factor = self.slider_zf.val
            self.params.hanning = int(self.slider_han.val)
            self.params.conv_points = int(self.slider_conv.val)
            self.params.poly_order = int(self.slider_poly.val)
            self.params.trunc_start = int(self.slider_trunc_s.val)
            self.params.trunc_end = int(self.slider_trunc_e.val)
            self.params.apodization = self.slider_apod.val
            
            # Debug: print parameter update
            print(f"Updating: ZF={self.params.zf_factor:.2f}, Conv={self.params.conv_points}, "
                  f"Poly={self.params.poly_order}, Trunc={self.params.trunc_start}/{self.params.trunc_end}, "
                  f"Apod={self.params.apodization:.2f}")
            
            # Process data
            xf, yf, time_signal = self.processor.process(self.params)
            print(f"Processed: {len(xf)} points, max amplitude: {np.max(np.abs(yf)):.2e}")
            
            # Update time domain plot
            time_axis = np.linspace(0, self.processor.acq_time_original * (1 + self.params.zf_factor), 
                                   len(time_signal))
            self.line1.set_data(time_axis, time_signal)
            self.ax1.set_xlim(0, time_axis[-1])
            self.ax1.set_ylim(np.min(time_signal), np.max(time_signal))
            
            # Update frequency domain plot 1
            idx1_start = np.where(xf >= self.freq_range1[0])[0][0]
            idx1_end = np.where(xf >= self.freq_range1[1])[0][0]
            yf_abs = np.abs(yf)
            
            # Avoid index errors
            if idx1_end > idx1_start + 10:
                max_val1 = np.max(yf_abs[idx1_start:idx1_end][10:])
            else:
                max_val1 = np.max(yf_abs[idx1_start:idx1_end])
            min_val1 = np.min(yf_abs[idx1_start:idx1_end])
            
            self.line2.set_data(xf, yf_abs)
            self.ax2.set_xlim(self.freq_range1)
            self.ax2.set_ylim(min_val1 - max_val1 * 0.1, max_val1 * 1.1)
            
            # Update frequency domain plot 2
            idx2_start = np.where(xf >= self.freq_range2[0])[0][0]
            idx2_end = np.where(xf >= self.freq_range2[1])[0][0]
            max_val2 = np.max(yf_abs[idx2_start:idx2_end])
            min_val2 = np.min(yf_abs[idx2_start:idx2_end])
            
            self.line3.set_data(xf, yf_abs)
            self.ax3.set_xlim(self.freq_range2)
            self.ax3.set_ylim(min_val2, max_val2 * 1.05)
            
            print(f"Plot updated: freq range 1: {min_val1:.2e} to {max_val1:.2e}")
            
            # Force redraw - try multiple methods
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.relim()
                ax.autoscale_view(tight=False)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            print("Update complete\n")
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()

    
    def on_key_press(self, event):
        """Handle keyboard events for slider control"""
        if event.key == 'up':
            # Switch to previous slider (cycle through)
            self.active_slider_idx = (self.active_slider_idx - 1) % len(self.sliders)
            self._highlight_active_slider()
            print(f"Active slider: {self.sliders[self.active_slider_idx].label.get_text()}")
            
        elif event.key == 'down':
            # Switch to next slider (cycle through)
            self.active_slider_idx = (self.active_slider_idx + 1) % len(self.sliders)
            self._highlight_active_slider()
            print(f"Active slider: {self.sliders[self.active_slider_idx].label.get_text()}")
            
        elif event.key == 'left':
            # Decrease active slider value
            slider = self.sliders[self.active_slider_idx]
            new_val = slider.val - slider.valstep
            if new_val >= slider.valmin:
                slider.set_val(new_val)
                
        elif event.key == 'right':
            # Increase active slider value
            slider = self.sliders[self.active_slider_idx]
            new_val = slider.val + slider.valstep
            if new_val <= slider.valmax:
                slider.set_val(new_val)
        
        elif event.key == 'shift+left':
            # Large decrease
            slider = self.sliders[self.active_slider_idx]
            new_val = slider.val - slider.valstep * 10
            if new_val >= slider.valmin:
                slider.set_val(new_val)
            else:
                slider.set_val(slider.valmin)
                
        elif event.key == 'shift+right':
            # Large increase
            slider = self.sliders[self.active_slider_idx]
            new_val = slider.val + slider.valstep * 10
            if new_val <= slider.valmax:
                slider.set_val(new_val)
            else:
                slider.set_val(slider.valmax)
    
    def _highlight_active_slider(self):
        """Highlight the currently active slider"""
        # Reset all slider labels to normal
        for i, slider in enumerate(self.sliders):
            if i == self.active_slider_idx:
                slider.label.set_weight('bold')
                slider.label.set_color('red')
            else:
                slider.label.set_weight('normal')
                slider.label.set_color('black')
        self.fig.canvas.draw()
    
    def save_params(self, event):
        """Save parameters button callback"""
        from tkinter import filedialog
        import tkinter as tk
        
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(title="Select folder to save parameters")
        
        if folder_path:
            self.params.save(folder_path)
            print(f"Parameters saved to: {folder_path}")
        
        root.destroy()
    
    def export_data(self, event):
        """Export data button callback"""
        from tkinter import filedialog, simpledialog
        import tkinter as tk
        
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(title="Select folder to export data")
        
        if folder_path:
            name = simpledialog.askstring("Filename", "Enter filename prefix:", initialvalue="processed")
            
            if name:
                xf, yf, time_signal = self.processor.process(self.params)
                
                # Save frequency domain data
                np.save(os.path.join(folder_path, f"{name}_frequency.npy"), xf)
                np.save(os.path.join(folder_path, f"{name}_amplitude.npy"), yf)
                
                # Save time domain data
                time_axis = np.linspace(0, self.processor.acq_time_original * (1 + self.params.zf_factor), 
                                       len(time_signal))
                np.save(os.path.join(folder_path, f"{name}_time_axis.npy"), time_axis)
                np.save(os.path.join(folder_path, f"{name}_time_signal.npy"), time_signal)
                
                # Save as CSV
                freq_data = np.column_stack((xf, np.abs(yf)))
                np.savetxt(os.path.join(folder_path, f"{name}_spectrum.csv"), 
                          freq_data, delimiter=',', header='Frequency(Hz),Amplitude', comments='')
                
                print(f"Data exported to: {folder_path}")
                print(f"  - {name}_frequency.npy")
                print(f"  - {name}_amplitude.npy")
                print(f"  - {name}_time_axis.npy")
                print(f"  - {name}_time_signal.npy")
                print(f"  - {name}_spectrum.csv")
        
        root.destroy()
    
    def show(self):
        """Show window"""
        # Force initial update before showing
        if hasattr(self, '_needs_initial_update') and self._needs_initial_update:
            self.update(None)
            self._needs_initial_update = False
        plt.show()


def load_nmr_data(folder_path: str) -> Tuple[np.ndarray, float, float]:
    """
    Load NMR data
    
    Args:
        folder_path: Data folder path
    
    Returns:
        (halp, sampling_rate, acq_time)
    """
    try:
        # Try to load compiled data
        compiled_path = os.path.join(folder_path, "halp_compiled.npy")
        
        if os.path.exists(compiled_path):
            halp = np.load(compiled_path)
            sampling_rate = np.load(os.path.join(folder_path, "sampling_rate_compiled.npy"))
            acq_time = np.load(os.path.join(folder_path, "acq_time_compiled.npy"))
            print("Successfully loaded compiled data")
            return halp, sampling_rate, acq_time
        else:
            raise FileNotFoundError("Compiled data files not found, please run data compilation first")
    
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise


def main():
    """Main function"""
    print("=" * 60)
    print("NMR Data Processing - Optimized Version".center(60))
    print("=" * 60)
    print()
    
    # Select data folder
    from tkinter import filedialog
    import tkinter as tk
    
    root = tk.Tk()
    root.withdraw()
    
    folder_path = filedialog.askdirectory(title="Select NMR data folder")
    
    if not folder_path:
        print("No folder selected, exiting")
        return
    
    root.destroy()
    
    print(f"Current experiment: {folder_path}")
    print()
    
    # Load data
    try:
        halp, sampling_rate, acq_time = load_nmr_data(folder_path)
        print(f"Acquisition time: {acq_time:.4f} s")
        print(f"Sampling rate: {sampling_rate:.2f} Hz")
        print(f"Data points: {len(halp)}")
        print()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Load or create processing parameters
    params = ProcessingParameters.load(folder_path)
    print()
    print("Current parameters:")
    print(f"  Zero filling factor: {params.zf_factor}")
    print(f"  Hanning window: {params.hanning}")
    print(f"  Convolution points: {params.conv_points}")
    print(f"  Polynomial order: {params.poly_order}")
    print(f"  Start truncation: {params.trunc_start}")
    print(f"  End truncation: {params.trunc_end}")
    print(f"  Apodization factor: {params.apodization}")
    print()
    
    # Create processor
    processor = NMRProcessor(halp, sampling_rate, acq_time)
    
    # Create interactive viewer
    viewer = InteractiveNMRViewer(processor, params)
    
    print("Interactive window opened")
    print("  - Use sliders to adjust parameters")
    print("  - Click 'Save Params' button to save current parameters")
    print("  - Click 'Export Data' button to export processed data")
    print()
    print("Keyboard shortcuts:")
    print("  - Up/Down arrows: Switch between sliders")
    print("  - Left/Right arrows: Adjust selected slider value")
    print("  - Shift + Left/Right: Adjust slider value by 10x step")
    print()
    
    viewer.show()


if __name__ == "__main__":
    main()
