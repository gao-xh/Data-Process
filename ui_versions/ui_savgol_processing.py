"""
NMR Savgol Filtering + Processing UI
====================================

Based on the Jupyter notebook workflow for NMR data processing.

Features matching notebook:
1. Load data from folder (compiled or individual scans)
2. Savgol filtering for baseline removal
3. Time domain truncation
4. Apodization (exponential decay)
5. Hanning window
6. Zero filling
7. FFT transform
8. Real-time visualization
9. Save/load processing parameters (JSON)
10. SNR calculation
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QFileDialog, QTextEdit, QMessageBox,
    QGridLayout, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.signal
from scipy.fft import fft

# Import nmrduino_util
try:
    import nmrduino_util as nmr_util
    HAS_NMRDUINO = True
except:
    HAS_NMRDUINO = False
    print("Warning: nmrduino_util not found, some features disabled")


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class ProcessingWorker(QThread):
    """Background worker for processing"""
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, halp, sampling_rate, acq_time, params):
        super().__init__()
        self.halp = halp
        self.sampling_rate = sampling_rate
        self.acq_time = acq_time
        self.params = params
        self._running = True
    
    def run(self):
        try:
            if self._running:
                result = self.process()
                if self._running:
                    self.finished.emit(result)
        except Exception as e:
            if self._running:
                self.error.emit(str(e))
    
    def stop(self):
        """Stop the worker thread"""
        self._running = False
    
    def process(self):
        """Process following notebook workflow"""
        halp = self.halp.copy()
        sampling_rate = self.sampling_rate
        acq_time = self.acq_time
        
        # Step 1: Savgol filtering (baseline removal)
        smooth_svd = scipy.signal.savgol_filter(
            halp, 
            int(self.params['conv_points']),
            int(self.params['poly_order']), 
            mode="mirror"
        )
        svd_corrected = halp - smooth_svd
        
        # Step 2: Time domain truncation
        trunc_start = int(self.params['trunc_start'])
        trunc_end = int(self.params['trunc_end'])
        svd_corrected = svd_corrected[trunc_start:-trunc_end if trunc_end > 0 else None]
        acq_time_effective = acq_time * (len(svd_corrected) / len(halp))
        
        # Step 3: Apodization (exponential decay)
        t = np.linspace(0, acq_time_effective, len(svd_corrected))
        apodization_window = np.exp(-self.params['apod_t2star'] * t)
        svd_corrected = svd_corrected * apodization_window
        
        # Step 4: Hanning window
        if int(self.params['use_hanning']) == 1:
            svd_corrected = np.hanning(len(svd_corrected)) * svd_corrected
        
        # Step 5: Zero filling
        zf_factor = self.params['zf_factor']
        if zf_factor > 0:
            zf_length = int(len(svd_corrected) * zf_factor)
            svd_corrected = np.concatenate((
                svd_corrected,
                np.ones(zf_length) * np.mean(svd_corrected)
            ))
        
        # Step 6: FFT
        yf = fft(svd_corrected)
        xf = np.linspace(0, sampling_rate, len(yf))
        
        return {
            'time_data': svd_corrected,
            'freq_axis': xf,
            'spectrum': yf,
            'acq_time_effective': acq_time * (1 + zf_factor)
        }


class SavgolProcessingUI(QMainWindow):
    """Main UI for Savgol Filtering + Processing"""
    
    def __init__(self):
        super().__init__()
        
        # Data
        self.halp = None
        self.sampling_rate = None
        self.acq_time = None
        self.scan_count = 0
        self.current_path = None
        
        # Processing results
        self.processed = None
        
        # Worker thread
        self.worker = None
        
        # Default parameters (matching notebook)
        self.params = {
            'zf_factor': 0.0,
            'use_hanning': 0,
            'conv_points': 300,
            'poly_order': 2,
            'trunc_start': 10,
            'trunc_end': 10,
            'apod_t2star': 0.0
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("NMR Savgol Filtering + Processing")
        self.setGeometry(50, 50, 1600, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        
        # Left: controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right: plots
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, 3)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data loading
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        btn_layout.addWidget(self.load_folder_btn)
        
        self.load_params_btn = QPushButton("Load Params")
        self.load_params_btn.clicked.connect(self.load_parameters)
        btn_layout.addWidget(self.load_params_btn)
        
        data_layout.addLayout(btn_layout)
        
        self.data_info = QLabel("No data loaded")
        self.data_info.setWordWrap(True)
        data_layout.addWidget(self.data_info)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Processing parameters
        params_group = QGroupBox("Processing Parameters")
        params_layout = QGridLayout()
        
        row = 0
        
        # Savgol filter - Convolution Points
        params_layout.addWidget(QLabel("Convolution Points:"), row, 0)
        self.conv_slider = QSlider(Qt.Horizontal)
        self.conv_slider.setRange(2, 12000)
        self.conv_slider.setValue(300)
        self.conv_slider.valueChanged.connect(self.on_conv_slider_changed)
        params_layout.addWidget(self.conv_slider, row, 1)
        self.conv_label = QLabel("300")
        self.conv_label.setMinimumWidth(60)
        params_layout.addWidget(self.conv_label, row, 2)
        row += 1
        
        # Poly Order
        params_layout.addWidget(QLabel("Poly Order:"), row, 0)
        self.poly_slider = QSlider(Qt.Horizontal)
        self.poly_slider.setRange(1, 20)
        self.poly_slider.setValue(2)
        self.poly_slider.valueChanged.connect(self.on_poly_slider_changed)
        params_layout.addWidget(self.poly_slider, row, 1)
        self.poly_label = QLabel("2")
        self.poly_label.setMinimumWidth(60)
        params_layout.addWidget(self.poly_label, row, 2)
        row += 1
        
        # Truncation Start
        params_layout.addWidget(QLabel("Trunc Start:"), row, 0)
        self.trunc_start_slider = QSlider(Qt.Horizontal)
        self.trunc_start_slider.setRange(0, 60000)
        self.trunc_start_slider.setValue(10)
        self.trunc_start_slider.valueChanged.connect(self.on_trunc_start_changed)
        params_layout.addWidget(self.trunc_start_slider, row, 1)
        self.trunc_start_label = QLabel("10")
        self.trunc_start_label.setMinimumWidth(60)
        params_layout.addWidget(self.trunc_start_label, row, 2)
        row += 1
        
        # Truncation End
        params_layout.addWidget(QLabel("Trunc End:"), row, 0)
        self.trunc_end_slider = QSlider(Qt.Horizontal)
        self.trunc_end_slider.setRange(0, 60000)
        self.trunc_end_slider.setValue(10)
        self.trunc_end_slider.valueChanged.connect(self.on_trunc_end_changed)
        params_layout.addWidget(self.trunc_end_slider, row, 1)
        self.trunc_end_label = QLabel("10")
        self.trunc_end_label.setMinimumWidth(60)
        params_layout.addWidget(self.trunc_end_label, row, 2)
        row += 1
        
        # Apodization T2*
        params_layout.addWidget(QLabel("Apodization T2*:"), row, 0)
        self.apod_slider = QSlider(Qt.Horizontal)
        self.apod_slider.setRange(-200, 200)  # -2.0 to 2.0, step 0.01
        self.apod_slider.setValue(0)
        self.apod_slider.valueChanged.connect(self.on_apod_slider_changed)
        params_layout.addWidget(self.apod_slider, row, 1)
        self.apod_label = QLabel("0.00")
        self.apod_label.setMinimumWidth(60)
        params_layout.addWidget(self.apod_label, row, 2)
        row += 1
        
        # Hanning Window
        params_layout.addWidget(QLabel("Hanning Window:"), row, 0)
        self.use_hanning = QCheckBox("Enable")
        self.use_hanning.stateChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.use_hanning, row, 1)
        row += 1
        
        # Zero Filling Factor
        params_layout.addWidget(QLabel("Zero Fill Factor:"), row, 0)
        self.zf_slider = QSlider(Qt.Horizontal)
        self.zf_slider.setRange(0, 1000)  # 0.0 to 10.0, step 0.01
        self.zf_slider.setValue(0)
        self.zf_slider.valueChanged.connect(self.on_zf_slider_changed)
        params_layout.addWidget(self.zf_slider, row, 1)
        self.zf_label = QLabel("0.00")
        self.zf_label.setMinimumWidth(60)
        params_layout.addWidget(self.zf_label, row, 2)
        row += 1
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # Save parameters button
        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.save_params_btn.setEnabled(False)
        layout.addWidget(self.save_params_btn)
        
        # Results and Metrics
        results_group = QGroupBox("Results & Metrics")
        results_layout = QVBoxLayout()
        
        # SNR display
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("SNR:"))
        self.snr_label = QLabel("--")
        self.snr_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        snr_layout.addWidget(self.snr_label)
        snr_layout.addStretch()
        results_layout.addLayout(snr_layout)
        
        # Peak height display
        peak_layout = QHBoxLayout()
        peak_layout.addWidget(QLabel("Peak Height:"))
        self.peak_label = QLabel("--")
        self.peak_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        peak_layout.addWidget(self.peak_label)
        peak_layout.addStretch()
        results_layout.addLayout(peak_layout)
        
        # Data info display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return panel
    
    def create_plot_panel(self):
        """Create plot panel with 3 subplots"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Time domain
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        self.time_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.time_toolbar = NavigationToolbar(self.time_canvas, self)
        time_layout.addWidget(self.time_toolbar)
        time_layout.addWidget(self.time_canvas)
        
        # Frequency domain (low freq)
        freq1_widget = QWidget()
        freq1_layout = QVBoxLayout(freq1_widget)
        self.freq1_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.freq1_toolbar = NavigationToolbar(self.freq1_canvas, self)
        freq1_layout.addWidget(self.freq1_toolbar)
        freq1_layout.addWidget(self.freq1_canvas)
        
        # Frequency domain (high freq)
        freq2_widget = QWidget()
        freq2_layout = QVBoxLayout(freq2_widget)
        self.freq2_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.freq2_toolbar = NavigationToolbar(self.freq2_canvas, self)
        freq2_layout.addWidget(self.freq2_toolbar)
        freq2_layout.addWidget(self.freq2_canvas)
        
        layout.addWidget(time_widget)
        layout.addWidget(freq1_widget)
        layout.addWidget(freq2_widget)
        
        return panel
    
    @Slot()
    def load_folder(self):
        """Load data from folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
        if not folder:
            return
        
        try:
            self.current_path = folder
            
            if HAS_NMRDUINO:
                # Try to load compiled data
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp = np.load(compiled_path)
                    self.sampling_rate = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                    self.scan_count = nmr_util.scan_number_extraction(folder)
                else:
                    # Load and compile
                    compiled = nmr_util.nmrduino_dat_interp(folder, 0)
                    self.halp = compiled[0]
                    self.sampling_rate = compiled[1]
                    self.acq_time = compiled[2]
                    self.scan_count = nmr_util.scan_number_extraction(folder)
                    
                    # Save compiled
                    np.save(compiled_path, self.halp)
                    np.save(os.path.join(folder, "sampling_rate_compiled.npy"), self.sampling_rate)
                    np.save(os.path.join(folder, "acq_time_compiled.npy"), self.acq_time)
            else:
                # Manual loading
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp = np.load(compiled_path)
                    self.sampling_rate = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                else:
                    raise FileNotFoundError("No compiled data found. Please compile first or install nmrduino_util.")
            
            # Load saved parameters if exist
            param_file = os.path.join(folder, "processing_params.json")
            if os.path.exists(param_file):
                self.load_parameters(param_file)
            
            self.data_info.setText(
                f"Loaded: {os.path.basename(folder)}\n"
                f"Points: {len(self.halp)}\n"
                f"Sampling Rate: {self.sampling_rate:.1f} Hz\n"
                f"Acq Time: {self.acq_time:.3f} s\n"
                f"Scans: {self.scan_count}"
            )
            
            self.process_btn.setEnabled(True)
            self.save_params_btn.setEnabled(True)
            
            # Auto process
            self.process_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{e}")
    
    @Slot()
    def on_conv_slider_changed(self, value):
        """Handle convolution points slider change"""
        self.conv_label.setText(str(value))
        self.on_param_changed()
    
    @Slot()
    def on_poly_slider_changed(self, value):
        """Handle poly order slider change"""
        self.poly_label.setText(str(value))
        self.on_param_changed()
    
    @Slot()
    def on_trunc_start_changed(self, value):
        """Handle truncation start slider change"""
        self.trunc_start_label.setText(str(value))
        self.on_param_changed()
    
    @Slot()
    def on_trunc_end_changed(self, value):
        """Handle truncation end slider change"""
        self.trunc_end_label.setText(str(value))
        self.on_param_changed()
    
    @Slot()
    def on_apod_slider_changed(self, value):
        """Handle apodization slider change"""
        actual_value = value / 100.0
        self.apod_label.setText(f"{actual_value:.2f}")
        self.on_param_changed()
    
    @Slot()
    def on_zf_slider_changed(self, value):
        """Handle zero fill slider change"""
        actual_value = value / 100.0
        self.zf_label.setText(f"{actual_value:.2f}")
        self.on_param_changed()
    
    @Slot()
    def on_param_changed(self):
        """Handle parameter change"""
        # Auto process if data loaded
        if self.halp is not None:
            self.process_data()
    
    @Slot()
    def process_data(self):
        """Process data with current parameters"""
        if self.halp is None:
            return
        
        # Stop previous worker if running
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # Update parameters from UI
        self.params = {
            'zf_factor': self.zf_slider.value() / 100.0,
            'use_hanning': 1 if self.use_hanning.isChecked() else 0,
            'conv_points': self.conv_slider.value(),
            'poly_order': self.poly_slider.value(),
            'trunc_start': self.trunc_start_slider.value(),
            'trunc_end': self.trunc_end_slider.value(),
            'apod_t2star': self.apod_slider.value() / 100.0
        }
        
        # Process in background
        self.worker = ProcessingWorker(
            self.halp,
            self.sampling_rate,
            self.acq_time,
            self.params
        )
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()
    
    @Slot(object)
    def on_processing_finished(self, result):
        """Handle processing finished"""
        self.processed = result
        self.plot_results()
        self.calculate_metrics()
    
    @Slot(str)
    def on_processing_error(self, error):
        """Handle processing error"""
        QMessageBox.critical(self, "Processing Error", f"Failed to process:\n{error}")
    
    def plot_results(self):
        """Plot processing results"""
        if self.processed is None:
            return
        
        time_data = self.processed['time_data']
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        acq_time = self.processed['acq_time_effective']
        
        # Time domain
        time_axis = np.linspace(0, acq_time, len(time_data))
        self.time_canvas.axes.clear()
        self.time_canvas.axes.plot(time_axis, np.real(time_data), 'k', linewidth=0.5)
        self.time_canvas.axes.set_xlabel('Time (s)')
        self.time_canvas.axes.set_ylabel('Amplitude')
        self.time_canvas.axes.set_title('Time Domain Signal')
        self.time_canvas.axes.grid(True, alpha=0.3)
        self.time_canvas.draw()
        
        # Frequency domain - low freq (0-30 Hz)
        freq_range_low = [0, 30]
        idx_low = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
        
        self.freq1_canvas.axes.clear()
        self.freq1_canvas.axes.plot(freq_axis[idx_low], np.abs(spectrum)[idx_low], 'k', linewidth=0.5)
        self.freq1_canvas.axes.set_xlabel('Frequency (Hz)')
        self.freq1_canvas.axes.set_ylabel('Amplitude')
        self.freq1_canvas.axes.set_title(f'Frequency Domain: {freq_range_low[0]}-{freq_range_low[1]} Hz')
        self.freq1_canvas.axes.grid(True, alpha=0.3)
        self.freq1_canvas.draw()
        
        # Frequency domain - high freq (100-275 Hz)
        freq_range_high = [100, 275]
        idx_high = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
        
        self.freq2_canvas.axes.clear()
        self.freq2_canvas.axes.plot(freq_axis[idx_high], np.abs(spectrum)[idx_high], 'k', linewidth=0.5)
        self.freq2_canvas.axes.set_xlabel('Frequency (Hz)')
        self.freq2_canvas.axes.set_ylabel('Amplitude')
        self.freq2_canvas.axes.set_title(f'Frequency Domain: {freq_range_high[0]}-{freq_range_high[1]} Hz')
        self.freq2_canvas.axes.grid(True, alpha=0.3)
        self.freq2_canvas.draw()
    
    def calculate_metrics(self):
        """Calculate and display metrics"""
        if self.processed is None:
            return
        
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        spectrum_abs = np.abs(spectrum)
        
        results = []
        results.append("Processing Applied:")
        results.append(f"  Savgol: window={self.params['conv_points']}, poly={self.params['poly_order']}")
        results.append(f"  Truncation: start={self.params['trunc_start']}, end={self.params['trunc_end']}")
        results.append(f"  Apodization T2*: {self.params['apod_t2star']:.2f}")
        results.append(f"  Hanning: {'Yes' if self.params['use_hanning'] else 'No'}")
        results.append(f"  Zero Fill Factor: {self.params['zf_factor']:.2f}")
        
        # Calculate peak height
        peak_height = np.max(spectrum_abs)
        self.peak_label.setText(f"{peak_height:.2f}")
        
        # SNR calculation (if nmrduino_util available)
        snr = 0
        if HAS_NMRDUINO:
            try:
                frequency_range_snr = [11, 14]
                noise_range_snr = [350, 400]
                snr = nmr_util.snr_calc(freq_axis, spectrum_abs, 
                                       frequency_range_snr, noise_range_snr)
                if self.scan_count > 0:
                    snr_per_scan = snr / np.sqrt(self.scan_count)
                    self.snr_label.setText(f"{snr_per_scan:.2f}")
                    results.append(f"\nSNR (per scan): {snr_per_scan:.2f}")
                    results.append(f"SNR (total): {snr:.2f}")
                else:
                    self.snr_label.setText(f"{snr:.2f}")
                    results.append(f"\nSNR: {snr:.2f}")
                
                # Calculate noise level
                noise_idx = (freq_axis >= noise_range_snr[0]) & (freq_axis <= noise_range_snr[1])
                noise_level = np.std(spectrum_abs[noise_idx])
                results.append(f"Noise Level: {noise_level:.2f}")
                
            except Exception as e:
                self.snr_label.setText("Error")
                results.append(f"\nSNR calculation failed: {e}")
        else:
            self.snr_label.setText("N/A")
            results.append("\nSNR: nmrduino_util not available")
        
        self.results_text.setText('\n'.join(results))
    
    @Slot()
    def save_parameters(self):
        """Save current parameters to JSON"""
        if self.current_path is None:
            return
        
        save_path = os.path.join(self.current_path, "processing_params.json")
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.params, f, indent=2)
            
            QMessageBox.information(self, "Saved", f"Parameters saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save parameters:\n{e}")
    
    @Slot()
    def load_parameters(self, filepath=None):
        """Load parameters from JSON"""
        if filepath is None:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", 
                self.current_path if self.current_path else "",
                "JSON files (*.json)"
            )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            # Update UI sliders
            self.conv_slider.setValue(int(params.get('conv_points', 300)))
            self.poly_slider.setValue(int(params.get('poly_order', 2)))
            self.trunc_start_slider.setValue(int(params.get('trunc_start', 10)))
            self.trunc_end_slider.setValue(int(params.get('trunc_end', 10)))
            self.apod_slider.setValue(int(float(params.get('apod_t2star', 0.0)) * 100))
            self.use_hanning.setChecked(bool(params.get('use_hanning', 0)))
            self.zf_slider.setValue(int(float(params.get('zf_factor', 0.0)) * 100))
            
            self.params = params
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load parameters:\n{e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop worker thread if running
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)  # Wait up to 1 second
            if self.worker.isRunning():
                self.worker.terminate()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set font
    font = QFont("Arial", 9)
    app.setFont(font)
    
    window = SavgolProcessingUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
