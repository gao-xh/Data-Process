"""
NMR Data Processing UI
======================

PySide6 UI for testing NMR data processing library.

Features:
- Load data from file or generate test data
- Real-time processing with parameter adjustment
- Visualization of time and frequency domain
- SNR calculation and quality metrics
- Export results
"""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QCheckBox, QFileDialog, QTextEdit, QTabWidget,
    QSplitter, QProgressBar, QMessageBox, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Import nmr_processing_lib
from nmr_processing_lib import DataInterface
from nmr_processing_lib.core import apply_fft, apply_phase_correction
from nmr_processing_lib.processing import (
    savgol_filter_nmr,
    apply_apodization,
    zero_filling,
    baseline_correction
)
from nmr_processing_lib.quality import (
    calculate_snr,
    estimate_noise
)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for plotting"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class ProcessingWorker(QThread):
    """Background worker for data processing"""
    
    finished = Signal(object)  # Processed data
    error = Signal(str)        # Error message
    
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params
    
    def run(self):
        try:
            result = self.process_data()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def process_data(self):
        """Process data with given parameters"""
        time_data = self.data.time_data.copy()
        sampling_rate = self.data.sampling_rate
        
        # Step 1: Savitzky-Golay filter
        if self.params['use_savgol']:
            time_data = savgol_filter_nmr(
                time_data,
                window_length=self.params['savgol_window'],
                polyorder=self.params['savgol_poly']
            )
        
        # Step 2: Apodization
        if self.params['use_apodization']:
            time_data = apply_apodization(
                time_data,
                t2_star=1.0 / (2 * np.pi * self.params['apod_broadening']),  # Convert Hz to t2_star
                apodization_type=self.params['apod_type']
            )
        
        # Step 3: Zero filling
        if self.params['use_zerofill']:
            time_data = zero_filling(
                time_data,
                final_size=self.params['zerofill_size']
            )
        
        # Step 4: FFT
        freq_axis, spectrum = apply_fft(time_data, sampling_rate)
        
        # Step 5: Phase correction
        if self.params['use_phase']:
            spectrum = apply_phase_correction(
                spectrum,
                phase0=self.params['phase0'],
                phase1=self.params['phase1']
            )
        
        # Step 6: Baseline correction
        if self.params['use_baseline']:
            spectrum = baseline_correction(
                spectrum,
                method=self.params['baseline_method']
            )
        
        return {
            'time_data': time_data,
            'freq_axis': freq_axis,
            'spectrum': spectrum,
            'original_data': self.data
        }


class NMRProcessingUI(QMainWindow):
    """Main window for NMR data processing"""
    
    def __init__(self):
        super().__init__()
        
        self.current_data = None
        self.processed_result = None
        self.processing_params = self.get_default_params()
        
        self.init_ui()
        
        # Generate test data on startup
        self.generate_test_data()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("NMR Data Processing Tool")
        self.setGeometry(100, 50, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Plots
        right_panel = self.create_plot_panel()
        main_layout.addWidget(right_panel, 3)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data loading
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.load_file_btn = QPushButton("Load File")
        self.load_file_btn.clicked.connect(self.load_data_file)
        btn_layout.addWidget(self.load_file_btn)
        
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_data_folder)
        btn_layout.addWidget(self.load_folder_btn)
        
        self.test_data_btn = QPushButton("Generate Test Data")
        self.test_data_btn.clicked.connect(self.generate_test_data)
        btn_layout.addWidget(self.test_data_btn)
        data_layout.addLayout(btn_layout)
        
        self.data_info = QLabel("No data loaded")
        self.data_info.setWordWrap(True)
        data_layout.addWidget(self.data_info)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Processing parameters - use tab widget
        param_tabs = QTabWidget()
        
        # Tab 1: Filtering
        filter_tab = self.create_filter_tab()
        param_tabs.addTab(filter_tab, "Filtering")
        
        # Tab 2: Transform
        transform_tab = self.create_transform_tab()
        param_tabs.addTab(transform_tab, "Transform")
        
        # Tab 3: Correction
        correction_tab = self.create_correction_tab()
        param_tabs.addTab(correction_tab, "Correction")
        
        layout.addWidget(param_tabs)
        
        # Processing button
        self.process_btn = QPushButton("▶ Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results display
        results_group = QGroupBox("Processing Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Export button
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
        
        return panel
    
    def create_filter_tab(self):
        """Create filtering parameters tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Savitzky-Golay filter
        savgol_group = QGroupBox("Savitzky-Golay Filter")
        savgol_layout = QGridLayout()
        
        self.use_savgol = QCheckBox("Enable")
        self.use_savgol.setChecked(True)
        self.use_savgol.stateChanged.connect(self.on_param_changed)
        savgol_layout.addWidget(self.use_savgol, 0, 0, 1, 3)
        
        savgol_layout.addWidget(QLabel("Window:"), 1, 0)
        self.savgol_window = QSpinBox()
        self.savgol_window.setRange(5, 12000)
        self.savgol_window.setSingleStep(1)
        self.savgol_window.setValue(300)
        self.savgol_window.valueChanged.connect(self.on_param_changed)
        savgol_layout.addWidget(self.savgol_window, 1, 1)
        self.savgol_window_label = QLabel("300")
        savgol_layout.addWidget(self.savgol_window_label, 1, 2)
        
        savgol_layout.addWidget(QLabel("Poly Order:"), 2, 0)
        self.savgol_poly = QSpinBox()
        self.savgol_poly.setRange(1, 20)
        self.savgol_poly.setValue(2)
        self.savgol_poly.valueChanged.connect(self.on_param_changed)
        savgol_layout.addWidget(self.savgol_poly, 2, 1)
        self.savgol_poly_label = QLabel("2")
        savgol_layout.addWidget(self.savgol_poly_label, 2, 2)
        
        savgol_group.setLayout(savgol_layout)
        layout.addWidget(savgol_group)
        
        # Truncation
        trunc_group = QGroupBox("Time Domain Truncation")
        trunc_layout = QGridLayout()
        
        self.use_truncation = QCheckBox("Enable")
        self.use_truncation.setChecked(False)
        self.use_truncation.stateChanged.connect(self.on_param_changed)
        trunc_layout.addWidget(self.use_truncation, 0, 0, 1, 3)
        
        trunc_layout.addWidget(QLabel("Start Points:"), 1, 0)
        self.trunc_start = QSpinBox()
        self.trunc_start.setRange(0, 60000)
        self.trunc_start.setValue(10)
        self.trunc_start.valueChanged.connect(self.on_param_changed)
        trunc_layout.addWidget(self.trunc_start, 1, 1)
        self.trunc_start_label = QLabel("10")
        trunc_layout.addWidget(self.trunc_start_label, 1, 2)
        
        trunc_layout.addWidget(QLabel("End Points:"), 2, 0)
        self.trunc_end = QSpinBox()
        self.trunc_end.setRange(0, 60000)
        self.trunc_end.setValue(10)
        self.trunc_end.valueChanged.connect(self.on_param_changed)
        trunc_layout.addWidget(self.trunc_end, 2, 1)
        self.trunc_end_label = QLabel("10")
        trunc_layout.addWidget(self.trunc_end_label, 2, 2)
        
        trunc_group.setLayout(trunc_layout)
        layout.addWidget(trunc_group)
        
        # Apodization
        apod_group = QGroupBox("Apodization (T2* decay)")
        apod_layout = QGridLayout()
        
        self.use_apodization = QCheckBox("Enable")
        self.use_apodization.setChecked(False)
        self.use_apodization.stateChanged.connect(self.on_param_changed)
        apod_layout.addWidget(self.use_apodization, 0, 0, 1, 3)
        
        apod_layout.addWidget(QLabel("T2* Factor:"), 1, 0)
        self.apod_t2star = QDoubleSpinBox()
        self.apod_t2star.setRange(-2.0, 2.0)
        self.apod_t2star.setSingleStep(0.01)
        self.apod_t2star.setValue(0.0)
        self.apod_t2star.valueChanged.connect(self.on_param_changed)
        apod_layout.addWidget(self.apod_t2star, 1, 1)
        self.apod_t2star_label = QLabel("0.00")
        apod_layout.addWidget(self.apod_t2star_label, 1, 2)
        
        # Hanning window
        apod_layout.addWidget(QLabel("Hanning:"), 2, 0)
        self.use_hanning = QCheckBox("Apply")
        self.use_hanning.setChecked(False)
        self.use_hanning.stateChanged.connect(self.on_param_changed)
        apod_layout.addWidget(self.use_hanning, 2, 1)
        
        apod_group.setLayout(apod_layout)
        layout.addWidget(apod_group)
        
        layout.addStretch()
        return tab
    
    def create_transform_tab(self):
        """Create transform parameters tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Zero filling
        zerofill_group = QGroupBox("Zero Filling")
        zerofill_layout = QGridLayout()
        
        self.use_zerofill = QCheckBox("Enable")
        self.use_zerofill.setChecked(False)
        self.use_zerofill.stateChanged.connect(self.on_param_changed)
        zerofill_layout.addWidget(self.use_zerofill, 0, 0, 1, 3)
        
        zerofill_layout.addWidget(QLabel("Factor:"), 1, 0)
        self.zerofill_factor = QDoubleSpinBox()
        self.zerofill_factor.setRange(0.0, 10.0)
        self.zerofill_factor.setSingleStep(0.01)
        self.zerofill_factor.setValue(0.0)
        self.zerofill_factor.valueChanged.connect(self.on_param_changed)
        zerofill_layout.addWidget(self.zerofill_factor, 1, 1)
        self.zerofill_factor_label = QLabel("0.00")
        zerofill_layout.addWidget(self.zerofill_factor_label, 1, 2)
        
        zerofill_group.setLayout(zerofill_layout)
        layout.addWidget(zerofill_group)
        
        # FFT info
        fft_group = QGroupBox("Fourier Transform (FFT)")
        fft_layout = QVBoxLayout()
        fft_layout.addWidget(QLabel("Always enabled"))
        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)
        
        layout.addStretch()
        return tab
    
    def create_correction_tab(self):
        """Create correction parameters tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Phase correction
        phase_group = QGroupBox("Phase Correction")
        phase_layout = QGridLayout()
        
        self.use_phase = QCheckBox("Enable")
        self.use_phase.setChecked(False)
        self.use_phase.stateChanged.connect(self.on_param_changed)
        phase_layout.addWidget(self.use_phase, 0, 0, 1, 2)
        
        phase_layout.addWidget(QLabel("Zero Order:"), 1, 0)
        self.phase0_slider = QSlider(Qt.Horizontal)
        self.phase0_slider.setRange(-180, 180)
        self.phase0_slider.setValue(0)
        self.phase0_slider.valueChanged.connect(self.on_phase0_changed)
        phase_layout.addWidget(self.phase0_slider, 1, 1)
        
        self.phase0_value = QLabel("0°")
        phase_layout.addWidget(self.phase0_value, 1, 2)
        
        phase_layout.addWidget(QLabel("First Order:"), 2, 0)
        self.phase1_slider = QSlider(Qt.Horizontal)
        self.phase1_slider.setRange(-180, 180)
        self.phase1_slider.setValue(0)
        self.phase1_slider.valueChanged.connect(self.on_phase1_changed)
        phase_layout.addWidget(self.phase1_slider, 2, 1)
        
        self.phase1_value = QLabel("0°")
        phase_layout.addWidget(self.phase1_value, 2, 2)
        
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        # Baseline correction
        baseline_group = QGroupBox("Baseline Correction")
        baseline_layout = QGridLayout()
        
        self.use_baseline = QCheckBox("Enable")
        self.use_baseline.setChecked(False)
        self.use_baseline.stateChanged.connect(self.on_param_changed)
        baseline_layout.addWidget(self.use_baseline, 0, 0, 1, 2)
        
        baseline_layout.addWidget(QLabel("Method:"), 1, 0)
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(['polynomial', 'linear', 'constant'])
        self.baseline_method.currentTextChanged.connect(self.on_param_changed)
        baseline_layout.addWidget(self.baseline_method, 1, 1)
        
        baseline_group.setLayout(baseline_layout)
        layout.addWidget(baseline_group)
        
        layout.addStretch()
        return tab
    
    def create_plot_panel(self):
        """Create plot panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for plots
        plot_tabs = QTabWidget()
        
        # Time domain plot
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        
        self.time_canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.time_toolbar = NavigationToolbar(self.time_canvas, self)
        time_layout.addWidget(self.time_toolbar)
        time_layout.addWidget(self.time_canvas)
        
        plot_tabs.addTab(time_widget, "Time Domain")
        
        # Frequency domain plot
        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        
        self.freq_canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.freq_toolbar = NavigationToolbar(self.freq_canvas, self)
        freq_layout.addWidget(self.freq_toolbar)
        freq_layout.addWidget(self.freq_canvas)
        
        plot_tabs.addTab(freq_widget, "Frequency Domain")
        
        layout.addWidget(plot_tabs)
        
        return panel
    
    def get_default_params(self):
        """Get default processing parameters"""
        return {
            'use_savgol': True,
            'savgol_window': 51,
            'savgol_poly': 2,
            'use_apodization': False,
            'apod_type': 'exponential',
            'apod_broadening': 5.0,
            'use_zerofill': False,
            'zerofill_size': 2048,
            'use_phase': False,
            'phase0': 0.0,
            'phase1': 0.0,
            'use_baseline': False,
            'baseline_method': 'polynomial'
        }
    
    def update_params_from_ui(self):
        """Update parameters from UI"""
        self.processing_params = {
            'use_savgol': self.use_savgol.isChecked(),
            'savgol_window': self.savgol_window.value(),
            'savgol_poly': self.savgol_poly.value(),
            'use_apodization': self.use_apodization.isChecked(),
            'apod_type': self.apod_type.currentText(),
            'apod_broadening': self.apod_broadening.value(),
            'use_zerofill': self.use_zerofill.isChecked(),
            'zerofill_size': self.zerofill_size.value(),
            'use_phase': self.use_phase.isChecked(),
            'phase0': self.phase0_slider.value(),
            'phase1': self.phase1_slider.value(),
            'use_baseline': self.use_baseline.isChecked(),
            'baseline_method': self.baseline_method.currentText()
        }
    
    @Slot()
    def on_param_changed(self):
        """Handle parameter change"""
        # Auto-process if data is loaded
        if self.current_data is not None:
            self.process_data()
    
    @Slot(int)
    def on_phase0_changed(self, value):
        """Handle phase0 slider change"""
        self.phase0_value.setText(f"{value}°")
        self.on_param_changed()
    
    @Slot(int)
    def on_phase1_changed(self, value):
        """Handle phase1 slider change"""
        self.phase1_value.setText(f"{value}°")
        self.on_param_changed()
    
    @Slot()
    def load_data_file(self):
        """Load data from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select NMR Data File",
            "",
            "Data files (*.dat *.txt *.csv);;All files (*.*)"
        )
        
        if filename:
            try:
                # Try to load data
                self.current_data = DataInterface.from_file(filename)
                
                self.data_info.setText(
                    f"✅ Loaded: {filename}\n"
                    f"Data points: {len(self.current_data.time_data)}\n"
                    f"Sampling rate: {self.current_data.sampling_rate} Hz"
                )
                
                self.process_btn.setEnabled(True)
                self.plot_raw_data()
                self.process_data()
                
            except Exception as e:
                QMessageBox.critical(self, "Load Failed", f"Cannot load file:\n{e}")
    
    @Slot()
    def generate_test_data(self):
        """Generate test data"""
        # Generate simulated NMR signal
        n_points = 1000
        sampling_rate = 5000.0
        time = np.arange(n_points) / sampling_rate
        
        # Signal components
        signal = np.zeros(n_points, dtype=complex)
        
        # Add peaks at different frequencies
        frequencies = [10, 50, 150, -80]  # Hz
        amplitudes = [1.0, 0.7, 0.5, 0.3]
        decays = [0.1, 0.15, 0.2, 0.12]
        
        for freq, amp, decay in zip(frequencies, amplitudes, decays):
            signal += amp * np.exp(1j * 2 * np.pi * freq * time) * np.exp(-decay * time)
        
        # Add noise
        noise = 0.05 * (np.random.randn(n_points) + 1j * np.random.randn(n_points))
        signal += noise
        
        # Create NMRData object
        self.current_data = DataInterface.from_arrays(
            signal,
            sampling_rate=sampling_rate,
            acquisition_time=n_points / sampling_rate
        )
        
        self.data_info.setText(
            f"✅ Test data generated\n"
            f"Data points: {n_points}\n"
            f"Sampling rate: {sampling_rate} Hz\n"
            f"Peak positions: {', '.join([f'{f} Hz' for f in frequencies])}"
        )
        
        self.process_btn.setEnabled(True)
        self.plot_raw_data()
        self.process_data()
    
    def plot_raw_data(self):
        """Plot raw time domain data"""
        if self.current_data is None:
            return
        
        time_data = self.current_data.time_data
        time_axis = np.arange(len(time_data)) / self.current_data.sampling_rate
        
        self.time_canvas.axes.clear()
        self.time_canvas.axes.plot(time_axis * 1000, np.real(time_data), 'b-', alpha=0.7, label='Real')
        self.time_canvas.axes.plot(time_axis * 1000, np.imag(time_data), 'r-', alpha=0.7, label='Imag')
        self.time_canvas.axes.set_xlabel('Time (ms)')
        self.time_canvas.axes.set_ylabel('Amplitude')
        self.time_canvas.axes.set_title('Raw Time Domain Signal')
        self.time_canvas.axes.legend()
        self.time_canvas.axes.grid(True, alpha=0.3)
        self.time_canvas.draw()
    
    @Slot()
    def process_data(self):
        """Process data with current parameters"""
        if self.current_data is None:
            return
        
        # Update parameters
        self.update_params_from_ui()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.process_btn.setEnabled(False)
        
        # Start processing in background
        self.worker = ProcessingWorker(self.current_data, self.processing_params)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()
    
    @Slot(object)
    def on_processing_finished(self, result):
        """Handle processing finished"""
        self.processed_result = result
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # Plot results
        self.plot_processed_data()
        
        # Calculate and display metrics
        self.calculate_metrics()
        
        # Enable export
        self.export_btn.setEnabled(True)
    
    @Slot(str)
    def on_processing_error(self, error):
        """Handle processing error"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Processing Failed", f"Data processing failed:\n{error}")
    
    def plot_processed_data(self):
        """Plot processed data"""
        if self.processed_result is None:
            return
        
        time_data = self.processed_result['time_data']
        freq_axis = self.processed_result['freq_axis']
        spectrum = self.processed_result['spectrum']
        
        # Time domain
        time_axis = np.arange(len(time_data)) / self.current_data.sampling_rate
        
        self.time_canvas.axes.clear()
        self.time_canvas.axes.plot(time_axis * 1000, np.real(time_data), 'b-', alpha=0.7, label='Real')
        self.time_canvas.axes.plot(time_axis * 1000, np.imag(time_data), 'r-', alpha=0.7, label='Imag')
        self.time_canvas.axes.set_xlabel('Time (ms)')
        self.time_canvas.axes.set_ylabel('Amplitude')
        self.time_canvas.axes.set_title('Processed Time Domain Signal')
        self.time_canvas.axes.legend()
        self.time_canvas.axes.grid(True, alpha=0.3)
        self.time_canvas.draw()
        
        # Frequency domain
        self.freq_canvas.axes.clear()
        self.freq_canvas.axes.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.5)
        self.freq_canvas.axes.set_xlabel('Frequency (Hz)')
        self.freq_canvas.axes.set_ylabel('Amplitude')
        self.freq_canvas.axes.set_title('Frequency Domain Spectrum')
        self.freq_canvas.axes.grid(True, alpha=0.3)
        
        # Mark peaks (simple peak detection)
        try:
            # Find local maxima
            spectrum_abs = np.abs(spectrum)
            threshold = 0.1 * np.max(spectrum_abs)
            
            # Simple peak detection
            peaks_idx = []
            for i in range(1, len(spectrum_abs) - 1):
                if (spectrum_abs[i] > spectrum_abs[i-1] and 
                    spectrum_abs[i] > spectrum_abs[i+1] and 
                    spectrum_abs[i] > threshold):
                    peaks_idx.append(i)
            
            if peaks_idx:
                peak_freqs = freq_axis[peaks_idx]
                peak_amps = spectrum_abs[peaks_idx]
                self.freq_canvas.axes.plot(peak_freqs, peak_amps, 'ro', markersize=8, label='Peaks')
                self.freq_canvas.axes.legend()
        except:
            pass
        
        self.freq_canvas.draw()
    
    def calculate_metrics(self):
        """Calculate and display quality metrics"""
        if self.processed_result is None:
            return
        
        freq_axis = self.processed_result['freq_axis']
        spectrum = self.processed_result['spectrum']
        
        results = []
        
        # SNR calculation
        try:
            # Try to find peak automatically
            freq_range = (freq_axis.min() + 50, freq_axis.max() - 50)
            noise_range = (freq_axis.max() - 200, freq_axis.max() - 100)
            
            snr = calculate_snr(freq_axis, spectrum, freq_range, noise_range)
            results.append(f"Signal-to-Noise Ratio (SNR): {snr:.2f}")
        except Exception as e:
            results.append(f"SNR calculation failed: {e}")
        
        # Noise level
        try:
            noise = estimate_noise(freq_axis, spectrum)
            results.append(f"Noise level: {noise:.2e}")
        except Exception as e:
            results.append(f"Noise estimation failed: {e}")
        
        # Find peaks (simple detection)
        try:
            spectrum_abs = np.abs(spectrum)
            threshold = 0.1 * np.max(spectrum_abs)
            
            # Simple peak detection
            peaks = []
            for i in range(1, len(spectrum_abs) - 1):
                if (spectrum_abs[i] > spectrum_abs[i-1] and 
                    spectrum_abs[i] > spectrum_abs[i+1] and 
                    spectrum_abs[i] > threshold):
                    peaks.append({
                        'frequency': freq_axis[i],
                        'amplitude': spectrum_abs[i],
                        'index': i
                    })
            
            # Sort by amplitude
            peaks = sorted(peaks, key=lambda x: x['amplitude'], reverse=True)
            
            results.append(f"\nDetected {len(peaks)} peaks:")
            for i, peak in enumerate(peaks[:5], 1):  # Show top 5
                results.append(
                    f"  Peak {i}: {peak['frequency']:.2f} Hz, "
                    f"Amplitude={peak['amplitude']:.2e}"
                )
        except Exception as e:
            results.append(f"Peak detection failed: {e}")
        
        # Processing info
        results.append(f"\nProcessing parameters:")
        if self.processing_params['use_savgol']:
            results.append(f"  ✓ Savgol filter (window={self.processing_params['savgol_window']})")
        if self.processing_params['use_apodization']:
            results.append(f"  ✓ Apodization ({self.processing_params['apod_type']})")
        if self.processing_params['use_zerofill']:
            results.append(f"  ✓ Zero fill (to {self.processing_params['zerofill_size']} points)")
        if self.processing_params['use_phase']:
            results.append(f"  ✓ Phase correction (φ0={self.processing_params['phase0']}°)")
        if self.processing_params['use_baseline']:
            results.append(f"  ✓ Baseline correction ({self.processing_params['baseline_method']})")
        
        self.results_text.setText('\n'.join(results))
    
    @Slot()
    def export_results(self):
        """Export processing results"""
        if self.processed_result is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Results",
            "nmr_processed.npz",
            "NumPy archive (*.npz);;All files (*.*)"
        )
        
        if filename:
            try:
                np.savez(
                    filename,
                    time_data=self.processed_result['time_data'],
                    freq_axis=self.processed_result['freq_axis'],
                    spectrum=self.processed_result['spectrum'],
                    sampling_rate=self.current_data.sampling_rate,
                    parameters=self.processing_params
                )
                
                QMessageBox.information(self, "Export Successful", f"Results saved to:\n{filename}")
            
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Cannot save file:\n{e}")


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    
    # Set font
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    
    window = NMRProcessingUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
