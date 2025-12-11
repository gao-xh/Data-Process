"""
Enhanced NMR Processing UI
==========================

Integrated UI combining:
- Savgol filtering workflow from notebook
- Advanced visualization with multiple views
- Tab-based parameter organization
- Real-time SNR and metrics display
- JSON parameter save/load
- Export functionality
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QRadioButton, QFileDialog, QTextEdit, QMessageBox,
    QGridLayout, QTabWidget, QProgressBar, QComboBox, QSplitter,
    QScrollArea, QMenuBar, QMenu, QDialog, QListWidget, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QSettings, QObject
from PySide6.QtGui import QFont

import matplotlib
try:
    # Try to use QtAgg (preferred for PySide6/PyQt6)
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        # Fallback to Qt5Agg (for PySide2/PyQt5)
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        # Last resort
        print("Warning: Could not load QtAgg or Qt5Agg backend for Matplotlib")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.signal
from scipy.fft import fft

# Import ZULF algorithms
try:
    from nmr_processing_lib.processing.zulf_algorithms import (
        backward_linear_prediction, apply_phase_correction, auto_phase
    )
    from nmr_processing_lib.processing.postprocessing import baseline_correction
    from nmr_processing_lib.processing.filtering import svd_denoising
except ImportError:
    print("Warning: Could not import ZULF algorithms. Phase correction disabled.")
    # Define dummy functions if import fails
    def backward_linear_prediction(data, n, order): return data
    def apply_phase_correction(spec, p0, p1): return spec
    def auto_phase(spec): return 0, 0
    def baseline_correction(freq, spec, method='polynomial', order=1, lambda_=100): return spec, np.zeros_like(spec)

# Import Realtime Monitor
try:
    from nmr_processing_lib.utils.realtime_monitor import RealtimeDataMonitor
except ImportError:
    print("Warning: Could not import RealtimeDataMonitor")
    RealtimeDataMonitor = None

# Import nmrduino_util (using fixed version with proper path handling)
try:
    from nmr_processing_lib import nmrduino_util_fixed as nmr_util
    HAS_NMRDUINO = True
except:
    HAS_NMRDUINO = False
    print("Warning: nmrduino_util not found, some features disabled")


class LiveMonitorSignals(QObject):
    """Signals for Live Monitor bridge"""
    data_received = Signal(object, int)  # nmr_data, scan_count
    error_occurred = Signal(str)
    status_updated = Signal(str)


class MultiFolderDialog(QDialog):
    """Dialog for selecting multiple folders to combine"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Multiple Folders (Combine)")
        self.resize(600, 400)
        self.folders = []
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Add multiple experiment folders to combine them into a single dataset.\n"
                          "This is useful for experiments split across multiple directories.\n"
                          "Data will be averaged weighted by scan count.")
        info_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Folder...")
        add_btn.clicked.connect(self.add_folder)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_folder)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)
        
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #e0e0e0;")
        layout.addWidget(line)
        
        action_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.load_btn = QPushButton("Load Combined Data")
        self.load_btn.clicked.connect(self.accept)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        
        action_layout.addStretch()
        action_layout.addWidget(cancel_btn)
        action_layout.addWidget(self.load_btn)
        layout.addLayout(action_layout)
        
        self.update_buttons()
        
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
        if folder and folder not in self.folders:
            self.folders.append(folder)
            self.list_widget.addItem(folder)
            self.update_buttons()
            
    def remove_folder(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.folders.pop(row)
            self.list_widget.takeItem(row)
            self.update_buttons()
            
    def clear_all(self):
        self.folders = []
        self.list_widget.clear()
        self.update_buttons()
            
    def update_buttons(self):
        self.load_btn.setEnabled(len(self.folders) > 0)
        
    def get_folders(self):
        return self.folders


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget"""
    def __init__(self, parent=None, width=8, height=3.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


class ProcessingWorker(QThread):
    """Background worker for processing"""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    
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
        if not self._running: return None
        
        halp = self.halp.copy()
        sampling_rate = self.sampling_rate
        acq_time = self.acq_time
        
        self.progress.emit("Applying Savgol filter...")
        
        # Step 1: Savgol filtering (baseline removal)
        if not self._running: return None
        smooth_svd = scipy.signal.savgol_filter(
            halp, 
            int(self.params['conv_points']),
            int(self.params['poly_order']), 
            mode="mirror"
        )
        svd_corrected = halp - smooth_svd
        
        # Step 1.2: SVD Denoising (Cadzow)
        if self.params.get('enable_svd', False):
            if not self._running: return None
            self.progress.emit("Applying SVD denoising...")
            rank = int(self.params.get('svd_rank', 5))
            try:
                from nmr_processing_lib.processing.filtering import svd_denoising
                svd_corrected = svd_denoising(svd_corrected, rank)
            except Exception as e:
                print(f"SVD Denoising failed: {e}")

        # Step 2: Time domain truncation (Start)
        if not self._running: return None
        self.progress.emit("Applying truncation...")
        trunc_start = int(self.params['trunc_start'])
        trunc_end = int(self.params['trunc_end'])
        
        # Apply start truncation first
        if trunc_start > 0:
            svd_corrected = svd_corrected[trunc_start:]
            
        # Step 1.5: Backward Linear Prediction
        # Now we reconstruct based on the "good" data after truncation
        n_backward_actual = 0
        lp_train_len = 0
        if self.params.get('enable_recon', False):
            self.progress.emit("Applying Backward LP...")
            n_backward = int(self.params.get('recon_points', 0))
            order = int(self.params.get('recon_order', 10))
            if n_backward > 0:
                # Calculate training length (logic matches zulf_algorithms.py)
                valid_len = len(svd_corrected)
                train_len = int(self.params.get('recon_train_len', 4 * order))
                # Ensure train_len is valid
                train_len = min(valid_len, train_len)
                
                svd_corrected = backward_linear_prediction(svd_corrected, n_backward, order, train_len=train_len)
                n_backward_actual = n_backward
                lp_train_len = train_len
        
        # Apply end truncation
        if trunc_end > 0:
            svd_corrected = svd_corrected[:-trunc_end]
            
        # Calculate effective acquisition time based on final length
        # Note: This assumes sampling rate is constant
        acq_time_effective = acq_time * (len(svd_corrected) / len(halp))
        
        self.progress.emit("Applying apodization...")
        
        # Step 3: Apodization (exponential decay)
        if not self._running: return None
        t = np.linspace(0, acq_time_effective, len(svd_corrected))
        apodization_window = np.exp(-self.params['apod_t2star'] * t)
        svd_corrected = svd_corrected * apodization_window
        
        # Step 4: Hanning window
        if int(self.params['use_hanning']) == 1:
            if not self._running: return None
            self.progress.emit("Applying Hanning window...")
            svd_corrected = np.hanning(len(svd_corrected)) * svd_corrected
        
        self.progress.emit("Applying zero filling...")
        
        # Step 5: Zero filling
        if not self._running: return None
        zf_factor = self.params['zf_factor']
        original_len = len(svd_corrected)
        if zf_factor > 0:
            zf_length = int(len(svd_corrected) * zf_factor)
            svd_corrected = np.concatenate((
                svd_corrected,
                np.ones(zf_length) * np.mean(svd_corrected)
            ))
        
        self.progress.emit("Computing FFT...")
        
        # Step 6: FFT
        if not self._running: return None
        yf = fft(svd_corrected)
        xf = np.linspace(0, sampling_rate, len(yf))
        
        # Step 7: Phase Correction
        self.progress.emit("Applying phase correction...")
        phi0 = self.params.get('phase0', 0.0)
        phi1 = self.params.get('phase1', 0.0)
        
        # Cache the unphased spectrum for fast updates
        spectrum_complex = yf.copy()
        
        # Apply phase with pivot at center
        pivot = len(yf) // 2
        yf_phased = apply_phase_correction(yf, phi0, phi1, pivot_index=pivot)
        
        # Step 8: Baseline Correction (Frequency Domain)
        baseline_method = self.params.get('baseline_method', 'none')
        freq_baseline = np.zeros_like(yf_phased)
        
        if baseline_method != 'none':
            self.progress.emit(f"Applying {baseline_method} baseline correction...")
            try:
                # Extract parameters
                order = int(self.params.get('baseline_order', 1))
                lam = float(self.params.get('baseline_lambda', 100))
                
                # Apply correction
                yf_phased, freq_baseline = baseline_correction(
                    xf, 
                    yf_phased, 
                    method=baseline_method, 
                    order=order, 
                    lambda_=lam
                )
            except Exception as e:
                print(f"Baseline correction error: {e}")
        
        self.progress.emit("Processing complete!")
        
        if not self._running: return None
        
        return {
            'time_data': svd_corrected,
            'freq_axis': xf,
            'spectrum': yf_phased,
            'spectrum_complex': spectrum_complex,
            'acq_time_effective': acq_time * (1 + zf_factor),
            'baseline': smooth_svd,
            'freq_baseline': freq_baseline,
            'n_backward': n_backward_actual,
            'lp_train_len': lp_train_len,
            'original_len': original_len
        }


class EnhancedNMRProcessingUI(QMainWindow):
    """Enhanced main UI for NMR Processing"""
    
    def __init__(self):
        super().__init__()
        
        # Comparison mode flag
        self.comparison_mode = False
        
        # Data A (original/primary)
        self.halp = None
        self.sampling_rate = None
        self.acq_time = None
        self.scan_count = 0
        self.current_path = None
        self.processed = None
        
        # Data B (comparison)
        self.halp_b = None
        self.sampling_rate_b = None
        self.acq_time_b = None
        self.scan_count_b = 0
        self.current_path_b = None
        self.processed_b = None
        
        # Worker thread
        self.worker = None
        
        # Live Monitor
        self.monitor = None
        self.live_signals = LiveMonitorSignals()
        self.live_signals.data_received.connect(self.handle_live_data)
        self.live_signals.error_occurred.connect(self.handle_live_error)
        self.live_signals.status_updated.connect(self.update_live_status)
        self.live_folder = None
        self.is_monitoring = False

        # Default parameters (matching notebook)
        self.params = {
            'zf_factor': 0.0,
            'use_hanning': 0,
            'conv_points': 300,
            'poly_order': 2,
            'trunc_start': 10,
            'trunc_end': 10,
            'apod_t2star': 0.0,
            # New params for ZULF
            'phase0': 0.0,
            'phase1': 0.0,
            'enable_recon': False,
            'recon_points': 0,
            'recon_order': 10
        }
        
        # Parameters for Data B (independent mode)
        self.params_b = {
            'zf_factor': 0.0,
            'use_hanning': 0,
            'conv_points': 300,
            'poly_order': 2,
            'trunc_start': 10,
            'trunc_end': 10,
            'apod_t2star': 0.0,
            # New params for ZULF
            'phase0': 0.0,
            'phase1': 0.0,
            'enable_recon': False,
            'recon_points': 0,
            'recon_order': 10
        }
        
        # Use same parameters for both datasets
        self.use_same_params = True
        
        # Auto-process timer (debounce)
        self.process_timer = QTimer()
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.process_data)
        
        # Settings for saving window state
        self.settings = QSettings('NMR_Processing', 'Enhanced_UI')
        
        self.init_ui()
        self.restore_window_state()
        
        # Force cursor reset after initialization
        QTimer.singleShot(500, self.force_cursor_reset)

    def force_cursor_reset(self):
        """Force cursor to be normal"""
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.setCursor(Qt.ArrowCursor)
        QApplication.setOverrideCursor(Qt.ArrowCursor)
        QApplication.restoreOverrideCursor()

    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #f5f5f5;
                border-bottom: 1px solid #e0e0e0;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QMenu {
                background-color: white;
                border: 1px solid #e0e0e0;
            }
            QMenu::item {
                padding: 8px 30px;
            }
            QMenu::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = file_menu.addAction('Load Folder...')
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_folder)
        
        load_multi_action = file_menu.addAction('Load Multiple Folders (Combine)...')
        load_multi_action.setShortcut('Ctrl+Shift+O')
        load_multi_action.triggered.connect(self.load_multiple_folders)
        
        load_params_action = file_menu.addAction('Load Parameters...')
        load_params_action.setShortcut('Ctrl+L')
        load_params_action.triggered.connect(self.load_parameters)
        
        file_menu.addSeparator()
        
        save_params_action = file_menu.addAction('Save Parameters...')
        save_params_action.setShortcut('Ctrl+S')
        save_params_action.triggered.connect(self.save_parameters)
        
        export_action = file_menu.addAction('Export Results...')
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)
        
        export_figures_action = file_menu.addAction('Export Figures as SVG...')
        export_figures_action.setShortcut('Ctrl+Shift+E')
        export_figures_action.triggered.connect(self.export_figures_svg)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_layout_action = view_menu.addAction('Reset Layout')
        reset_layout_action.triggered.connect(self.reset_layout)
        
        view_menu.addSeparator()
        
        maximize_time_action = view_menu.addAction('Maximize Time Domain')
        maximize_time_action.triggered.connect(lambda: self.maximize_plot('time'))
        
        maximize_freq1_action = view_menu.addAction('Maximize Low Frequency')
        maximize_freq1_action.triggered.connect(lambda: self.maximize_plot('freq1'))
        
        maximize_freq2_action = view_menu.addAction('Maximize High Frequency')
        maximize_freq2_action.triggered.connect(lambda: self.maximize_plot('freq2'))
        
        # Process menu
        process_menu = menubar.addMenu('Process')
        
        process_action = process_menu.addAction('Process Data')
        process_action.setShortcut('F5')
        process_action.triggered.connect(self.process_data)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        # Comparison mode toggle
        self.comparison_mode_action = tools_menu.addAction('Enable Comparison Mode')
        self.comparison_mode_action.setCheckable(True)
        self.comparison_mode_action.setChecked(False)
        self.comparison_mode_action.triggered.connect(self.toggle_comparison_mode)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
    
    def reset_layout(self):
        """Reset layout to default sizes"""
        self.main_splitter.setSizes([500, 1300])
        self.plot_splitter.setSizes([333, 333, 334])
        QMessageBox.information(self, "Layout Reset", "Layout has been reset to default sizes.")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Enhanced NMR Processing</h2>
        <p><b>Version:</b> 1.0</p>
        <p><b>Description:</b> Advanced NMR data processing tool with Savgol filtering, 
        apodization, zero filling, and FFT transformation.</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Real-time data processing</li>
            <li>Interactive parameter adjustment</li>
            <li>SNR calculation and metrics</li>
            <li>JSON parameter save/load</li>
            <li>Resizable panels and maximizable plots</li>
        </ul>
        <p><b>Based on:</b> Jupyter notebook workflow for NMR processing</p>
        """
        QMessageBox.about(self, "About", about_text)
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Enhanced NMR Processing")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout with splitter
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Horizontal splitter for left/right panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                border: 1px solid #bdbdbd;
            }
            QSplitter::handle:hover {
                background-color: #5c7a99;
            }
        """)
        
        # Left: controls (scrollable)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(350)
        control_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #fafafa;
            }
            QScrollBar:vertical {
                border: none;
                background: #f5f5f5;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #bdbdbd;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9e9e9e;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        control_panel = self.create_control_panel()
        control_scroll.setWidget(control_panel)
        self.main_splitter.addWidget(control_scroll)
        
        # Right: plots
        plot_panel = self.create_plot_panel()
        self.main_splitter.addWidget(plot_panel)
        
        # Set initial splitter sizes (30% left, 70% right)
        self.main_splitter.setSizes([500, 1300])
        
        main_layout.addWidget(self.main_splitter)
    
    def get_slider_style(self, color, hover_color):
        """Generate CSS for QSlider"""
        return f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {color};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {hover_color};
            }}
        """

    def get_spinbox_style(self, color, btn_color, hover_color):
        """Generate CSS for QSpinBox/QDoubleSpinBox"""
        return f"""
            QSpinBox, QDoubleSpinBox {{
                font-weight: bold;
                color: white;
                background-color: {color};
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background-color: {btn_color};
                border: none;
                width: 16px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {hover_color};
            }}
        """

    def get_groupbox_style(self, title_color):
        """Generate CSS for QGroupBox"""
        return f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 11px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: {title_color};
            }}
        """

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Data loading
        data_group = QGroupBox("Data Loading")
        data_group.setStyleSheet(self.get_groupbox_style("#424242"))
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #5c7a99;
                color: white;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4a6580;
            }
            QPushButton:pressed {
                background-color: #3a5166;
            }
        """)
        self.load_folder_btn.clicked.connect(self.load_folder)
        btn_layout.addWidget(self.load_folder_btn)
        
        self.load_params_btn = QPushButton("Load Parameters")
        self.load_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
        """)
        self.load_params_btn.clicked.connect(self.load_parameters)
        btn_layout.addWidget(self.load_params_btn)
        
        data_layout.addLayout(btn_layout)
        
        self.data_info = QLabel("No data loaded")
        self.data_info.setWordWrap(True)
        self.data_info.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                color: #616161;
                font-size: 10px;
            }
        """)
        data_layout.addWidget(self.data_info)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Live Monitor Group
        self.live_group = QGroupBox("Live Monitor")
        self.live_group.setStyleSheet(self.get_groupbox_style("#d32f2f"))  # Red accent
        live_layout = QVBoxLayout()
        live_layout.setSpacing(8)
        
        # Folder selection
        live_folder_layout = QHBoxLayout()
        self.live_folder_edit = QLineEdit()
        self.live_folder_edit.setPlaceholderText("Select monitor folder...")
        self.live_folder_edit.setReadOnly(True)
        self.live_folder_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: #f5f5f5;
                color: #616161;
            }
        """)
        
        live_browse_btn = QPushButton("...")
        live_browse_btn.setFixedWidth(30)
        live_browse_btn.clicked.connect(self.select_live_folder)
        
        live_folder_layout.addWidget(self.live_folder_edit)
        live_folder_layout.addWidget(live_browse_btn)
        live_layout.addLayout(live_folder_layout)
        
        # Controls
        live_ctrl_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        self.live_mode_avg = QRadioButton("Average")
        self.live_mode_avg.setChecked(True)
        self.live_mode_avg.toggled.connect(self.on_live_mode_changed)
        
        self.live_mode_single = QRadioButton("Single (Latest)")
        self.live_mode_single.toggled.connect(self.on_live_mode_changed)
        
        mode_layout.addWidget(self.live_mode_avg)
        mode_layout.addWidget(self.live_mode_single)
        live_ctrl_layout.addLayout(mode_layout)
        
        self.monitor_btn = QPushButton("Start Monitoring")
        self.monitor_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        self.monitor_btn.setCheckable(True)
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        self.monitor_btn.setEnabled(False)
        
        live_ctrl_layout.addWidget(self.monitor_btn)
        live_layout.addLayout(live_ctrl_layout)
        
        # Status
        self.live_status_label = QLabel("Ready")
        self.live_status_label.setStyleSheet("color: #757575; font-style: italic; font-size: 10px;")
        live_layout.addWidget(self.live_status_label)
        
        self.live_group.setLayout(live_layout)
        layout.addWidget(self.live_group)

        # Data B Loading (for comparison mode)
        self.data_b_group = QGroupBox("Data B Loading (Comparison)")
        self.data_b_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e3f2fd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f9fc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #1976d2;
            }
        """)
        self.data_b_group.setVisible(False)  # Hidden by default
        
        data_b_layout = QVBoxLayout()
        data_b_layout.setSpacing(8)
        
        load_b_btn = QPushButton("Load Data B Folder")
        load_b_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        load_b_btn.clicked.connect(self.load_folder_b)
        data_b_layout.addWidget(load_b_btn)
        
        self.data_b_info = QLabel("No data B loaded")
        self.data_b_info.setWordWrap(True)
        self.data_b_info.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #ffffff;
                border: 1px solid #bbdefb;
                border-radius: 5px;
                color: #616161;
                font-size: 10px;
            }
        """)
        data_b_layout.addWidget(self.data_b_info)
        
        self.data_b_group.setLayout(data_b_layout)
        layout.addWidget(self.data_b_group)
        
        # Comparison Controls
        self.comparison_controls_group = QGroupBox("Comparison Settings")
        self.comparison_controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #424242;
            }
        """)
        self.comparison_controls_group.setVisible(False)  # Hidden by default
        
        comparison_layout = QVBoxLayout()
        comparison_layout.setSpacing(8)
        
        # Parameters sync option
        self.same_params_checkbox = QCheckBox("Use Same Parameters for Both Datasets")
        self.same_params_checkbox.setChecked(True)
        self.same_params_checkbox.setStyleSheet("font-size: 10px;")
        self.same_params_checkbox.stateChanged.connect(self.on_same_params_changed)
        comparison_layout.addWidget(self.same_params_checkbox)
        
        # Display mode
        display_label = QLabel("Display Mode:")
        display_label.setStyleSheet("font-size: 10px; font-weight: bold; margin-top: 10px;")
        comparison_layout.addWidget(display_label)
        
        self.display_side_by_side = QRadioButton("Side by Side")
        self.display_side_by_side.setChecked(True)
        self.display_side_by_side.setStyleSheet("font-size: 10px;")
        self.display_side_by_side.toggled.connect(self.on_display_mode_changed)
        comparison_layout.addWidget(self.display_side_by_side)
        
        self.display_overlay = QRadioButton("Overlay (Unified Scale)")
        self.display_overlay.setStyleSheet("font-size: 10px;")
        self.display_overlay.toggled.connect(self.on_display_mode_changed)
        comparison_layout.addWidget(self.display_overlay)
        
        self.display_overlay_norm = QRadioButton("Overlay (Normalized)")
        self.display_overlay_norm.setStyleSheet("font-size: 10px;")
        self.display_overlay_norm.toggled.connect(self.on_display_mode_changed)
        comparison_layout.addWidget(self.display_overlay_norm)
        
        # Apply comparison button
        apply_comparison_btn = QPushButton("Apply Comparison")
        apply_comparison_btn.setStyleSheet("""
            QPushButton {
                background-color: #43a047;
                color: white;
                padding: 8px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 4px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #2e7d32;
            }
        """)
        apply_comparison_btn.clicked.connect(self.apply_comparison)
        comparison_layout.addWidget(apply_comparison_btn)
        
        self.comparison_controls_group.setLayout(comparison_layout)
        layout.addWidget(self.comparison_controls_group)
        
        # Data B Parameters Group (initially hidden)
        self.data_b_params_group = QGroupBox("Data B Parameters")
        self.data_b_params_group.setVisible(False)
        self.data_b_params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e57373;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: #ffebee;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #c62828;
            }
        """)
        data_b_params_layout = QVBoxLayout()
        data_b_params_layout.setSpacing(8)
        
        info_label = QLabel("Configure parameters for Data B separately:")
        info_label.setStyleSheet("font-size: 9px; color: #666; font-style: italic;")
        data_b_params_layout.addWidget(info_label)
        
        # Add key parameters for Data B (simplified version)
        param_grid = QGridLayout()
        param_grid.setSpacing(8)
        
        # Savgol Conv Points
        row = 0
        conv_label = QLabel("Conv Points:")
        conv_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(conv_label, row, 0)
        self.conv_spinbox_b = QSpinBox()
        self.conv_spinbox_b.setRange(2, 12000)
        self.conv_spinbox_b.setValue(300)
        self.conv_spinbox_b.setStyleSheet("""
            QSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.conv_spinbox_b, row, 1)
        
        # Polynomial Order
        poly_label = QLabel("Poly Order:")
        poly_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(poly_label, row, 2)
        self.poly_spinbox_b = QSpinBox()
        self.poly_spinbox_b.setRange(1, 20)
        self.poly_spinbox_b.setValue(2)
        self.poly_spinbox_b.setStyleSheet("""
            QSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.poly_spinbox_b, row, 3)
        row += 1
        
        # Truncation
        trunc_label = QLabel("Truncation:")
        trunc_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(trunc_label, row, 0)
        self.trunc_spinbox_b = QSpinBox()
        self.trunc_spinbox_b.setRange(1, 100000)
        self.trunc_spinbox_b.setValue(1600)
        self.trunc_spinbox_b.setStyleSheet("""
            QSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.trunc_spinbox_b, row, 1)
        
        # T2star
        t2_label = QLabel("T2*:")
        t2_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(t2_label, row, 2)
        self.t2_spinbox_b = QDoubleSpinBox()
        self.t2_spinbox_b.setRange(0.0001, 10.0)
        self.t2_spinbox_b.setValue(0.029)
        self.t2_spinbox_b.setDecimals(4)
        self.t2_spinbox_b.setSingleStep(0.001)
        self.t2_spinbox_b.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.t2_spinbox_b, row, 3)
        row += 1
        
        # Hanning Factor
        hanning_label = QLabel("Hanning:")
        hanning_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(hanning_label, row, 0)
        self.hanning_spinbox_b = QDoubleSpinBox()
        self.hanning_spinbox_b.setRange(0.0, 1.0)
        self.hanning_spinbox_b.setValue(0.2)
        self.hanning_spinbox_b.setDecimals(2)
        self.hanning_spinbox_b.setSingleStep(0.05)
        self.hanning_spinbox_b.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.hanning_spinbox_b, row, 1)
        
        # Zero Fill
        zerofill_label = QLabel("Zero Fill:")
        zerofill_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        param_grid.addWidget(zerofill_label, row, 2)
        self.zerofill_spinbox_b = QSpinBox()
        self.zerofill_spinbox_b.setRange(0, 1000000)
        self.zerofill_spinbox_b.setValue(10000)
        self.zerofill_spinbox_b.setSingleStep(1000)
        self.zerofill_spinbox_b.setStyleSheet("""
            QSpinBox {
                background-color: #ef5350;
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        param_grid.addWidget(self.zerofill_spinbox_b, row, 3)
        
        data_b_params_layout.addLayout(param_grid)
        
        # Apply button for Data B params
        apply_b_params_btn = QPushButton("Apply Data B Parameters")
        apply_b_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #e53935;
                color: white;
                padding: 6px;
                font-weight: bold;
                font-size: 9px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c62828;
            }
        """)
        apply_b_params_btn.clicked.connect(self.apply_data_b_params)
        data_b_params_layout.addWidget(apply_b_params_btn)
        
        self.data_b_params_group.setLayout(data_b_params_layout)
        layout.addWidget(self.data_b_params_group)
        
        # Scan Selection Group
        scan_group = QGroupBox("Scan Selection")
        scan_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #424242;
            }
        """)
        scan_layout = QVBoxLayout()
        scan_layout.setSpacing(8)
        
        # Mode selection - using QRadioButton for mutual exclusion
        mode_layout = QGridLayout()
        mode_layout.setSpacing(8)
        
        self.scan_mode_all = QRadioButton("All Scans (Default)")
        self.scan_mode_all.setChecked(True)
        self.scan_mode_all.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.scan_mode_all.toggled.connect(self.on_scan_mode_changed)
        mode_layout.addWidget(self.scan_mode_all, 0, 0, 1, 3)
        
        self.scan_mode_single = QRadioButton("Single Scan:")
        self.scan_mode_single.setStyleSheet("font-size: 10px;")
        self.scan_mode_single.toggled.connect(self.on_scan_mode_changed)
        mode_layout.addWidget(self.scan_mode_single, 1, 0)
        
        self.scan_single_num = QSpinBox()
        self.scan_single_num.setMinimum(0)
        self.scan_single_num.setMaximum(0)
        self.scan_single_num.setEnabled(False)
        self.scan_single_num.valueChanged.connect(self.on_scan_value_changed)
        mode_layout.addWidget(self.scan_single_num, 1, 1, 1, 2)
        
        self.scan_mode_range = QRadioButton("Scan Range:")
        self.scan_mode_range.setStyleSheet("font-size: 10px;")
        self.scan_mode_range.toggled.connect(self.on_scan_mode_changed)
        mode_layout.addWidget(self.scan_mode_range, 2, 0)
        
        range_layout = QHBoxLayout()
        range_layout.setSpacing(5)
        
        self.scan_range_start = QSpinBox()
        self.scan_range_start.setMinimum(0)
        self.scan_range_start.setMaximum(0)
        self.scan_range_start.setEnabled(False)
        self.scan_range_start.valueChanged.connect(self.on_scan_value_changed)
        range_layout.addWidget(self.scan_range_start)
        
        range_layout.addWidget(QLabel("to"))
        
        self.scan_range_end = QSpinBox()
        self.scan_range_end.setMinimum(0)
        self.scan_range_end.setMaximum(0)
        self.scan_range_end.setEnabled(False)
        self.scan_range_end.valueChanged.connect(self.on_scan_value_changed)
        range_layout.addWidget(self.scan_range_end)
        
        mode_layout.addLayout(range_layout, 2, 1, 1, 2)
        
        scan_layout.addLayout(mode_layout)
        
        # Apply button
        apply_scan_btn_layout = QHBoxLayout()
        apply_scan_btn_layout.addStretch()
        
        self.apply_scan_btn = QPushButton("Apply Selection")
        self.apply_scan_btn.setEnabled(False)
        self.apply_scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #90caf9;
                color: #0d47a1;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #64b5f6;
            }
            QPushButton:pressed {
                background-color: #42a5f5;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        self.apply_scan_btn.clicked.connect(self.reload_selected_scans)
        apply_scan_btn_layout.addWidget(self.apply_scan_btn)
        apply_scan_btn_layout.addStretch()
        
        scan_layout.addLayout(apply_scan_btn_layout)
        
        # Info label
        self.scan_selection_info = QLabel("Load data to enable scan selection")
        self.scan_selection_info.setWordWrap(True)
        self.scan_selection_info.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                color: #616161;
                font-size: 9px;
            }
        """)
        scan_layout.addWidget(self.scan_selection_info)
        
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)
        
        # Parameter tabs
        param_tabs = QTabWidget()
        param_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f5f5f5;
                color: #424242;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
                font-size: 10px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #5c7a99;
                border: 2px solid #e0e0e0;
                border-bottom-color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #eeeeee;
            }
        """)
        
        # Tab 1: Time Domain Processing
        time_tab = self.create_time_tab()
        param_tabs.addTab(time_tab, "Time Domain Processing")
        
        # Tab 2: Frequency Domain Processing
        freq_tab = self.create_freq_tab()
        param_tabs.addTab(freq_tab, "Frequency Domain Processing")
        
        layout.addWidget(param_tabs)
        
        # Visualization Settings (Global)
        view_group = QGroupBox("Visualization Settings")
        view_group.setStyleSheet(self.get_groupbox_style("#00897b"))
        view_layout = QVBoxLayout()
        
        # Display Mode
        display_layout = QHBoxLayout()
        self.view_real = QRadioButton("Real")
        self.view_real.setChecked(True)
        self.view_real.toggled.connect(self.update_plot_view)
        display_layout.addWidget(self.view_real)
        
        self.view_imag = QRadioButton("Imag")
        self.view_imag.toggled.connect(self.update_plot_view)
        display_layout.addWidget(self.view_imag)
        
        self.view_mag = QRadioButton("Mag")
        self.view_mag.toggled.connect(self.update_plot_view)
        display_layout.addWidget(self.view_mag)
        
        self.show_absolute = QCheckBox("Abs")
        self.show_absolute.setStyleSheet("font-weight: bold; color: #d32f2f;")
        self.show_absolute.stateChanged.connect(self.update_plot_view)
        display_layout.addWidget(self.show_absolute)
        
        self.show_baseline = QCheckBox("Show Baseline")
        self.show_baseline.setStyleSheet("font-weight: bold; color: #795548;")
        self.show_baseline.stateChanged.connect(self.update_plot_view)
        display_layout.addWidget(self.show_baseline)
        
        view_layout.addLayout(display_layout)
        
        # Frequency Range
        range_layout = QGridLayout()
        range_layout.setSpacing(5)
        
        range_layout.addWidget(QLabel("Low Freq View (Hz):"), 0, 0)
        self.freq_low_min = QDoubleSpinBox()
        self.freq_low_min.setRange(0, 1000)
        self.freq_low_min.setValue(0)
        range_layout.addWidget(self.freq_low_min, 0, 1)
        self.freq_low_max = QDoubleSpinBox()
        self.freq_low_max.setRange(0, 1000)
        self.freq_low_max.setValue(30)
        range_layout.addWidget(self.freq_low_max, 0, 2)
        
        range_layout.addWidget(QLabel("High Freq View (Hz):"), 1, 0)
        self.freq_high_min = QDoubleSpinBox()
        self.freq_high_min.setRange(0, 1000)
        self.freq_high_min.setValue(100)
        range_layout.addWidget(self.freq_high_min, 1, 1)
        self.freq_high_max = QDoubleSpinBox()
        self.freq_high_max.setRange(0, 1000)
        self.freq_high_max.setValue(275)
        range_layout.addWidget(self.freq_high_max, 1, 2)
        
        apply_range_btn = QPushButton("Update View Range")
        apply_range_btn.setStyleSheet("""
            QPushButton {
                background-color: #80cbc4;
                color: #004d40;
                padding: 6px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4db6ac;
            }
            QPushButton:pressed {
                background-color: #26a69a;
            }
        """)
        apply_range_btn.clicked.connect(self.plot_results)
        range_layout.addWidget(apply_range_btn, 2, 0, 1, 3)
        
        view_layout.addLayout(range_layout)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Process button
        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #a5d6a7;
                color: #1b5e20;
                font-size: 13px;
                font-weight: bold;
                padding: 14px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #81c784;
            }
            QPushButton:pressed {
                background-color: #66bb6a;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                text-align: center;
                background-color: #f5f5f5;
                height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #a5d6a7, stop:1 #81c784);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #757575;
                font-size: 10px;
                font-style: italic;
                padding: 4px;
            }
        """)
        layout.addWidget(self.progress_label)
        
        # Save/Export buttons
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(8)
        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #d7ccc8;
                color: #5d4037;
                padding: 10px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #bcaaa4;
            }
            QPushButton:pressed {
                background-color: #a1887f;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.save_params_btn.setEnabled(False)
        btn_layout2.addWidget(self.save_params_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #d1c4e9;
                color: #4527a0;
                padding: 10px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #b39ddb;
            }
            QPushButton:pressed {
                background-color: #9575cd;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        btn_layout2.addWidget(self.export_btn)
        layout.addLayout(btn_layout2)
        
        # Results and Metrics
        results_group = QGroupBox("Results and Metrics")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #424242;
            }
        """)
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)
        
        # Unified metrics table (for both single and comparison mode)
        from PySide6.QtWidgets import QTableWidget, QHeaderView, QTableWidgetItem
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)  # Will hide column 2 in single mode
        self.metrics_table.setRowCount(5)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value", ""])
        self.metrics_table.setVerticalHeaderLabels(["", "", "", "", ""])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setMaximumHeight(180)
        self.metrics_table.setColumnHidden(2, True)  # Hide Data B column initially
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                gridline-color: #e0e0e0;
                font-size: 10px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #e0e0e0;
                font-weight: bold;
                font-size: 10px;
                color: #424242;
            }
        """)
        
        # Initialize table items (will be updated in calculate_metrics)
        for row in range(5):
            for col in range(3):
                item = QTableWidgetItem("--")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                self.metrics_table.setItem(row, col, item)
        
        results_layout.addWidget(self.metrics_table)
        
        # Detailed info
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(110)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9px;
                color: #424242;
                padding: 8px;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return panel
    
    def create_time_tab(self):
        """Create time domain processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Savgol filter
        savgol_group = QGroupBox("Signal Filtering (Savitzky-Golay)")
        savgol_group.setStyleSheet(self.get_groupbox_style("#1976d2"))
        savgol_layout = QGridLayout()
        savgol_layout.setSpacing(10)
        savgol_layout.setContentsMargins(12, 15, 12, 12)
        
        row = 0
        conv_label_title = QLabel("Convolution Points:")
        conv_label_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        savgol_layout.addWidget(conv_label_title, row, 0)
        self.conv_slider = QSlider(Qt.Horizontal)
        self.conv_slider.setRange(2, 12000)
        self.conv_slider.setValue(300)
        self.conv_slider.setStyleSheet(self.get_slider_style("#5c7a99", "#4a6580"))
        self.conv_slider.valueChanged.connect(self.on_conv_changed)
        savgol_layout.addWidget(self.conv_slider, row, 1)
        self.conv_spinbox = QSpinBox()
        self.conv_spinbox.setRange(2, 12000)
        self.conv_spinbox.setValue(300)
        self.conv_spinbox.setMinimumWidth(80)
        self.conv_spinbox.setStyleSheet(self.get_spinbox_style("#5c7a99", "#4a6580", "#3a5166"))
        self.conv_spinbox.valueChanged.connect(self.on_conv_spinbox_changed)
        savgol_layout.addWidget(self.conv_spinbox, row, 2)
        row += 1
        
        poly_label_title = QLabel("Polynomial Order:")
        poly_label_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        savgol_layout.addWidget(poly_label_title, row, 0)
        self.poly_slider = QSlider(Qt.Horizontal)
        self.poly_slider.setRange(1, 20)
        self.poly_slider.setValue(2)
        self.poly_slider.setStyleSheet(self.get_slider_style("#5c7a99", "#4a6580"))
        self.poly_slider.valueChanged.connect(self.on_poly_changed)
        savgol_layout.addWidget(self.poly_slider, row, 1)
        self.poly_spinbox = QSpinBox()
        self.poly_spinbox.setRange(1, 20)
        self.poly_spinbox.setValue(2)
        self.poly_spinbox.setMinimumWidth(80)
        self.poly_spinbox.setStyleSheet(self.get_spinbox_style("#5c7a99", "#4a6580", "#3a5166"))
        self.poly_spinbox.valueChanged.connect(self.on_poly_spinbox_changed)
        savgol_layout.addWidget(self.poly_spinbox, row, 2)
        row += 1
        
        savgol_group.setLayout(savgol_layout)
        layout.addWidget(savgol_group)
        
        # 1.5 SVD Denoising
        svd_group = QGroupBox("Advanced Denoising (SVD/Cadzow)")
        svd_group.setStyleSheet(self.get_groupbox_style("#7b1fa2"))
        svd_layout = QGridLayout()
        svd_layout.setSpacing(10)
        svd_layout.setContentsMargins(12, 15, 12, 12)
        
        # Enable Checkbox
        self.enable_svd = QCheckBox("Enable SVD Denoising")
        self.enable_svd.setToolTip("Enable Singular Value Decomposition (Cadzow) denoising.\nWarning: Computationally intensive.")
        self.enable_svd.stateChanged.connect(self.on_param_changed)
        svd_layout.addWidget(self.enable_svd, 0, 0, 1, 3)
        
        # Rank Slider
        svd_layout.addWidget(QLabel("Rank (Components):"), 1, 0)
        self.svd_rank_slider = QSlider(Qt.Horizontal)
        self.svd_rank_slider.setRange(1, 50)
        self.svd_rank_slider.setValue(5)
        self.svd_rank_slider.setStyleSheet(self.get_slider_style("#ce93d8", "#ba68c8"))
        self.svd_rank_slider.valueChanged.connect(self.on_svd_rank_slider_changed)
        svd_layout.addWidget(self.svd_rank_slider, 1, 1)
        
        self.svd_rank_spin = QSpinBox()
        self.svd_rank_spin.setRange(1, 50)
        self.svd_rank_spin.setValue(5)
        self.svd_rank_spin.setMinimumWidth(80)
        self.svd_rank_spin.setStyleSheet(self.get_spinbox_style("#ce93d8", "#ba68c8", "#ab47bc"))
        self.svd_rank_spin.valueChanged.connect(self.on_svd_rank_spin_changed)
        svd_layout.addWidget(self.svd_rank_spin, 1, 2)
        
        svd_group.setLayout(svd_layout)
        layout.addWidget(svd_group)
        
        # 2. Backward Linear Prediction (Moved from old Tab 2)
        recon_group = QGroupBox("Signal Recovery (Backward LP)")
        recon_group.setStyleSheet(self.get_groupbox_style("#d32f2f"))
        recon_layout = QGridLayout()
        recon_layout.setSpacing(10)
        recon_layout.setContentsMargins(12, 15, 12, 12)
        
        # Row 0: Checkboxes
        checkbox_layout = QHBoxLayout()
        self.enable_recon = QCheckBox("Enable Backward LP")
        self.enable_recon.stateChanged.connect(self.on_param_changed)
        checkbox_layout.addWidget(self.enable_recon)
        
        self.sync_recon_checkbox = QCheckBox("Sync with Truncation Start")
        self.sync_recon_checkbox.toggled.connect(self.on_sync_recon_toggled)
        checkbox_layout.addWidget(self.sync_recon_checkbox)
        
        self.recon_train_auto = QCheckBox("Auto Training (4x Order)")
        self.recon_train_auto.setChecked(True)
        self.recon_train_auto.toggled.connect(self.on_recon_train_auto_toggled)
        checkbox_layout.addWidget(self.recon_train_auto)
        
        checkbox_layout.addStretch()
        
        recon_layout.addLayout(checkbox_layout, 0, 0, 1, 3)
        
        # Row 1: Points
        points_label = QLabel("Backward Points:")
        points_label.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        recon_layout.addWidget(points_label, 1, 0)
        
        self.recon_slider = QSlider(Qt.Horizontal)
        self.recon_slider.setRange(0, 3000)
        self.recon_slider.setValue(0)
        self.recon_slider.setStyleSheet(self.get_slider_style("#d32f2f", "#b71c1c"))
        self.recon_slider.valueChanged.connect(self.on_recon_slider_changed)
        recon_layout.addWidget(self.recon_slider, 1, 1)

        self.recon_points = QSpinBox()
        self.recon_points.setRange(0, 3000)
        self.recon_points.setValue(0)
        self.recon_points.setMinimumWidth(80)
        self.recon_points.setStyleSheet(self.get_spinbox_style("#d32f2f", "#b71c1c", "#9a0007"))
        self.recon_points.setToolTip("Number of points to predict backwards.\nExtends the FID to recover lost initial points (dead time).")
        self.recon_points.valueChanged.connect(self.on_recon_spinbox_changed)
        recon_layout.addWidget(self.recon_points, 1, 2)
        
        # Row 2: Order
        order_label = QLabel("LP Order:")
        order_label.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        recon_layout.addWidget(order_label, 2, 0)
        
        self.recon_order_slider = QSlider(Qt.Horizontal)
        self.recon_order_slider.setToolTip("Order of the Linear Prediction model.\nHigher order = models more peaks, but less stable.")
        self.recon_order_slider.setRange(1, 512)
        self.recon_order_slider.setValue(64)
        self.recon_order_slider.setStyleSheet(self.get_slider_style("#d32f2f", "#b71c1c"))
        self.recon_order_slider.valueChanged.connect(self.on_recon_order_slider_changed)
        recon_layout.addWidget(self.recon_order_slider, 2, 1)

        self.recon_order = QSpinBox()
        self.recon_order.setRange(1, 512)
        self.recon_order.setValue(64)
        self.recon_order.setMinimumWidth(80)
        self.recon_order.setToolTip("Linear Prediction Order (Number of coefficients)")
        self.recon_order.setStyleSheet(self.get_spinbox_style("#d32f2f", "#b71c1c", "#9a0007"))
        self.recon_order.valueChanged.connect(self.on_recon_order_spinbox_changed)
        recon_layout.addWidget(self.recon_order, 2, 2)
        
        # Row 3: Training Points
        train_label = QLabel("Training Points:")
        train_label.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        recon_layout.addWidget(train_label, 3, 0)
        
        self.recon_train_slider = QSlider(Qt.Horizontal)
        self.recon_train_slider.setRange(1, 5120)
        self.recon_train_slider.setValue(256)
        self.recon_train_slider.setEnabled(False)
        self.recon_train_slider.setStyleSheet(self.get_slider_style("#d32f2f", "#b71c1c"))
        self.recon_train_slider.valueChanged.connect(self.on_recon_train_slider_changed)
        recon_layout.addWidget(self.recon_train_slider, 3, 1)
        
        self.recon_train_points = QSpinBox()
        self.recon_train_points.setRange(1, 10000)
        self.recon_train_points.setValue(256) # 4 * 64
        self.recon_train_points.setEnabled(False)
        self.recon_train_points.setMinimumWidth(80)
        self.recon_train_points.setStyleSheet(self.get_spinbox_style("#d32f2f", "#b71c1c", "#9a0007"))
        self.recon_train_points.valueChanged.connect(self.on_recon_train_spinbox_changed)
        recon_layout.addWidget(self.recon_train_points, 3, 2)
        
        recon_group.setLayout(recon_layout)
        layout.addWidget(recon_group)
        
        # 3. Time Domain Operations (Truncation & Apodization)
        time_group = QGroupBox("Time Domain Operations")
        time_group.setStyleSheet(self.get_groupbox_style("#43a047"))
        time_layout = QGridLayout()
        time_layout.setSpacing(10)
        time_layout.setContentsMargins(12, 15, 12, 12)
        
        # Truncation Section
        row = 0
        trunc_header = QLabel("Truncation")
        trunc_header.setStyleSheet("font-size: 10px; font-weight: bold; color: #2e7d32; text-decoration: underline;")
        time_layout.addWidget(trunc_header, row, 0, 1, 3)
        row += 1

        trunc_start_title = QLabel("Start Points:")
        trunc_start_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        time_layout.addWidget(trunc_start_title, row, 0)
        self.trunc_start_slider = QSlider(Qt.Horizontal)
        self.trunc_start_slider.setRange(0, 3000)
        self.trunc_start_slider.setValue(10)
        self.trunc_start_slider.setStyleSheet(self.get_slider_style("#6b9b7c", "#568266"))
        self.trunc_start_slider.valueChanged.connect(self.on_trunc_start_changed)
        time_layout.addWidget(self.trunc_start_slider, row, 1)
        self.trunc_start_spinbox = QSpinBox()
        self.trunc_start_spinbox.setRange(0, 3000)
        self.trunc_start_spinbox.setValue(10)
        self.trunc_start_spinbox.setMinimumWidth(80)
        self.trunc_start_spinbox.setStyleSheet(self.get_spinbox_style("#6b9b7c", "#568266", "#446952"))
        self.trunc_start_spinbox.valueChanged.connect(self.on_trunc_start_spinbox_changed)
        time_layout.addWidget(self.trunc_start_spinbox, row, 2)
        row += 1
        
        trunc_end_title = QLabel("End Points:")
        trunc_end_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        time_layout.addWidget(trunc_end_title, row, 0)
        self.trunc_end_slider = QSlider(Qt.Horizontal)
        self.trunc_end_slider.setRange(0, 60000)
        self.trunc_end_slider.setValue(10)
        self.trunc_end_slider.setStyleSheet(self.get_slider_style("#6b9b7c", "#568266"))
        self.trunc_end_slider.valueChanged.connect(self.on_trunc_end_changed)
        time_layout.addWidget(self.trunc_end_slider, row, 1)
        self.trunc_end_spinbox = QSpinBox()
        self.trunc_end_spinbox.setRange(0, 60000)
        self.trunc_end_spinbox.setValue(10)
        self.trunc_end_spinbox.setMinimumWidth(80)
        self.trunc_end_spinbox.setStyleSheet(self.get_spinbox_style("#6b9b7c", "#568266", "#446952"))
        self.trunc_end_spinbox.valueChanged.connect(self.on_trunc_end_spinbox_changed)
        time_layout.addWidget(self.trunc_end_spinbox, row, 2)
        row += 1
        
        # Separator
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #e0e0e0; margin: 5px 0;")
        time_layout.addWidget(line, row, 0, 1, 3)
        row += 1

        # Apodization Section
        apod_header = QLabel("Apodization")
        apod_header.setStyleSheet("font-size: 10px; font-weight: bold; color: #ef6c00; text-decoration: underline;")
        time_layout.addWidget(apod_header, row, 0, 1, 3)
        row += 1

        apod_title = QLabel("T2* Factor:")
        apod_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        time_layout.addWidget(apod_title, row, 0)
        self.apod_slider = QSlider(Qt.Horizontal)
        self.apod_slider.setRange(-200, 200)
        self.apod_slider.setValue(0)
        self.apod_slider.setStyleSheet(self.get_slider_style("#b8865f", "#9d714d"))
        self.apod_slider.valueChanged.connect(self.on_apod_changed)
        time_layout.addWidget(self.apod_slider, row, 1)
        self.apod_spinbox = QDoubleSpinBox()
        self.apod_spinbox.setRange(-2.00, 2.00)
        self.apod_spinbox.setValue(0.00)
        self.apod_spinbox.setSingleStep(0.01)
        self.apod_spinbox.setDecimals(2)
        self.apod_spinbox.setMinimumWidth(80)
        self.apod_spinbox.setStyleSheet(self.get_spinbox_style("#b8865f", "#9d714d", "#825c3d"))
        self.apod_spinbox.valueChanged.connect(self.on_apod_spinbox_changed)
        time_layout.addWidget(self.apod_spinbox, row, 2)
        row += 1
        
        hanning_title = QLabel("Hanning Window:")
        hanning_title.setStyleSheet("font-size: 9px; font-weight: bold;")
        time_layout.addWidget(hanning_title, row, 0)
        self.use_hanning = QCheckBox("Apply Hanning")
        self.use_hanning.setStyleSheet("""
            QCheckBox {
                font-size: 10px;
                color: #424242;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #bdbdbd;
                border-radius: 3px;
                background: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #f57c00;
                border-radius: 3px;
                background: #f57c00;
            }
        """)
        self.use_hanning.stateChanged.connect(self.on_param_changed)
        time_layout.addWidget(self.use_hanning, row, 1)
        row += 1
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        layout.addStretch()
        return tab
    
    def create_freq_tab(self):
        """Create frequency domain processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Transformation Settings (Zero Filling + FFT)
        trans_group = QGroupBox("Transformation Settings")
        trans_group.setStyleSheet(self.get_groupbox_style("#5e35b1"))
        trans_layout = QGridLayout()
        trans_layout.setSpacing(10)
        trans_layout.setContentsMargins(12, 15, 12, 12)
        
        # Zero Filling
        row = 0
        zf_title = QLabel("Zero Fill Factor:")
        zf_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        trans_layout.addWidget(zf_title, row, 0)
        self.zf_slider = QSlider(Qt.Horizontal)
        self.zf_slider.setRange(0, 1000)
        self.zf_slider.setValue(0)
        self.zf_slider.setStyleSheet(self.get_slider_style("#7d6b9d", "#685983"))
        self.zf_slider.valueChanged.connect(self.on_zf_changed)
        trans_layout.addWidget(self.zf_slider, row, 1)
        self.zf_spinbox = QDoubleSpinBox()
        self.zf_spinbox.setRange(0.00, 10.00)
        self.zf_spinbox.setValue(0.00)
        self.zf_spinbox.setSingleStep(0.01)
        self.zf_spinbox.setDecimals(2)
        self.zf_spinbox.setMinimumWidth(80)
        self.zf_spinbox.setStyleSheet(self.get_spinbox_style("#7d6b9d", "#685983", "#544769"))
        self.zf_spinbox.valueChanged.connect(self.on_zf_spinbox_changed)
        trans_layout.addWidget(self.zf_spinbox, row, 2)
        row += 1
        
        zf_hint = QLabel("0 = No filling, 2.7 = 2.7x data length")
        zf_hint.setStyleSheet("font-size: 9px; color: #757575; font-style: italic;")
        trans_layout.addWidget(zf_hint, row, 0, 1, 3)
        row += 1
        
        # FFT Info
        fft_info = QLabel("FFT is automatically applied after time domain processing.")
        fft_info.setStyleSheet("color: #757575; font-size: 10px; margin-top: 5px;")
        trans_layout.addWidget(fft_info, row, 0, 1, 3)
        
        trans_group.setLayout(trans_layout)
        layout.addWidget(trans_group)
        
        # 2. Spectral Correction (Phase + Baseline)
        # We'll put them in a horizontal splitter or just vertical layout
        # Vertical is safer for width
        
        # Phase Correction
        phase_group = QGroupBox("Phase Correction")
        phase_group.setStyleSheet(self.get_groupbox_style("#1976d2"))
        phase_layout = QGridLayout()
        
        # Phase 0
        phase_layout.addWidget(QLabel("Phase 0:"), 0, 0)
        self.phase0_slider = QSlider(Qt.Horizontal)
        self.phase0_slider.setRange(-180, 180)
        self.phase0_slider.setValue(0)
        self.phase0_slider.setStyleSheet(self.get_slider_style("#1976d2", "#1565c0"))
        self.phase0_slider.valueChanged.connect(self.on_phase_changed) # Special handler for fast update
        phase_layout.addWidget(self.phase0_slider, 0, 1)
        self.phase0_spin = QDoubleSpinBox()
        self.phase0_spin.setRange(-180, 180)
        self.phase0_spin.setSingleStep(1)
        self.phase0_spin.setStyleSheet(self.get_spinbox_style("#1976d2", "#1565c0", "#0d47a1"))
        self.phase0_spin.valueChanged.connect(self.on_phase_spin_changed)
        phase_layout.addWidget(self.phase0_spin, 0, 2)
        
        # Phase 1
        phase_layout.addWidget(QLabel("Phase 1:"), 1, 0)
        self.phase1_slider = QSlider(Qt.Horizontal)
        self.phase1_slider.setRange(-10000, 10000)
        self.phase1_slider.setValue(0)
        self.phase1_slider.setStyleSheet(self.get_slider_style("#1976d2", "#1565c0"))
        self.phase1_slider.valueChanged.connect(self.on_phase_changed)
        phase_layout.addWidget(self.phase1_slider, 1, 1)
        self.phase1_spin = QDoubleSpinBox()
        self.phase1_spin.setRange(-10000, 10000)
        self.phase1_spin.setSingleStep(10)
        self.phase1_spin.setStyleSheet(self.get_spinbox_style("#1976d2", "#1565c0", "#0d47a1"))
        self.phase1_spin.valueChanged.connect(self.on_phase_spin_changed)
        phase_layout.addWidget(self.phase1_spin, 1, 2)
        
        # Auto Phase
        self.auto_phase_btn = QPushButton("Auto Phase")
        self.auto_phase_btn.setStyleSheet("""
            QPushButton {
                background-color: #90caf9;
                color: #0d47a1;
                padding: 6px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #64b5f6;
            }
            QPushButton:pressed {
                background-color: #42a5f5;
            }
        """)
        self.auto_phase_btn.clicked.connect(self.run_auto_phase)
        phase_layout.addWidget(self.auto_phase_btn, 2, 0, 1, 3)
        
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)

        # Baseline Correction
        baseline_group = QGroupBox("Baseline Correction")
        baseline_group.setStyleSheet(self.get_groupbox_style("#a68d42"))
        baseline_layout = QGridLayout()
        
        baseline_layout.addWidget(QLabel("Method:"), 0, 0)
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["None", "Polynomial", "AirPLS"])
        self.baseline_method.currentTextChanged.connect(self.on_baseline_method_changed)
        baseline_layout.addWidget(self.baseline_method, 0, 1, 1, 2)
        
        # Polynomial Order
        self.lbl_poly = QLabel("Fitting Order:")
        baseline_layout.addWidget(self.lbl_poly, 1, 0)
        
        self.baseline_poly_slider = QSlider(Qt.Horizontal)
        self.baseline_poly_slider.setRange(0, 10)
        self.baseline_poly_slider.setValue(1)
        self.baseline_poly_slider.setStyleSheet(self.get_slider_style("#a68d42", "#a68d42"))
        self.baseline_poly_slider.valueChanged.connect(self.on_baseline_poly_slider_changed)
        baseline_layout.addWidget(self.baseline_poly_slider, 1, 1)
        
        self.baseline_poly_order = QSpinBox()
        self.baseline_poly_order.setRange(0, 10)
        self.baseline_poly_order.setValue(1)
        self.baseline_poly_order.setMinimumWidth(80)
        self.baseline_poly_order.setStyleSheet(self.get_spinbox_style("#a68d42", "#a68d42", "#a68d42"))
        self.baseline_poly_order.valueChanged.connect(self.on_baseline_poly_spinbox_changed)
        baseline_layout.addWidget(self.baseline_poly_order, 1, 2)
        
        # AirPLS Lambda
        self.lbl_lambda = QLabel("Lambda:")
        baseline_layout.addWidget(self.lbl_lambda, 2, 0)
        
        self.baseline_lambda_slider = QSlider(Qt.Horizontal)
        self.baseline_lambda_slider.setRange(10, 100000)
        self.baseline_lambda_slider.setValue(100)
        self.baseline_lambda_slider.setSingleStep(100)
        self.baseline_lambda_slider.setStyleSheet(self.get_slider_style("#a68d42", "#a68d42"))
        self.baseline_lambda_slider.setToolTip("Smoothness parameter for AirPLS.\nLarger value = smoother/stiffer baseline (10^4 - 10^5).\nSmaller value = more flexible baseline (10 - 100).")
        self.baseline_lambda_slider.valueChanged.connect(self.on_baseline_lambda_slider_changed)
        baseline_layout.addWidget(self.baseline_lambda_slider, 2, 1)
        
        self.baseline_lambda = QSpinBox()
        self.baseline_lambda.setRange(10, 100000)
        self.baseline_lambda.setValue(100)
        self.baseline_lambda.setSingleStep(100)
        self.baseline_lambda.setMinimumWidth(80)
        self.baseline_lambda.setStyleSheet(self.get_spinbox_style("#a68d42", "#a68d42", "#a68d42"))
        self.baseline_lambda.setToolTip("Smoothness parameter for AirPLS.\nLarger value = smoother/stiffer baseline (10^4 - 10^5).\nSmaller value = more flexible baseline (10 - 100).")
        self.baseline_lambda.valueChanged.connect(self.on_baseline_lambda_spinbox_changed)
        baseline_layout.addWidget(self.baseline_lambda, 2, 2)
        
        # Initial visibility
        self.lbl_lambda.setVisible(False)
        self.baseline_lambda_slider.setVisible(False)
        self.baseline_lambda.setVisible(False)
        self.baseline_lambda.setVisible(False)
        
        baseline_group.setLayout(baseline_layout)
        layout.addWidget(baseline_group)
        
        # 3. SNR Calculation Settings
        snr_group = QGroupBox("SNR Calculation Range")
        snr_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        snr_layout = QGridLayout()
        snr_layout.setSpacing(10)
        
        # Signal range
        signal_label = QLabel("Signal Range (Hz):")
        signal_label.setStyleSheet("font-size: 10px;")
        snr_layout.addWidget(signal_label, 0, 0)
        
        signal_range_layout = QHBoxLayout()
        signal_range_layout.setSpacing(5)
        
        self.signal_range_min = QDoubleSpinBox()
        self.signal_range_min.setRange(0, 100000)
        self.signal_range_min.setValue(110)
        self.signal_range_min.setDecimals(0)
        self.signal_range_min.setSingleStep(10)
        self.signal_range_min.setStyleSheet("font-size: 10px;")
        self.signal_range_min.valueChanged.connect(self.schedule_processing)
        signal_range_layout.addWidget(self.signal_range_min)
        
        signal_range_layout.addWidget(QLabel("to"))
        
        self.signal_range_max = QDoubleSpinBox()
        self.signal_range_max.setRange(0, 100000)
        self.signal_range_max.setValue(140)
        self.signal_range_max.setDecimals(0)
        self.signal_range_max.setSingleStep(10)
        self.signal_range_max.setStyleSheet("font-size: 10px;")
        self.signal_range_max.valueChanged.connect(self.schedule_processing)
        signal_range_layout.addWidget(self.signal_range_max)
        
        snr_layout.addLayout(signal_range_layout, 0, 1)
        
        # Noise range
        noise_label = QLabel("Noise Range (Hz):")
        noise_label.setStyleSheet("font-size: 10px;")
        snr_layout.addWidget(noise_label, 1, 0)
        
        noise_range_layout = QHBoxLayout()
        noise_range_layout.setSpacing(5)
        
        self.noise_range_min = QDoubleSpinBox()
        self.noise_range_min.setRange(0, 100000)
        self.noise_range_min.setValue(350)
        self.noise_range_min.setDecimals(0)
        self.noise_range_min.setSingleStep(10)
        self.noise_range_min.setStyleSheet("font-size: 10px;")
        self.noise_range_min.valueChanged.connect(self.schedule_processing)
        noise_range_layout.addWidget(self.noise_range_min)
        
        noise_range_layout.addWidget(QLabel("to"))
        
        self.noise_range_max = QDoubleSpinBox()
        self.noise_range_max.setRange(0, 100000)
        self.noise_range_max.setValue(400)
        self.noise_range_max.setDecimals(0)
        self.noise_range_max.setSingleStep(10)
        self.noise_range_max.setStyleSheet("font-size: 10px;")
        self.noise_range_max.valueChanged.connect(self.schedule_processing)
        noise_range_layout.addWidget(self.noise_range_max)
        
        snr_layout.addLayout(noise_range_layout, 1, 1)
        
        # Info label
        snr_info = QLabel("Adjust these ranges to match your sample's signal and noise regions")
        snr_info.setWordWrap(True)
        snr_info.setStyleSheet("font-size: 9px; color: #757575; font-style: italic;")
        snr_layout.addWidget(snr_info, 2, 0, 1, 2)
        
        snr_group.setLayout(snr_layout)
        layout.addWidget(snr_group)
        
        layout.addStretch()
        return tab
    
    def create_plot_panel(self):
        """Create plot panel with 3 subplots"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Vertical splitter for plots
        self.plot_splitter = QSplitter(Qt.Vertical)
        self.plot_splitter.setHandleWidth(6)
        self.plot_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                border: 1px solid #bdbdbd;
            }
            QSplitter::handle:hover {
                background-color: #1976d2;
            }
        """)
        
        # Time domain
        time_widget = QWidget()
        time_widget.setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 4px;")
        time_layout = QVBoxLayout(time_widget)
        time_layout.setContentsMargins(2, 2, 2, 2)
        time_layout.setSpacing(2)
        
        # Add maximize button for time domain
        time_header = QHBoxLayout()
        time_title = QLabel("Time Domain Signal")
        time_title.setStyleSheet("font-weight: bold; color: #424242; font-size: 11px;")
        time_header.addWidget(time_title)
        time_header.addStretch()
        self.time_maximize_btn = QPushButton("Maximize")
        self.time_maximize_btn.setStyleSheet("""
            QPushButton {
                background-color: #90caf9;
                color: #0d47a1;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #64b5f6;
            }
        """)
        self.time_maximize_btn.clicked.connect(lambda: self.maximize_plot('time'))
        time_header.addWidget(self.time_maximize_btn)
        time_layout.addLayout(time_header)
        
        self.time_canvas = MplCanvas(self, width=10, height=3, dpi=100)
        self.time_toolbar = NavigationToolbar(self.time_canvas, self)
        time_layout.addWidget(self.time_toolbar)
        time_layout.addWidget(self.time_canvas)
        
        # Frequency domain (low freq)
        freq1_widget = QWidget()
        freq1_widget.setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 4px;")
        freq1_layout = QVBoxLayout(freq1_widget)
        freq1_layout.setContentsMargins(2, 2, 2, 2)
        freq1_layout.setSpacing(2)
        
        # Add maximize button for freq1
        freq1_header = QHBoxLayout()
        freq1_title = QLabel("Low Frequency Spectrum")
        freq1_title.setStyleSheet("font-weight: bold; color: #424242; font-size: 11px;")
        freq1_header.addWidget(freq1_title)
        freq1_header.addStretch()
        self.freq1_maximize_btn = QPushButton("Maximize")
        self.freq1_maximize_btn.setStyleSheet("""
            QPushButton {
                background-color: #90caf9;
                color: #0d47a1;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #64b5f6;
            }
        """)
        self.freq1_maximize_btn.clicked.connect(lambda: self.maximize_plot('freq1'))
        freq1_header.addWidget(self.freq1_maximize_btn)
        freq1_layout.addLayout(freq1_header)
        
        self.freq1_canvas = MplCanvas(self, width=10, height=3, dpi=100)
        self.freq1_toolbar = NavigationToolbar(self.freq1_canvas, self)
        freq1_layout.addWidget(self.freq1_toolbar)
        freq1_layout.addWidget(self.freq1_canvas)
        
        # Frequency domain (high freq)
        freq2_widget = QWidget()
        freq2_widget.setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 4px;")
        freq2_layout = QVBoxLayout(freq2_widget)
        freq2_layout.setContentsMargins(2, 2, 2, 2)
        freq2_layout.setSpacing(2)
        
        # Add maximize button for freq2
        freq2_header = QHBoxLayout()
        freq2_title = QLabel("High Frequency Spectrum")
        freq2_title.setStyleSheet("font-weight: bold; color: #424242; font-size: 11px;")
        freq2_header.addWidget(freq2_title)
        freq2_header.addStretch()
        self.freq2_maximize_btn = QPushButton("Maximize")
        self.freq2_maximize_btn.setStyleSheet("""
            QPushButton {
                background-color: #90caf9;
                color: #0d47a1;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #64b5f6;
            }
        """)
        self.freq2_maximize_btn.clicked.connect(lambda: self.maximize_plot('freq2'))
        freq2_header.addWidget(self.freq2_maximize_btn)
        freq2_layout.addLayout(freq2_header)
        
        self.freq2_canvas = MplCanvas(self, width=10, height=3, dpi=100)
        self.freq2_toolbar = NavigationToolbar(self.freq2_canvas, self)
        freq2_layout.addWidget(self.freq2_toolbar)
        freq2_layout.addWidget(self.freq2_canvas)
        
        # Add widgets to splitter
        self.plot_splitter.addWidget(time_widget)
        self.plot_splitter.addWidget(freq1_widget)
        self.plot_splitter.addWidget(freq2_widget)
        
        # Set equal initial sizes
        self.plot_splitter.setSizes([333, 333, 334])
        
        layout.addWidget(self.plot_splitter)
        
        return panel
    
    @Slot()
    def on_conv_changed(self, value):
        self.conv_spinbox.blockSignals(True)
        self.conv_spinbox.setValue(value)
        self.conv_spinbox.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_conv_spinbox_changed(self, value):
        self.conv_slider.blockSignals(True)
        self.conv_slider.setValue(int(value))
        self.conv_slider.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_poly_changed(self, value):
        self.poly_spinbox.blockSignals(True)
        self.poly_spinbox.setValue(value)
        self.poly_spinbox.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_poly_spinbox_changed(self, value):
        self.poly_slider.blockSignals(True)
        self.poly_slider.setValue(int(value))
        self.poly_slider.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_trunc_start_changed(self, value):
        self.trunc_start_spinbox.blockSignals(True)
        self.trunc_start_spinbox.setValue(value)
        self.trunc_start_spinbox.blockSignals(False)
        
        if hasattr(self, 'sync_recon_checkbox') and self.sync_recon_checkbox.isChecked():
            self.recon_slider.blockSignals(True)
            self.recon_points.blockSignals(True)
            self.recon_slider.setValue(value)
            self.recon_points.setValue(value)
            self.recon_slider.blockSignals(False)
            self.recon_points.blockSignals(False)
            
        self.schedule_processing()
    
    @Slot()
    def on_trunc_start_spinbox_changed(self, value):
        self.trunc_start_slider.blockSignals(True)
        self.trunc_start_slider.setValue(int(value))
        self.trunc_start_slider.blockSignals(False)
        
        if hasattr(self, 'sync_recon_checkbox') and self.sync_recon_checkbox.isChecked():
            self.recon_slider.blockSignals(True)
            self.recon_points.blockSignals(True)
            self.recon_slider.setValue(int(value))
            self.recon_points.setValue(int(value))
            self.recon_slider.blockSignals(False)
            self.recon_points.blockSignals(False)
            
        self.schedule_processing()
    
    @Slot()
    def on_trunc_end_changed(self, value):
        self.trunc_end_spinbox.blockSignals(True)
        self.trunc_end_spinbox.setValue(value)
        self.trunc_end_spinbox.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_trunc_end_spinbox_changed(self, value):
        self.trunc_end_slider.blockSignals(True)
        self.trunc_end_slider.setValue(int(value))
        self.trunc_end_slider.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_apod_changed(self, value):
        actual = value / 100.0
        self.apod_spinbox.blockSignals(True)
        self.apod_spinbox.setValue(actual)
        self.apod_spinbox.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_apod_spinbox_changed(self, value):
        self.apod_slider.blockSignals(True)
        self.apod_slider.setValue(int(value * 100))
        self.apod_slider.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_svd_rank_slider_changed(self, value):
        self.svd_rank_spin.blockSignals(True)
        self.svd_rank_spin.setValue(value)
        self.svd_rank_spin.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_svd_rank_spin_changed(self, value):
        self.svd_rank_slider.blockSignals(True)
        self.svd_rank_slider.setValue(value)
        self.svd_rank_slider.blockSignals(False)
        self.schedule_processing()

    
    @Slot()
    def on_zf_changed(self, value):
        actual = value / 100.0
        self.zf_spinbox.blockSignals(True)
        self.zf_spinbox.setValue(actual)
        self.zf_spinbox.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_zf_spinbox_changed(self, value):
        self.zf_slider.blockSignals(True)
        self.zf_slider.setValue(int(value * 100))
        self.zf_slider.blockSignals(False)
        self.schedule_processing()
    
    @Slot()
    def on_recon_slider_changed(self, value):
        self.recon_points.blockSignals(True)
        self.recon_points.setValue(value)
        self.recon_points.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_recon_spinbox_changed(self, value):
        self.recon_slider.blockSignals(True)
        self.recon_slider.setValue(int(value))
        self.recon_slider.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_recon_order_slider_changed(self, value):
        self.recon_order.blockSignals(True)
        self.recon_order.setValue(value)
        self.recon_order.blockSignals(False)
        
        if hasattr(self, 'recon_train_auto') and self.recon_train_auto.isChecked():
            val = value * 4
            self.recon_train_points.setValue(val)
            self.recon_train_slider.setValue(val)
            
        self.schedule_processing()

    @Slot()
    def on_recon_order_spinbox_changed(self, value):
        self.recon_order_slider.blockSignals(True)
        self.recon_order_slider.setValue(int(value))
        self.recon_order_slider.blockSignals(False)
        
        if hasattr(self, 'recon_train_auto') and self.recon_train_auto.isChecked():
            val = int(value) * 4
            self.recon_train_points.setValue(val)
            self.recon_train_slider.setValue(val)
            
        self.schedule_processing()

    @Slot(bool)
    def on_recon_train_auto_toggled(self, checked):
        self.recon_train_points.setEnabled(not checked)
        self.recon_train_slider.setEnabled(not checked)
        if checked:
            order = self.recon_order.value()
            val = order * 4
            self.recon_train_points.setValue(val)
            self.recon_train_slider.setValue(val)
        self.schedule_processing()

    @Slot()
    def on_recon_train_slider_changed(self, value):
        self.recon_train_points.blockSignals(True)
        self.recon_train_points.setValue(value)
        self.recon_train_points.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_recon_train_spinbox_changed(self, value):
        self.recon_train_slider.blockSignals(True)
        self.recon_train_slider.setValue(int(value))
        self.recon_train_slider.blockSignals(False)
        self.schedule_processing()

    @Slot(bool)
    def on_sync_recon_toggled(self, checked):
        self.recon_slider.setEnabled(not checked)
        self.recon_points.setEnabled(not checked)
        if checked:
            val = self.trunc_start_slider.value()
            self.recon_slider.setValue(val)
            self.recon_points.setValue(val)
            self.schedule_processing()

    @Slot()
    def on_baseline_poly_slider_changed(self, value):
        self.baseline_poly_order.blockSignals(True)
        self.baseline_poly_order.setValue(value)
        self.baseline_poly_order.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_baseline_poly_spinbox_changed(self, value):
        self.baseline_poly_slider.blockSignals(True)
        self.baseline_poly_slider.setValue(value)
        self.baseline_poly_slider.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_baseline_lambda_slider_changed(self, value):
        self.baseline_lambda.blockSignals(True)
        self.baseline_lambda.setValue(value)
        self.baseline_lambda.blockSignals(False)
        self.schedule_processing()

    @Slot()
    def on_baseline_lambda_spinbox_changed(self, value):
        self.baseline_lambda_slider.blockSignals(True)
        self.baseline_lambda_slider.setValue(value)
        self.baseline_lambda_slider.blockSignals(False)
        self.schedule_processing()

    @Slot(str)
    def on_baseline_method_changed(self, method):
        """Update visibility of baseline correction parameters"""
        is_poly = method == "Polynomial"
        is_airpls = method == "AirPLS"
        
        self.lbl_poly.setVisible(is_poly)
        self.baseline_poly_slider.setVisible(is_poly)
        self.baseline_poly_order.setVisible(is_poly)
        
        self.lbl_lambda.setVisible(is_airpls)
        self.baseline_lambda_slider.setVisible(is_airpls)
        self.baseline_lambda.setVisible(is_airpls)
        
        self.schedule_processing()
    
    @Slot()
    def on_param_changed(self):
        self.schedule_processing()
    
    @Slot()
    def on_scan_mode_changed(self, checked):
        """Handle scan range mode changes (QRadioButton auto-handles mutual exclusion)"""
        if not checked:
            # Ignore unchecked signal, only process checked signal
            return
        
        sender = self.sender()
        
        if sender == self.scan_mode_all:
            # All scans mode - disable single and range controls
            self.scan_single_num.setEnabled(False)
            self.scan_range_start.setEnabled(False)
            self.scan_range_end.setEnabled(False)
            self.update_scan_selection_info()
            self.reload_selected_scans()
            
        elif sender == self.scan_mode_single:
            # Single scan mode - enable single selector, disable range
            self.scan_single_num.setEnabled(True)
            self.scan_range_start.setEnabled(False)
            self.scan_range_end.setEnabled(False)
            self.update_scan_selection_info()
            
        elif sender == self.scan_mode_range:
            # Range mode - enable range selectors, disable single
            self.scan_single_num.setEnabled(False)
            self.scan_range_start.setEnabled(True)
            self.scan_range_end.setEnabled(True)
            self.update_scan_selection_info()
    
    def on_scan_value_changed(self):
        """Handle scan number spinbox value changes"""
        # Update info label only, don't reload until Apply button is clicked
        self.update_scan_selection_info()
    
    def update_scan_selection_info(self):
        """Update scan selection info label"""
        if not hasattr(self, 'scan_count') or self.scan_count == 0:
            self.scan_selection_info.setText("Load data to enable scan selection")
            return
        
        if self.scan_mode_all.isChecked():
            self.scan_selection_info.setText(
                f"<b>Mode:</b> All Scans<br>"
                f"<b>Selected:</b> {self.scan_count} scans<br>"
                f"<b>Range:</b> 0 to {self.scan_count - 1}"
            )
        elif self.scan_mode_single.isChecked():
            scan_num = self.scan_single_num.value()
            self.scan_selection_info.setText(
                f"<b>Mode:</b> Single Scan<br>"
                f"<b>Selected:</b> Scan #{scan_num}<br>"
                f"<b>Total Available:</b> {self.scan_count} scans"
            )
        elif self.scan_mode_range.isChecked():
            start = self.scan_range_start.value()
            end = self.scan_range_end.value()
            count = max(0, end - start + 1)
            self.scan_selection_info.setText(
                f"<b>Mode:</b> Scan Range<br>"
                f"<b>Selected:</b> {count} scans<br>"
                f"<b>Range:</b> {start} to {end}"
            )
    
    def get_selected_scan_indices(self):
        """Get list of selected scan indices based on current mode
        
        This is the interface for future good scans integration.
        Returns list of scan indices to process.
        
        Returns:
            list: List of scan indices to process
        """
        if not hasattr(self, 'scan_count') or self.scan_count == 0:
            return []
        
        if self.scan_mode_all.isChecked():
            # All scans
            return list(range(self.scan_count))
        
        elif self.scan_mode_single.isChecked():
            # Single scan
            return [self.scan_single_num.value()]
        
        elif self.scan_mode_range.isChecked():
            # Scan range
            start = self.scan_range_start.value()
            end = self.scan_range_end.value()
            if start <= end:
                return list(range(start, end + 1))
            else:
                return []
        
        return []
    
    def reload_selected_scans(self):
        """Reload data based on selected scan range"""
        if not hasattr(self, 'current_path') or not self.current_path:
            return
        
        if not HAS_NMRDUINO:
            print("Warning: Cannot reload scans, nmrduino_util not available")
            return
        
        # Prevent multiple simultaneous reloads
        if hasattr(self, '_is_reloading') and self._is_reloading:
            return
        
        try:
            self._is_reloading = True
            
            # Determine which scans to load based on selection mode
            if self.scan_mode_all.isChecked():
                # Load all scans
                scan_param = 0
                status_msg = f"Loading all {self.scan_count} scans..."
            elif self.scan_mode_single.isChecked():
                # Load single scan (1-indexed for nmrduino_dat_interp)
                scan_num = self.scan_single_num.value()
                scan_param = scan_num + 1
                status_msg = f"Loading scan #{scan_num}..."
            elif self.scan_mode_range.isChecked():
                # Load scan range (1-indexed)
                start = self.scan_range_start.value()
                end = self.scan_range_end.value()
                scan_param = [start + 1, end + 1]
                num_scans = end - start + 1
                status_msg = f"Loading scans {start} to {end} ({num_scans} scans)..."
            else:
                scan_param = 0
                status_msg = "Loading scans..."
            
            # Show loading message
            print(status_msg)
            QApplication.processEvents()  # Update UI
            
            # Reload data
            compiled = nmr_util.nmrduino_dat_interp(self.current_path, scan_param)
            self.halp = compiled[0]
            self.sampling_rate = compiled[1]
            self.acq_time = compiled[2]
            
            print(f"Loaded {len(self.halp)} data points")
            
            # Process with current parameters
            self.schedule_processing()
            
        except Exception as e:
            print(f"Error reloading scans: {e}")
            QMessageBox.warning(self, "Error", f"Failed to reload scans:\n{str(e)}")
        finally:
            self._is_reloading = False
    
    def schedule_processing(self):
        """Schedule processing with debounce"""
        if self.halp is not None:
            self.process_timer.start(300)  # 300ms debounce
    
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
                    np.save(os.path.join(folder, "scan_count.npy"), self.scan_count)
            else:
                # Manual loading
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp = np.load(compiled_path)
                    self.sampling_rate = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                    
                    # Load scan count if available, otherwise default to 1
                    scan_count_path = os.path.join(folder, "scan_count.npy")
                    if os.path.exists(scan_count_path):
                        self.scan_count = int(np.load(scan_count_path))
                    else:
                        # Try to count .dat files in folder
                        dat_files = [f for f in os.listdir(folder) if f.endswith('.dat')]
                        self.scan_count = len(dat_files) if dat_files else 1
                        print(f"Warning: scan_count.npy not found, detected {self.scan_count} scans from .dat files")
                else:
                    raise FileNotFoundError("No compiled data found. Please compile first or install nmrduino_util.")
            
            # Auto-load saved parameters if exist
            param_file = os.path.join(folder, "processing_params.json")
            param_loaded = False
            if os.path.exists(param_file):
                try:
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                    
                    # Update UI sliders silently
                    self.conv_slider.setValue(int(params.get('conv_points', 300)))
                    self.poly_slider.setValue(int(params.get('poly_order', 2)))
                    self.trunc_start_slider.setValue(int(params.get('trunc_start', 10)))
                    self.trunc_end_slider.setValue(int(params.get('trunc_end', 10)))
                    self.apod_slider.setValue(int(float(params.get('apod_t2star', 0.0)) * 100))
                    self.use_hanning.setChecked(bool(params.get('use_hanning', 0)))
                    self.zf_slider.setValue(int(float(params.get('zf_factor', 0.0)) * 100))
                    
                    # Update frequency display range sliders if present
                    if 'freq_low_min' in params:
                        self.freq_low_min.setValue(float(params['freq_low_min']))
                    if 'freq_low_max' in params:
                        self.freq_low_max.setValue(float(params['freq_low_max']))
                    if 'freq_high_min' in params:
                        self.freq_high_min.setValue(float(params['freq_high_min']))
                    if 'freq_high_max' in params:
                        self.freq_high_max.setValue(float(params['freq_high_max']))
                    
                    # Update ZULF params
                    if 'phase0' in params:
                        self.phase0_spin.setValue(float(params['phase0']))
                    if 'phase1' in params:
                        self.phase1_spin.setValue(float(params['phase1']))
                    if 'enable_recon' in params:
                        self.enable_recon.setChecked(bool(params['enable_recon']))
                    if 'recon_points' in params:
                        self.recon_points.setValue(int(params['recon_points']))
                    
                    self.params = params
                    param_loaded = True
                except Exception as e:
                    print(f"Warning: Failed to auto-load parameters: {e}")
            
            param_status = "<br><b>Parameters:</b> Auto-loaded" if param_loaded else ""
            self.data_info.setText(
                f"<b>Loaded:</b> {os.path.basename(folder)}<br>"
                f"<b>Points:</b> {len(self.halp)}<br>"
                f"<b>Sampling:</b> {self.sampling_rate:.1f} Hz<br>"
                f"<b>Acq Time:</b> {self.acq_time:.3f} s<br>"
                f"<b>Scans:</b> {self.scan_count}"
                f"{param_status}"
            )
            
            # Initialize scan selection controls (block signals to prevent reload)
            if self.scan_count > 0:
                # Update spin box ranges without triggering signals
                self.scan_single_num.blockSignals(True)
                self.scan_range_start.blockSignals(True)
                self.scan_range_end.blockSignals(True)
                
                self.scan_single_num.setMaximum(self.scan_count - 1)
                self.scan_single_num.setValue(0)
                self.scan_range_start.setMaximum(self.scan_count - 1)
                self.scan_range_start.setValue(0)
                self.scan_range_end.setMaximum(self.scan_count - 1)
                self.scan_range_end.setValue(self.scan_count - 1)
                
                # Re-enable signals
                self.scan_single_num.blockSignals(False)
                self.scan_range_start.blockSignals(False)
                self.scan_range_end.blockSignals(False)
                
                # Update info label
                self.update_scan_selection_info()
                
                # Enable Apply Selection button
                self.apply_scan_btn.setEnabled(True)
            
            self.process_btn.setEnabled(True)
            self.save_params_btn.setEnabled(True)
            
            # Auto process
            self.process_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{e}")

    def load_multiple_folders(self):
        """Load and combine data from multiple folders"""
        dialog = MultiFolderDialog(self)
        if dialog.exec() == QDialog.Accepted:
            folders = dialog.get_folders()
            if not folders:
                return
                
            try:
                self.progress_bar.setVisible(True)
                self.progress_label.setText("Combining data from multiple folders...")
                QApplication.processEvents()
                
                total_scans = 0
                weighted_sum_data = None
                common_sr = None
                common_at = None
                
                for i, folder in enumerate(folders):
                    # Update progress
                    self.progress_label.setText(f"Loading folder {i+1}/{len(folders)}: {os.path.basename(folder)}")
                    QApplication.processEvents()
                    
                    # Load data from each folder
                    if HAS_NMRDUINO:
                        # Try to load compiled data
                        compiled_path = os.path.join(folder, "halp_compiled.npy")
                        if os.path.exists(compiled_path):
                            halp = np.load(compiled_path)
                            sr = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                            at = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                            scans = nmr_util.scan_number_extraction(folder)
                        else:
                            # Load and compile
                            compiled = nmr_util.nmrduino_dat_interp(folder, 0)
                            halp = compiled[0]
                            sr = compiled[1]
                            at = compiled[2]
                            scans = nmr_util.scan_number_extraction(folder)
                    else:
                        # Manual loading fallback
                        compiled_path = os.path.join(folder, "halp_compiled.npy")
                        if os.path.exists(compiled_path):
                            halp = np.load(compiled_path)
                            sr = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                            at = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                            # Try to count .dat files
                            dat_files = [f for f in os.listdir(folder) if f.endswith('.dat')]
                            scans = len(dat_files) if dat_files else 1
                        else:
                            raise FileNotFoundError(f"No compiled data found in {folder}")
                    
                    # Check consistency
                    if common_sr is None:
                        common_sr = sr
                        common_at = at
                        weighted_sum_data = np.zeros_like(halp, dtype=complex)
                    else:
                        if abs(sr - common_sr) > 1.0:
                            raise ValueError(f"Sampling rate mismatch in {folder}")
                        if abs(at - common_at) > 0.01:
                            # Allow small mismatch, maybe truncate?
                            # For now, strict check or resize
                            if len(halp) != len(weighted_sum_data):
                                # Resize to match smallest? Or error?
                                # Let's assume they must match
                                raise ValueError(f"Data length mismatch in {folder}")
                    
                    # Accumulate weighted sum
                    # Assuming halp is the average of scans in that folder
                    weighted_sum_data += halp * scans
                    total_scans += scans
                
                # Calculate final average
                if total_scans > 0:
                    self.halp = weighted_sum_data / total_scans
                    self.sampling_rate = common_sr
                    self.acq_time = common_at
                    self.scan_count = total_scans
                    self.current_path = folders[0] # Use first folder as reference path
                    
                    self.data_info.setText(
                        f"<b>Combined:</b> {len(folders)} folders<br>"
                        f"<b>Points:</b> {len(self.halp)}<br>"
                        f"<b>Sampling:</b> {self.sampling_rate:.1f} Hz<br>"
                        f"<b>Acq Time:</b> {self.acq_time:.3f} s<br>"
                        f"<b>Total Scans:</b> {self.scan_count}"
                    )
                    
                    # Enable controls
                    self.process_btn.setEnabled(True)
                    self.save_params_btn.setEnabled(True)
                    
                    # Reset scan selection (since we combined, individual scan selection is complex)
                    self.scan_mode_all.setChecked(True)
                    self.scan_selection_info.setText("Scan selection disabled for combined data")
                    self.apply_scan_btn.setEnabled(False)
                    
                    # Auto process
                    self.process_data()
                    
                self.progress_bar.setVisible(False)
                self.progress_label.setText("")
                
            except Exception as e:
                self.progress_bar.setVisible(False)
                self.progress_label.setText("")
                QMessageBox.critical(self, "Combine Error", f"Failed to combine folders:\n{e}")
    
    @Slot()
    def process_data(self):
        """Process data with current parameters"""
        if self.halp is None:
            return
        
        # Stop previous worker if running
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.quit()
                    self.worker.wait(1000)
                    if self.worker.isRunning():
                        self.worker.terminate()
                        self.worker.wait(500)
            except RuntimeError:
                pass # Worker might be already deleted
            
            # Cleanup old worker
            try:
                self.worker.deleteLater()
            except RuntimeError:
                pass
            self.worker = None
        
        # Update parameters from UI
        self.params = {
            'zf_factor': self.zf_slider.value() / 100.0,
            'use_hanning': 1 if self.use_hanning.isChecked() else 0,
            'conv_points': self.conv_slider.value(),
            'poly_order': self.poly_slider.value(),
            'trunc_start': self.trunc_start_slider.value(),
            'trunc_end': self.trunc_end_slider.value(),
            'apod_t2star': self.apod_slider.value() / 100.0,
            'freq_low_min': self.freq_low_min.value(),
            'freq_low_max': self.freq_low_max.value(),
            'freq_high_min': self.freq_high_min.value(),
            'freq_high_max': self.freq_high_max.value(),
            # NEW Params
            'enable_recon': self.enable_recon.isChecked(),
            'recon_points': self.recon_points.value(),
            'recon_order': self.recon_order.value(),
            'recon_train_len': self.recon_train_points.value(),
            'enable_svd': self.enable_svd.isChecked(),
            'svd_rank': self.svd_rank_spin.value(),
            'phase0': self.phase0_spin.value(),
            'phase1': self.phase1_spin.value(),
            # Baseline Params
            'baseline_method': self.baseline_method.currentText().lower(),
            'baseline_order': self.baseline_poly_order.value(),
            'baseline_lambda': self.baseline_lambda.value()
        }
        
        # Update params_b if using same parameters
        if self.use_same_params:
            self.params_b = self.params.copy()
        else:
            # Update params_b from UI controls with correct key mapping
            self.params_b['conv_points'] = self.conv_spinbox_b.value()
            self.params_b['poly_order'] = self.poly_spinbox_b.value()
            self.params_b['trunc_start'] = self.trunc_spinbox_b.value()
            self.params_b['trunc_end'] = 10  # Default fixed value
            self.params_b['apod_t2star'] = self.t2_spinbox_b.value()
            self.params_b['use_hanning'] = 1 if self.hanning_spinbox_b.value() > 0.1 else 0
            # Map Zero Fill: treat > 0 as factor 1.0 (double) for safety
            self.params_b['zf_factor'] = 1.0 if self.zerofill_spinbox_b.value() > 0 else 0.0
        
        # Reset processed_b to trigger reprocessing if in comparison mode
        if self.comparison_mode and self.halp_b is not None:
            self.processed_b = None
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Process Data A
        self.worker = ProcessingWorker(
            self.halp,
            self.sampling_rate,
            self.acq_time,
            self.params
        )
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_processing_progress)
        self.worker.start()
        
        # Process Data B if in comparison mode
        if self.comparison_mode and self.halp_b is not None:
            # Wait for Data A processing to finish, then process Data B
            # For now, we'll process them sequentially
            pass  # Will be handled in on_processing_finished
    
    @Slot(str)
    def on_processing_progress(self, message):
        """Update progress message"""
        self.progress_label.setText(message)
    
    @Slot(object)
    def on_processing_finished(self, result):
        """Handle processing finished"""
        self.processed = result
        
        # If in comparison mode and Data B needs processing
        if self.comparison_mode and self.halp_b is not None and self.processed_b is None:
            # Cleanup Data A worker
            if self.worker is not None:
                self.worker.deleteLater()
                self.worker = None
            
            # Process Data B
            self.progress_label.setText("Processing Data B...")
            self.worker = ProcessingWorker(
                self.halp_b,
                self.sampling_rate_b,
                self.acq_time_b,
                self.params_b
            )
            self.worker.finished.connect(self.on_processing_b_finished)
            self.worker.error.connect(self.on_processing_error)
            self.worker.progress.connect(self.on_processing_progress)
            self.worker.start()
            return
        
        # Cleanup worker after processing
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.plot_results()
        self.calculate_metrics()
        self.export_btn.setEnabled(True)
    
    @Slot(object)
    def on_processing_b_finished(self, result):
        """Handle Data B processing finished"""
        self.processed_b = result
        
        # Cleanup worker after Data B processing
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.plot_results()
        self.calculate_metrics()
        self.export_btn.setEnabled(True)
    
    @Slot(str)
    def on_processing_error(self, error):
        """Handle processing error"""
        # Cleanup worker on error
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Processing Error", f"Failed to process:\n{error}")
    
    def plot_results(self):
        """Plot processing results"""
        if self.processed is None:
            return
        
        # Check if in comparison mode with side-by-side
        if self.comparison_mode and self.processed_b is not None and self.display_side_by_side.isChecked():
            self.plot_side_by_side_comparison()
            return
        
        # Check if in comparison mode with overlay (Unified or Normalized)
        if self.comparison_mode and self.processed_b is not None and (self.display_overlay.isChecked() or self.display_overlay_norm.isChecked()):
            self.plot_overlay_comparison()
            return
        
        time_data = self.processed['time_data']
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        acq_time = self.processed['acq_time_effective']
        
        # Determine plot data based on view mode
        if self.view_real.isChecked():
            plot_data = np.real(spectrum)
            if self.show_absolute.isChecked():
                plot_data = np.abs(plot_data)
                ylabel = "Amplitude |Real|"
            else:
                ylabel = "Amplitude (Real)"
        elif self.view_imag.isChecked():
            plot_data = np.imag(spectrum)
            if self.show_absolute.isChecked():
                plot_data = np.abs(plot_data)
                ylabel = "Amplitude |Imag|"
            else:
                ylabel = "Amplitude (Imag)"
        else:
            plot_data = np.abs(spectrum)
            ylabel = "Magnitude"
        
        # Time domain
        time_axis = np.linspace(0, acq_time, len(time_data))
        self.time_canvas.axes.clear()
        
        # Plot main signal
        self.time_canvas.axes.plot(time_axis, np.real(time_data), 'b-', linewidth=0.8, alpha=0.8, label='Processed Signal')
        
        # Highlight Backward LP region
        n_backward = self.processed.get('n_backward', 0)
        lp_train_len = self.processed.get('lp_train_len', 0)
        
        if n_backward > 0:
            # Ensure we don't go out of bounds
            n_backward = min(n_backward, len(time_axis))
            self.time_canvas.axes.plot(
                time_axis[:n_backward], 
                np.real(time_data[:n_backward]), 
                'r-', 
                linewidth=1.2, 
                label='Backward LP'
            )
            
            # Highlight Training Region
            if lp_train_len > 0:
                # Start index: n_backward
                # End index: n_backward + lp_train_len
                idx_start = n_backward
                idx_end = min(n_backward + lp_train_len, len(time_axis)-1)
                
                if idx_end > idx_start:
                    t_start = time_axis[idx_start]
                    t_end = time_axis[idx_end]
                    self.time_canvas.axes.axvspan(
                        t_start, t_end, 
                        color='#4CAF50', # Green
                        alpha=0.15, 
                        label='LP Training Region'
                    )
            
        # Highlight Zero Filled region
        original_len = self.processed.get('original_len', len(time_data))
        if original_len < len(time_data):
            self.time_canvas.axes.plot(
                time_axis[original_len:], 
                np.real(time_data[original_len:]), 
                color='#FF9800', # Orange
                linestyle='--',
                linewidth=1.0, 
                label='Zero Filled'
            )
            # Add a vertical line to mark the boundary
            self.time_canvas.axes.axvline(x=time_axis[original_len], color='gray', linestyle=':', alpha=0.5)
        
        self.time_canvas.axes.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        self.time_canvas.axes.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        self.time_canvas.axes.set_title('Time Domain Signal', fontsize=11, fontweight='bold')
        self.time_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.time_canvas.axes.legend(fontsize=8, loc='upper right')
        self.time_canvas.axes.autoscale(enable=True, axis='y', tight=False)
        self.time_canvas.fig.tight_layout()
        self.time_canvas.draw()
        
        # Frequency domain - low freq
        freq_range_low = [self.freq_low_min.value(), self.freq_low_max.value()]
        
        self.freq1_canvas.axes.clear()
        self.freq1_canvas.axes.plot(freq_axis, plot_data, 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        
        # Plot Baseline if requested
        freq_baseline = self.processed.get('freq_baseline', None)
        if self.show_baseline.isChecked() and freq_baseline is not None and self.view_real.isChecked():
            self.freq1_canvas.axes.plot(freq_axis, freq_baseline, 'g--', linewidth=1.0, alpha=0.7, label='Baseline')
        
        self.freq1_canvas.axes.set_xlim(freq_range_low[0], freq_range_low[1])
        idx_visible = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
        if np.any(idx_visible):
            y_visible = plot_data[idx_visible]
            y_max = np.max(y_visible)
            y_min = np.min(y_visible)
            range_y = y_max - y_min
            if range_y == 0: range_y = 1.0
            self.freq1_canvas.axes.set_ylim(y_min - 0.05 * range_y, y_max + 0.05 * range_y)
            
        self.freq1_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq1_canvas.axes.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        self.freq1_canvas.axes.set_title(f'Low Frequency Spectrum ({freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)', fontsize=11, fontweight='bold')
        self.freq1_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq1_canvas.fig.tight_layout()
        self.freq1_canvas.draw()
        
        # Frequency domain - high freq
        freq_range_high = [self.freq_high_min.value(), self.freq_high_max.value()]
        
        self.freq2_canvas.axes.clear()
        self.freq2_canvas.axes.plot(freq_axis, plot_data, 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        
        # Plot Baseline if requested
        if self.show_baseline.isChecked() and freq_baseline is not None and self.view_real.isChecked():
            self.freq2_canvas.axes.plot(freq_axis, freq_baseline, 'g--', linewidth=1.0, alpha=0.7, label='Baseline')
        
        self.freq2_canvas.axes.set_xlim(freq_range_high[0], freq_range_high[1])
        idx_visible = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
        if np.any(idx_visible):
            y_visible = plot_data[idx_visible]
            y_max = np.max(y_visible)
            y_min = np.min(y_visible)
            range_y = y_max - y_min
            if range_y == 0: range_y = 1.0
            self.freq2_canvas.axes.set_ylim(y_min - 0.05 * range_y, y_max + 0.05 * range_y)
            
        self.freq2_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq2_canvas.axes.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        self.freq2_canvas.axes.set_title(f'High Frequency Spectrum ({freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)', fontsize=11, fontweight='bold')
        self.freq2_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq2_canvas.fig.tight_layout()
        self.freq2_canvas.draw()
    
    def plot_side_by_side_comparison(self):
        """Plot Data A and B side by side using subplots"""
        time_data = self.processed['time_data']
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        acq_time = self.processed['acq_time_effective']
        
        time_data_b = self.processed_b['time_data']
        freq_axis_b = self.processed_b['freq_axis']
        spectrum_b = self.processed_b['spectrum']
        acq_time_b = self.processed_b['acq_time_effective']
        
        # Time domain - side by side
        time_axis = np.linspace(0, acq_time, len(time_data))
        time_axis_b = np.linspace(0, acq_time_b, len(time_data_b))
        
        self.time_canvas.fig.clear()
        ax1 = self.time_canvas.fig.add_subplot(1, 2, 1)
        ax2 = self.time_canvas.fig.add_subplot(1, 2, 2)
        
        ax1.plot(time_axis, np.real(time_data), 'b-', linewidth=0.8)
        ax1.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax1.set_title('Data A - Time Domain', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.plot(time_axis_b, np.real(time_data_b), 'r-', linewidth=0.8)
        ax2.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax2.set_title('Data B - Time Domain', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        self.time_canvas.fig.tight_layout()
        self.time_canvas.draw()
        
        # Frequency domain - low freq - side by side
        freq_range_low = [self.freq_low_min.value(), self.freq_low_max.value()]
        
        self.freq1_canvas.fig.clear()
        ax1 = self.freq1_canvas.fig.add_subplot(1, 2, 1)
        ax2 = self.freq1_canvas.fig.add_subplot(1, 2, 2)
        
        ax1.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.0)
        ax1.set_xlim(freq_range_low[0], freq_range_low[1])
        idx_visible = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
        if np.any(idx_visible):
            y_visible = np.abs(spectrum)[idx_visible]
            y_max = np.max(y_visible)
            ax1.set_ylim(-0.05 * y_max, 1.1 * y_max)
        ax1.set_xlabel('Frequency (Hz)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax1.set_title(f'Data A - Low Freq ({freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.plot(freq_axis_b, np.abs(spectrum_b), 'r-', linewidth=1.0)
        ax2.set_xlim(freq_range_low[0], freq_range_low[1])
        idx_visible_b = (freq_axis_b >= freq_range_low[0]) & (freq_axis_b <= freq_range_low[1])
        if np.any(idx_visible_b):
            y_visible_b = np.abs(spectrum_b)[idx_visible_b]
            y_max_b = np.max(y_visible_b)
            ax2.set_ylim(-0.05 * y_max_b, 1.1 * y_max_b)
        ax2.set_xlabel('Frequency (Hz)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax2.set_title(f'Data B - Low Freq ({freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        self.freq1_canvas.fig.tight_layout()
        self.freq1_canvas.draw()
        
        # Frequency domain - high freq - side by side
        freq_range_high = [self.freq_high_min.value(), self.freq_high_max.value()]
        
        self.freq2_canvas.fig.clear()
        ax1 = self.freq2_canvas.fig.add_subplot(1, 2, 1)
        ax2 = self.freq2_canvas.fig.add_subplot(1, 2, 2)
        
        ax1.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.0)
        ax1.set_xlim(freq_range_high[0], freq_range_high[1])
        idx_visible = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
        if np.any(idx_visible):
            y_visible = np.abs(spectrum)[idx_visible]
            y_max = np.max(y_visible)
            ax1.set_ylim(-0.05 * y_max, 1.1 * y_max)
        ax1.set_xlabel('Frequency (Hz)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax1.set_title(f'Data A - High Freq ({freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.plot(freq_axis_b, np.abs(spectrum_b), 'r-', linewidth=1.0)
        ax2.set_xlim(freq_range_high[0], freq_range_high[1])
        idx_visible_b = (freq_axis_b >= freq_range_high[0]) & (freq_axis_b <= freq_range_high[1])
        if np.any(idx_visible_b):
            y_visible_b = np.abs(spectrum_b)[idx_visible_b]
            y_max_b = np.max(y_visible_b)
            ax2.set_ylim(-0.05 * y_max_b, 1.1 * y_max_b)
        ax2.set_xlabel('Frequency (Hz)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
        ax2.set_title(f'Data B - High Freq ({freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        self.freq2_canvas.fig.tight_layout()
        self.freq2_canvas.draw()
    
    def plot_overlay_comparison(self):
        """Plot Data A and B overlaid on same axes"""
        time_data = self.processed['time_data']
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        acq_time = self.processed['acq_time_effective']
        
        time_data_b = self.processed_b['time_data']
        freq_axis_b = self.processed_b['freq_axis']
        spectrum_b = self.processed_b['spectrum']
        acq_time_b = self.processed_b['acq_time_effective']
        
        # Check if normalized mode
        is_normalized = self.display_overlay_norm.isChecked()
        
        # Prepare data for plotting
        y_time_a = np.real(time_data)
        y_time_b = np.real(time_data_b)
        y_spec_a = np.abs(spectrum)
        y_spec_b = np.abs(spectrum_b)
        
        if is_normalized:
            # Normalize Time Domain (Global)
            max_time_a = np.max(np.abs(y_time_a))
            max_time_b = np.max(np.abs(y_time_b))
            if max_time_a > 0: y_time_a = y_time_a / max_time_a
            if max_time_b > 0: y_time_b = y_time_b / max_time_b
            
            ylabel = "Normalized Amplitude"
        else:
            ylabel = "Amplitude"
        
        # Time domain - overlay
        time_axis = np.linspace(0, acq_time, len(time_data))
        time_axis_b = np.linspace(0, acq_time_b, len(time_data_b))
        
        self.time_canvas.fig.clear()
        self.time_canvas.axes = self.time_canvas.fig.add_subplot(111)
        self.time_canvas.axes.plot(time_axis, y_time_a, 'b-', linewidth=0.8, alpha=0.8, label='Data A')
        self.time_canvas.axes.plot(time_axis_b, y_time_b, 'r-', linewidth=0.8, alpha=0.6, label='Data B')
        
        # Set y-axis limits based on both datasets
        y_max_a = np.max(np.abs(y_time_a))
        y_max_b = np.max(np.abs(y_time_b))
        y_max_combined = max(y_max_a, y_max_b)
        self.time_canvas.axes.set_ylim(-1.1 * y_max_combined, 1.1 * y_max_combined)
        
        self.time_canvas.axes.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        self.time_canvas.axes.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        title_suffix = " (Normalized)" if is_normalized else ""
        self.time_canvas.axes.set_title(f'Time Domain - Overlay Comparison{title_suffix}', fontsize=11, fontweight='bold')
        self.time_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.time_canvas.axes.legend(fontsize=9)
        self.time_canvas.fig.tight_layout()
        self.time_canvas.draw()
        
        # Frequency domain - low freq - overlay
        freq_range_low = [self.freq_low_min.value(), self.freq_low_max.value()]
        
        # Prepare data for Low Freq (Local Normalization if needed)
        y_spec_a_low = y_spec_a.copy()
        y_spec_b_low = y_spec_b.copy()
        
        if is_normalized:
            # Local normalization for Low Freq view
            idx_visible_a = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
            idx_visible_b = (freq_axis_b >= freq_range_low[0]) & (freq_axis_b <= freq_range_low[1])
            
            if np.any(idx_visible_a):
                max_val = np.max(y_spec_a_low[idx_visible_a])
                if max_val > 0: y_spec_a_low = y_spec_a_low / max_val
            
            if np.any(idx_visible_b):
                max_val = np.max(y_spec_b_low[idx_visible_b])
                if max_val > 0: y_spec_b_low = y_spec_b_low / max_val
        else:
            # Global normalization for Low Freq view
            # For Unified Scale, we want to preserve relative amplitudes, so we don't normalize individually
            # or we could normalize both by the global max of both.
            # Here we choose to use raw values (or normalized by global max if we wanted 0-1)
            # But to be consistent with "Unified Scale" meaning "Same Scale", raw values are fine.
            pass
        
        self.freq1_canvas.fig.clear()
        self.freq1_canvas.axes = self.freq1_canvas.fig.add_subplot(111)
        self.freq1_canvas.axes.plot(freq_axis, y_spec_a_low, 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        self.freq1_canvas.axes.plot(freq_axis_b, y_spec_b_low, 'r-', linewidth=1.0, alpha=0.6, label='Data B')
        self.freq1_canvas.axes.set_xlim(freq_range_low[0], freq_range_low[1])
        
        # Set y-axis limits based on both datasets in visible range
        idx_visible = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
        idx_visible_b = (freq_axis_b >= freq_range_low[0]) & (freq_axis_b <= freq_range_low[1])
        y_max_list = []
        if np.any(idx_visible):
            y_max_list.append(np.max(y_spec_a_low[idx_visible]))
        if np.any(idx_visible_b):
            y_max_list.append(np.max(y_spec_b_low[idx_visible_b]))
        if y_max_list:
            y_max_combined = max(y_max_list)
            self.freq1_canvas.axes.set_ylim(-0.05 * y_max_combined, 1.1 * y_max_combined)
        
        self.freq1_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq1_canvas.axes.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        self.freq1_canvas.axes.set_title(f'Low Freq Comparison{title_suffix} ({freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)', fontsize=11, fontweight='bold')
        self.freq1_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq1_canvas.axes.legend(fontsize=9)
        self.freq1_canvas.fig.tight_layout()
        self.freq1_canvas.draw()
        
        # Frequency domain - high freq - overlay
        freq_range_high = [self.freq_high_min.value(), self.freq_high_max.value()]
        
        # Prepare data for High Freq (Local Normalization if needed)
        y_spec_a_high = y_spec_a.copy()
        y_spec_b_high = y_spec_b.copy()
        
        if is_normalized:
            # Local normalization for High Freq view
            idx_visible_a = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
            idx_visible_b = (freq_axis_b >= freq_range_high[0]) & (freq_axis_b <= freq_range_high[1])
            
            if np.any(idx_visible_a):
                max_val = np.max(y_spec_a_high[idx_visible_a])
                if max_val > 0: y_spec_a_high = y_spec_a_high / max_val
            
            if np.any(idx_visible_b):
                max_val = np.max(y_spec_b_high[idx_visible_b])
                if max_val > 0: y_spec_b_high = y_spec_b_high / max_val
        else:
            # Global normalization for High Freq view
            # For Unified Scale, use raw values to preserve relative amplitudes
            pass
        
        self.freq2_canvas.fig.clear()
        self.freq2_canvas.axes = self.freq2_canvas.fig.add_subplot(111)
        self.freq2_canvas.axes.plot(freq_axis, y_spec_a_high, 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        self.freq2_canvas.axes.plot(freq_axis_b, y_spec_b_high, 'r-', linewidth=1.0, alpha=0.6, label='Data B')
        self.freq2_canvas.axes.set_xlim(freq_range_high[0], freq_range_high[1])
        
        idx_visible_a = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
        idx_visible_b = (freq_axis_b >= freq_range_high[0]) & (freq_axis_b <= freq_range_high[1])
        y_max_list = []
        if np.any(idx_visible_a):
            y_max_list.append(np.max(y_spec_a_high[idx_visible_a]))
        if np.any(idx_visible_b):
            y_max_list.append(np.max(y_spec_b_high[idx_visible_b]))
        if y_max_list:
            y_max_combined = max(y_max_list)
            self.freq2_canvas.axes.set_ylim(-0.05 * y_max_combined, 1.1 * y_max_combined)
            
        self.freq2_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq2_canvas.axes.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        self.freq2_canvas.axes.set_title(f'High Freq Comparison{title_suffix} ({freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)', fontsize=11, fontweight='bold')
        self.freq2_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq2_canvas.axes.legend(fontsize=9)
        self.freq2_canvas.fig.tight_layout()
        self.freq2_canvas.draw()
    
    # --- NEW Methods for Phase and Recon ---
    
    def on_phase_changed(self):
        """Handle phase slider changes (Fast Update)"""
        # Update params
        self.params['phase0'] = self.phase0_slider.value()
        self.params['phase1'] = self.phase1_slider.value() / 10.0 # Scale down
        
        # Sync spinboxes
        self.phase0_spin.blockSignals(True)
        self.phase0_spin.setValue(self.params['phase0'])
        self.phase0_spin.blockSignals(False)
        
        self.phase1_spin.blockSignals(True)
        self.phase1_spin.setValue(self.params['phase1'])
        self.phase1_spin.blockSignals(False)
        
        # Fast update
        self.update_phase_only()
        
    def on_phase_spin_changed(self):
        """Handle phase spinbox changes"""
        self.params['phase0'] = self.phase0_spin.value()
        self.params['phase1'] = self.phase1_spin.value()
        
        # Sync sliders
        self.phase0_slider.blockSignals(True)
        self.phase0_slider.setValue(int(self.params['phase0']))
        self.phase0_slider.blockSignals(False)
        
        self.phase1_slider.blockSignals(True)
        self.phase1_slider.setValue(int(self.params['phase1'] * 10))
        self.phase1_slider.blockSignals(False)
        
        self.update_phase_only()
        
    def update_phase_only(self):
        """Apply phase correction to cached spectrum without re-processing"""
        if self.processed is None or 'spectrum_complex' not in self.processed:
            return
            
        # Get cached unphased spectrum
        spec = self.processed['spectrum_complex']
        
        # Apply phase
        phi0 = self.params['phase0']
        phi1 = self.params['phase1']
        
        # Use center as pivot for better UX
        pivot = len(spec) // 2
        
        phased_spec = apply_phase_correction(spec, phi0, phi1, pivot_index=pivot)
        
        # Update processed result
        self.processed['spectrum'] = phased_spec
        
        # Re-plot
        self.plot_results()
        
        # Recalculate metrics (SNR depends on phase if using Real part)
        self.calculate_metrics()
        
    def run_auto_phase(self):
        """Run auto-phasing algorithm"""
        if self.processed is None or 'spectrum_complex' not in self.processed:
            return
            
        self.progress_label.setText("Auto-phasing...")
        QApplication.processEvents()
        
        spec = self.processed['spectrum_complex']
        p0, p1 = auto_phase(spec)
        
        # Update UI
        self.phase0_spin.setValue(p0)
        self.phase1_spin.setValue(p1)
        
        self.progress_label.setText("Auto-phase complete")
        
    def update_plot_view(self):
        """Update plot view mode (Real/Imag/Mag)"""
        self.plot_results()
        self.calculate_metrics()
        
    # --- End NEW Methods ---

    def load_parameters(self):
        """Load parameters from JSON file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return
            
        try:
            with open(file_name, 'r') as f:
                params = json.load(f)
            
            # Update UI sliders
            self.conv_slider.setValue(int(params.get('conv_points', 300)))
            self.poly_slider.setValue(int(params.get('poly_order', 2)))
            self.trunc_start_slider.setValue(int(params.get('trunc_start', 10)))
            self.trunc_end_slider.setValue(int(params.get('trunc_end', 10)))
            self.apod_slider.setValue(int(float(params.get('apod_t2star', 0.0)) * 100))
            self.use_hanning.setChecked(bool(params.get('use_hanning', 0)))
            self.zf_slider.setValue(int(float(params.get('zf_factor', 0.0)) * 100))
            
            # Update frequency display range sliders if present
            if 'freq_low_min' in params:
                self.freq_low_min.setValue(float(params['freq_low_min']))
            if 'freq_low_max' in params:
                self.freq_low_max.setValue(float(params['freq_low_max']))
            if 'freq_high_min' in params:
                self.freq_high_min.setValue(float(params['freq_high_min']))
            if 'freq_high_max' in params:
                self.freq_high_max.setValue(float(params['freq_high_max']))
                
            # Update ZULF params
            if 'phase0' in params:
                self.phase0_spin.setValue(float(params['phase0']))
            if 'phase1' in params:
                self.phase1_spin.setValue(float(params['phase1']))
            if 'enable_recon' in params:
                self.enable_recon.setChecked(bool(params['enable_recon']))
            if 'recon_points' in params:
                self.recon_points.setValue(int(params['recon_points']))
            if 'recon_order' in params:
                self.recon_order.setValue(int(params['recon_order']))
                
            # Update Baseline params
            if 'baseline_method' in params:
                self.baseline_method.setCurrentText(str(params['baseline_method']))
            if 'baseline_order' in params:
                self.baseline_poly_order.setValue(int(params['baseline_order']))
            if 'baseline_lambda' in params:
                self.baseline_lambda.setValue(int(params['baseline_lambda']))
            
            self.params = params
            QMessageBox.information(self, "Success", "Parameters loaded successfully.")
            self.schedule_processing()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters:\n{e}")

    def save_parameters(self):
        """Save current parameters to JSON file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return
            
        try:
            # Ensure params are up to date
            current_params = {
                'zf_factor': self.zf_slider.value() / 100.0,
                'use_hanning': 1 if self.use_hanning.isChecked() else 0,
                'conv_points': self.conv_slider.value(),
                'poly_order': self.poly_slider.value(),
                'trunc_start': self.trunc_start_slider.value(),
                'trunc_end': self.trunc_end_slider.value(),
                'apod_t2star': self.apod_slider.value() / 100.0,
                'freq_low_min': self.freq_low_min.value(),
                'freq_low_max': self.freq_low_max.value(),
                'freq_high_min': self.freq_high_min.value(),
                'freq_high_max': self.freq_high_max.value(),
                'enable_recon': self.enable_recon.isChecked(),
                'recon_points': self.recon_points.value(),
                'recon_order': self.recon_order.value(),
                'phase0': self.phase0_spin.value(),
                'phase1': self.phase1_spin.value(),
                'baseline_method': self.baseline_method.currentText(),
                'baseline_order': self.baseline_poly_order.value(),
                'baseline_lambda': self.baseline_lambda.value()
            }
            
            with open(file_name, 'w') as f:
                json.dump(current_params, f, indent=4)
                
            QMessageBox.information(self, "Success", "Parameters saved successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters:\n{e}")

    def export_results(self):
        """Export processed data to NPZ"""
        if self.processed is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "NumPy Files (*.npz);;All Files (*)"
        )
        if not file_name:
            return
            
        try:
            np.savez(
                file_name,
                time_data=self.processed['time_data'],
                freq_axis=self.processed['freq_axis'],
                spectrum=self.processed['spectrum'],
                params=self.params
            )
            QMessageBox.information(self, "Success", "Results exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results:\n{e}")

    def export_figures_svg(self):
        """Export figures as SVG"""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
            
        try:
            self.time_canvas.fig.savefig(os.path.join(folder, "time_domain.svg"))
            self.freq1_canvas.fig.savefig(os.path.join(folder, "freq_low.svg"))
            self.freq2_canvas.fig.savefig(os.path.join(folder, "freq_high.svg"))
            QMessageBox.information(self, "Success", "Figures exported as SVG.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export figures:\n{e}")

    def toggle_comparison_mode(self, checked):
        """Toggle comparison mode UI elements"""
        self.comparison_mode = checked
        self.data_b_group.setVisible(checked)
        self.comparison_controls_group.setVisible(checked)
        
        # If disabling, clear Data B plots
        if not checked:
            self.processed_b = None
            self.plot_results()

    def load_folder_b(self):
        """Load Data B folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data B Folder")
        if not folder:
            return
            
        try:
            self.current_path_b = folder
            if HAS_NMRDUINO:
                compiled = nmr_util.nmrduino_dat_interp(folder, 0)
                self.halp_b = compiled[0]
                self.sampling_rate_b = compiled[1]
                self.acq_time_b = compiled[2]
                self.scan_count_b = nmr_util.scan_number_extraction(folder)
            else:
                # Fallback manual load
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp_b = np.load(compiled_path)
                    self.sampling_rate_b = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time_b = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                else:
                    raise FileNotFoundError("No compiled data found for Data B")
            
            self.data_b_info.setText(f"Loaded: {os.path.basename(folder)}")
            
            # Process if we have params
            if self.use_same_params:
                self.process_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load Data B:\n{e}")

    # =========================================================================
    # Live Monitor Methods
    # =========================================================================

    def select_live_folder(self):
        """Select folder for live monitoring"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Monitor")
        if folder:
            self.live_folder = folder
            self.live_folder_edit.setText(folder)
            self.monitor_btn.setEnabled(True)
            self.live_status_label.setText("Ready to monitor")

    def toggle_monitoring(self, checked):
        """Start or stop live monitoring"""
        if checked:
            # Start monitoring
            if not self.live_folder:
                QMessageBox.warning(self, "Error", "Please select a folder first")
                self.monitor_btn.setChecked(False)
                return
            
            if RealtimeDataMonitor is None:
                QMessageBox.critical(self, "Error", "RealtimeDataMonitor module not found")
                self.monitor_btn.setChecked(False)
                return
                
            try:
                self.monitor = RealtimeDataMonitor(self.live_folder, poll_interval=1.0)
                
                # Connect callbacks to signals
                # We use a closure to capture 'self' safely
                def on_new_data(nmr_data, count):
                    self.live_signals.data_received.emit(nmr_data, count)
                    
                def on_error(msg):
                    self.live_signals.error_occurred.emit(msg)
                    
                self.monitor.on_average_updated = on_new_data
                self.monitor.on_new_scan = on_new_data  # Handle both modes
                self.monitor.on_error = on_error
                
                # Start
                avg_mode = self.live_mode_avg.isChecked()
                self.monitor.start(average_mode=avg_mode)
                
                self.is_monitoring = True
                self.monitor_btn.setText("Stop Monitoring")
                self.monitor_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #424242;
                        color: white;
                        padding: 8px;
                        font-weight: bold;
                        border-radius: 4px;
                    }
                """)
                self.live_folder_edit.setEnabled(False)
                self.live_status_label.setText("Monitoring active...")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start monitor:\n{e}")
                self.monitor_btn.setChecked(False)
        else:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop()
                self.monitor = None
            
            self.is_monitoring = False
            self.monitor_btn.setText("Start Monitoring")
            self.monitor_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f;
                    color: white;
                    padding: 8px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:checked {
                    background-color: #b71c1c;
                }
            """)
            self.live_folder_edit.setEnabled(True)
            self.live_status_label.setText("Monitoring stopped")

    def on_live_mode_changed(self):
        """Handle live mode change"""
        if self.monitor and self.is_monitoring:
            avg_mode = self.live_mode_avg.isChecked()
            self.monitor.set_mode(avg_mode)

    @Slot(object, int)
    def handle_live_data(self, nmr_data, scan_count):
        """Handle incoming live data"""
        try:
            # Update data structures
            self.halp = nmr_data.time_data
            self.sampling_rate = nmr_data.sampling_rate
            self.acq_time = nmr_data.acquisition_time
            self.scan_count = scan_count
            self.current_path = self.live_folder
            
            # Update UI info
            self.data_info.setText(f"LIVE MONITORING\nScans: {scan_count}\nPoints: {len(self.halp)}")
            self.live_status_label.setText(f"Last update: {scan_count} scans")
            
            # Trigger processing
            self.process_data()
            
        except Exception as e:
            print(f"Error handling live data: {e}")

    @Slot(str)
    def handle_live_error(self, msg):
        """Handle live monitor error"""
        self.live_status_label.setText(f"Error: {msg}")
        print(f"Live Monitor Error: {msg}")

    @Slot(str)
    def update_live_status(self, msg):
        """Update status label"""
        self.live_status_label.setText(msg)

    def on_same_params_changed(self, state):
        """Handle same params checkbox change"""
        self.use_same_params = (state == Qt.Checked)
        self.data_b_params_group.setVisible(not self.use_same_params)
        
    def apply_data_b_params(self):
        """Apply independent parameters for Data B"""
        if not self.use_same_params:
            self.process_data()
            
    def apply_comparison(self):
        """Apply comparison settings (re-plot)"""
        self.plot_results()
        
    def on_display_mode_changed(self, checked):
        """Handle display mode change"""
        if checked:
            self.plot_results()

    def maximize_plot(self, plot_type):
        """Maximize a specific plot"""
        if plot_type == 'time':
            self.plot_splitter.setSizes([1000, 0, 0])
        elif plot_type == 'freq1':
            self.plot_splitter.setSizes([0, 1000, 0])
        elif plot_type == 'freq2':
            self.plot_splitter.setSizes([0, 0, 1000])
            
    def restore_window_state(self):
        """Restore window geometry and state"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        # Ensure cursor is reset
        QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        """Save window state on close"""
        # Stop worker thread if running
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1000)
                if self.worker.isRunning():
                    self.worker.terminate()
        
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
    
    def calculate_metrics(self):
        """Calculate and display metrics"""
        if self.processed is None:
            return
            
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        
        # Determine data source based on visualization settings
        if self.view_real.isChecked():
            spectrum_data = np.real(spectrum)
            mode_name = "Real"
            if self.show_absolute.isChecked():
                spectrum_data = np.abs(spectrum_data)
                mode_name = "|Real|"
        elif self.view_imag.isChecked():
            spectrum_data = np.imag(spectrum)
            mode_name = "Imag"
            if self.show_absolute.isChecked():
                spectrum_data = np.abs(spectrum_data)
                mode_name = "|Imag|"
        else:
            spectrum_data = np.abs(spectrum)
            mode_name = "Mag"
        
        # Get SNR ranges from UI
        sig_min = self.signal_range_min.value()
        sig_max = self.signal_range_max.value()
        noise_min = self.noise_range_min.value()
        noise_max = self.noise_range_max.value()
        
        # Calculate SNR for Data A
        snr_a = 0
        noise_a = 0
        sig_idx = None
        try:
            # Signal peak
            sig_idx = (freq_axis >= sig_min) & (freq_axis <= sig_max)
            if np.any(sig_idx):
                # Use absolute value for peak finding if not already absolute
                # This ensures we find the peak even if it's negative (e.g. in Real mode)
                # But if user selected Abs mode, data is already positive.
                if self.show_absolute.isChecked() or self.view_mag.isChecked():
                    sig_peak = np.max(spectrum_data[sig_idx])
                else:
                    sig_peak = np.max(np.abs(spectrum_data[sig_idx]))
            else:
                sig_peak = 0
                
            # Noise std
            noise_idx = (freq_axis >= noise_min) & (freq_axis <= noise_max)
            if np.any(noise_idx):
                noise_a = np.std(spectrum_data[noise_idx])
            else:
                noise_a = 1.0 # Avoid div by zero
                
            if noise_a > 0:
                snr_a = sig_peak / noise_a
        except Exception:
            pass
            
        # Update Table
        self.metrics_table.item(0, 0).setText(f"SNR Total ({mode_name})")
        self.metrics_table.item(0, 1).setText(f"{snr_a:.2f}")
        
        if self.scan_count > 0:
            snr_norm_a = snr_a / np.sqrt(self.scan_count)
            self.metrics_table.item(1, 0).setText("SNR / sqrt(N)")
            self.metrics_table.item(1, 1).setText(f"{snr_norm_a:.2f}")
        else:
            self.metrics_table.item(1, 0).setText("SNR / sqrt(N)")
            self.metrics_table.item(1, 1).setText("--")
            
        self.metrics_table.item(2, 0).setText("Noise Level")
        self.metrics_table.item(2, 1).setText(f"{noise_a:.2f}")
        
        self.metrics_table.item(3, 0).setText("Linewidth (Hz)")
        
        # Calculate FWHM
        fwhm = 0.0
        if sig_idx is not None and np.any(sig_idx):
            subset_freq = freq_axis[sig_idx]
            subset_spec = spectrum_data[sig_idx]
            
            # Use Abs for FWHM calculation to handle negative peaks in Real mode
            subset_spec_abs = np.abs(subset_spec)
            peak_idx_local = np.argmax(subset_spec_abs)
            peak_val = subset_spec_abs[peak_idx_local]
            half_max = peak_val / 2.0
            
            # Find left crossing
            left_idx = peak_idx_local
            while left_idx > 0 and subset_spec_abs[left_idx] > half_max:
                left_idx -= 1
                
            # Find right crossing
            right_idx = peak_idx_local
            while right_idx < len(subset_spec_abs) - 1 and subset_spec_abs[right_idx] > half_max:
                right_idx += 1
                
            # Linear interpolation for better precision
            if left_idx < peak_idx_local and right_idx > peak_idx_local:
                # Left interp
                y1 = subset_spec_abs[left_idx]
                y2 = subset_spec_abs[left_idx+1]
                x1 = subset_freq[left_idx]
                x2 = subset_freq[left_idx+1]
                if y2 != y1:
                    freq_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    freq_left = x1
                
                # Right interp
                y1 = subset_spec_abs[right_idx-1]
                y2 = subset_spec_abs[right_idx]
                x1 = subset_freq[right_idx-1]
                x2 = subset_freq[right_idx]
                if y2 != y1:
                    freq_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    freq_right = x2
                
                fwhm = freq_right - freq_left
        
        if fwhm > 0:
            self.metrics_table.item(3, 1).setText(f"{fwhm:.2f}")
        else:
            self.metrics_table.item(3, 1).setText("--") 
        
        self.metrics_table.item(4, 0).setText("Peak Freq (Hz)")
        # Find peak freq
        if sig_idx is not None and np.any(sig_idx):
            subset_freq = freq_axis[sig_idx]
            subset_spec = spectrum_data[sig_idx]
            peak_idx = np.argmax(np.abs(subset_spec))
            peak_freq = subset_freq[peak_idx]
            self.metrics_table.item(4, 1).setText(f"{peak_freq:.2f}")
        else:
            self.metrics_table.item(4, 1).setText("--")

        # Update Results Text
        results = []
        results.append("Processing Applied:")
        results.append(f"  Savgol: {self.params['conv_points']}, {self.params['poly_order']}")
        results.append(f"  Phase: {self.params['phase0']:.1f}, {self.params['phase1']:.1f}")
        if self.params['enable_recon']:
            results.append(f"  Recon: {self.params['recon_points']} pts")
        
        self.results_text.setText('\n'.join(results))


def main():
    app = QApplication(sys.argv)
    
    # Set font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Set style
    app.setStyle('Fusion')
    
    window = EnhancedNMRProcessingUI()
    window.show()
    
    # Ensure cursor is normal after initialization
    while app.overrideCursor() is not None:
        app.restoreOverrideCursor()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
