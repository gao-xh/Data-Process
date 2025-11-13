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
    QScrollArea, QMenuBar, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QSettings
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.signal
from scipy.fft import fft

# Import nmrduino_util (using fixed version with proper path handling)
try:
    from nmr_processing_lib import nmrduino_util_fixed as nmr_util
    HAS_NMRDUINO = True
except:
    HAS_NMRDUINO = False
    print("Warning: nmrduino_util not found, some features disabled")


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
        halp = self.halp.copy()
        sampling_rate = self.sampling_rate
        acq_time = self.acq_time
        
        self.progress.emit("Applying Savgol filter...")
        
        # Step 1: Savgol filtering (baseline removal)
        smooth_svd = scipy.signal.savgol_filter(
            halp, 
            int(self.params['conv_points']),
            int(self.params['poly_order']), 
            mode="mirror"
        )
        svd_corrected = halp - smooth_svd
        
        self.progress.emit("Applying truncation...")
        
        # Step 2: Time domain truncation
        trunc_start = int(self.params['trunc_start'])
        trunc_end = int(self.params['trunc_end'])
        svd_corrected = svd_corrected[trunc_start:-trunc_end if trunc_end > 0 else None]
        acq_time_effective = acq_time * (len(svd_corrected) / len(halp))
        
        self.progress.emit("Applying apodization...")
        
        # Step 3: Apodization (exponential decay)
        t = np.linspace(0, acq_time_effective, len(svd_corrected))
        apodization_window = np.exp(-self.params['apod_t2star'] * t)
        svd_corrected = svd_corrected * apodization_window
        
        # Step 4: Hanning window
        if int(self.params['use_hanning']) == 1:
            self.progress.emit("Applying Hanning window...")
            svd_corrected = np.hanning(len(svd_corrected)) * svd_corrected
        
        self.progress.emit("Applying zero filling...")
        
        # Step 5: Zero filling
        zf_factor = self.params['zf_factor']
        if zf_factor > 0:
            zf_length = int(len(svd_corrected) * zf_factor)
            svd_corrected = np.concatenate((
                svd_corrected,
                np.ones(zf_length) * np.mean(svd_corrected)
            ))
        
        self.progress.emit("Computing FFT...")
        
        # Step 6: FFT
        yf = fft(svd_corrected)
        xf = np.linspace(0, sampling_rate, len(yf))
        
        self.progress.emit("Processing complete!")
        
        return {
            'time_data': svd_corrected,
            'freq_axis': xf,
            'spectrum': yf,
            'acq_time_effective': acq_time * (1 + zf_factor),
            'baseline': smooth_svd
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
        
        # Parameters for Data B (independent mode)
        self.params_b = {
            'zf_factor': 0.0,
            'use_hanning': 0,
            'conv_points': 300,
            'poly_order': 2,
            'trunc_start': 10,
            'trunc_end': 10,
            'apod_t2star': 0.0
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
        
        # Mode toggle
        self.comparison_mode_action = view_menu.addAction('Enable Comparison Mode')
        self.comparison_mode_action.setCheckable(True)
        self.comparison_mode_action.setChecked(False)
        self.comparison_mode_action.triggered.connect(self.toggle_comparison_mode)
        
        view_menu.addSeparator()
        
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
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("NMR Data Processing")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1a237e;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border-radius: 6px;
                border: 1px solid #90caf9;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Data loading
        data_group = QGroupBox("Data Loading")
        data_group.setStyleSheet("""
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
        comparison_layout.addWidget(self.display_side_by_side)
        
        self.display_overlay = QRadioButton("Overlay")
        self.display_overlay.setStyleSheet("font-size: 10px;")
        comparison_layout.addWidget(self.display_overlay)
        
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
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
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
        
        # Tab 1: Filtering
        filter_tab = self.create_filter_tab()
        param_tabs.addTab(filter_tab, "Filtering & Apodization")
        
        # Tab 2: Transform
        transform_tab = self.create_transform_tab()
        param_tabs.addTab(transform_tab, "Transform & Display")
        
        layout.addWidget(param_tabs)
        
        # Process button
        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b9b7c;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 14px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #568266;
            }
            QPushButton:pressed {
                background-color: #446952;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
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
                    stop:0 #6b9b7c, stop:1 #88b698);
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
                background-color: #b8865f;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #9d714d;
            }
            QPushButton:pressed {
                background-color: #825c3d;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
        """)
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.save_params_btn.setEnabled(False)
        btn_layout2.addWidget(self.save_params_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #7d6b9d;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #685983;
            }
            QPushButton:pressed {
                background-color: #544769;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
        """)
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        btn_layout2.addWidget(self.export_btn)
        layout.addLayout(btn_layout2)
        
        # Results and Metrics
        results_group = QGroupBox("Results & Metrics")
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
        
        # Metrics display grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)
        metrics_grid.setContentsMargins(10, 10, 10, 10)
        
        # SNR (Total)
        snr_title = QLabel("SNR (Total):")
        snr_title.setStyleSheet("font-size: 11px; color: #757575; font-weight: normal;")
        metrics_grid.addWidget(snr_title, 0, 0)
        self.snr_label = QLabel("--")
        self.snr_label.setStyleSheet("font-size: 13px; color: #424242; font-weight: bold;")
        metrics_grid.addWidget(self.snr_label, 0, 1)
        
        # SNR (Per Scan)
        snr_per_scan_title = QLabel("SNR (Per Scan):")
        snr_per_scan_title.setStyleSheet("font-size: 11px; color: #757575; font-weight: normal;")
        metrics_grid.addWidget(snr_per_scan_title, 1, 0)
        self.snr_per_scan_label = QLabel("--")
        self.snr_per_scan_label.setStyleSheet("font-size: 11px; color: #616161; font-weight: normal;")
        metrics_grid.addWidget(self.snr_per_scan_label, 1, 1)
        
        # Peak height
        peak_title = QLabel("Peak:")
        peak_title.setStyleSheet("font-size: 11px; color: #757575; font-weight: normal;")
        metrics_grid.addWidget(peak_title, 0, 2)
        self.peak_label = QLabel("--")
        self.peak_label.setStyleSheet("font-size: 11px; color: #616161; font-weight: normal;")
        metrics_grid.addWidget(self.peak_label, 0, 3)
        
        # Noise level
        noise_title = QLabel("Noise:")
        noise_title.setStyleSheet("font-size: 11px; color: #757575; font-weight: normal;")
        metrics_grid.addWidget(noise_title, 1, 2)
        self.noise_label = QLabel("--")
        self.noise_label.setStyleSheet("font-size: 11px; color: #616161; font-weight: bold;")
        metrics_grid.addWidget(self.noise_label, 1, 3)
        
        # Scans
        scans_title = QLabel("Scans:")
        scans_title.setStyleSheet("font-size: 11px; color: #757575; font-weight: normal;")
        metrics_grid.addWidget(scans_title, 2, 0)
        self.scans_label = QLabel("--")
        self.scans_label.setStyleSheet("font-size: 11px; color: #616161; font-weight: bold;")
        metrics_grid.addWidget(self.scans_label, 2, 1)
        
        results_layout.addLayout(metrics_grid)
        
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
    
    def create_filter_tab(self):
        """Create filtering tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Savgol filter
        savgol_group = QGroupBox("Savitzky-Golay Filter (Baseline Removal)")
        savgol_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
                color: #1976d2;
            }
        """)
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
        self.conv_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5c7a99;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #4a6580;
            }
        """)
        self.conv_slider.valueChanged.connect(self.on_conv_changed)
        savgol_layout.addWidget(self.conv_slider, row, 1)
        self.conv_spinbox = QSpinBox()
        self.conv_spinbox.setRange(2, 12000)
        self.conv_spinbox.setValue(300)
        self.conv_spinbox.setMinimumWidth(80)
        self.conv_spinbox.setStyleSheet("""
            QSpinBox {
                font-weight: bold;
                color: white;
                background-color: #5c7a99;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #4a6580;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3a5166;
            }
        """)
        self.conv_spinbox.valueChanged.connect(self.on_conv_spinbox_changed)
        savgol_layout.addWidget(self.conv_spinbox, row, 2)
        row += 1
        
        poly_label_title = QLabel("Polynomial Order:")
        poly_label_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        savgol_layout.addWidget(poly_label_title, row, 0)
        self.poly_slider = QSlider(Qt.Horizontal)
        self.poly_slider.setRange(1, 20)
        self.poly_slider.setValue(2)
        self.poly_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5c7a99;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #4a6580;
            }
        """)
        self.poly_slider.valueChanged.connect(self.on_poly_changed)
        savgol_layout.addWidget(self.poly_slider, row, 1)
        self.poly_spinbox = QSpinBox()
        self.poly_spinbox.setRange(1, 20)
        self.poly_spinbox.setValue(2)
        self.poly_spinbox.setMinimumWidth(80)
        self.poly_spinbox.setStyleSheet("""
            QSpinBox {
                font-weight: bold;
                color: white;
                background-color: #5c7a99;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #4a6580;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3a5166;
            }
        """)
        self.poly_spinbox.valueChanged.connect(self.on_poly_spinbox_changed)
        savgol_layout.addWidget(self.poly_spinbox, row, 2)
        row += 1
        
        savgol_group.setLayout(savgol_layout)
        layout.addWidget(savgol_group)
        
        # Truncation
        trunc_group = QGroupBox("Time Domain Truncation")
        trunc_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
                color: #43a047;
            }
        """)
        trunc_layout = QGridLayout()
        trunc_layout.setSpacing(10)
        trunc_layout.setContentsMargins(12, 15, 12, 12)
        
        row = 0
        trunc_start_title = QLabel("Start Points:")
        trunc_start_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        trunc_layout.addWidget(trunc_start_title, row, 0)
        self.trunc_start_slider = QSlider(Qt.Horizontal)
        self.trunc_start_slider.setRange(0, 60000)
        self.trunc_start_slider.setValue(10)
        self.trunc_start_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6b9b7c;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #568266;
            }
        """)
        self.trunc_start_slider.valueChanged.connect(self.on_trunc_start_changed)
        trunc_layout.addWidget(self.trunc_start_slider, row, 1)
        self.trunc_start_spinbox = QSpinBox()
        self.trunc_start_spinbox.setRange(0, 60000)
        self.trunc_start_spinbox.setValue(10)
        self.trunc_start_spinbox.setMinimumWidth(80)
        self.trunc_start_spinbox.setStyleSheet("""
            QSpinBox {
                font-weight: bold;
                color: white;
                background-color: #6b9b7c;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #568266;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #446952;
            }
        """)
        self.trunc_start_spinbox.valueChanged.connect(self.on_trunc_start_spinbox_changed)
        trunc_layout.addWidget(self.trunc_start_spinbox, row, 2)
        row += 1
        
        trunc_end_title = QLabel("End Points:")
        trunc_end_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        trunc_layout.addWidget(trunc_end_title, row, 0)
        self.trunc_end_slider = QSlider(Qt.Horizontal)
        self.trunc_end_slider.setRange(0, 60000)
        self.trunc_end_slider.setValue(10)
        self.trunc_end_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6b9b7c;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #568266;
            }
        """)
        self.trunc_end_slider.valueChanged.connect(self.on_trunc_end_changed)
        trunc_layout.addWidget(self.trunc_end_slider, row, 1)
        self.trunc_end_spinbox = QSpinBox()
        self.trunc_end_spinbox.setRange(0, 60000)
        self.trunc_end_spinbox.setValue(10)
        self.trunc_end_spinbox.setMinimumWidth(80)
        self.trunc_end_spinbox.setStyleSheet("""
            QSpinBox {
                font-weight: bold;
                color: white;
                background-color: #6b9b7c;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #568266;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #446952;
            }
        """)
        self.trunc_end_spinbox.valueChanged.connect(self.on_trunc_end_spinbox_changed)
        trunc_layout.addWidget(self.trunc_end_spinbox, row, 2)
        row += 1
        
        trunc_group.setLayout(trunc_layout)
        layout.addWidget(trunc_group)
        
        # Apodization
        apod_group = QGroupBox("Apodization (T2* Exponential Decay)")
        apod_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
                color: #f57c00;
            }
        """)
        apod_layout = QGridLayout()
        apod_layout.setSpacing(10)
        apod_layout.setContentsMargins(12, 15, 12, 12)
        
        row = 0
        apod_title = QLabel("T2* Factor:")
        apod_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        apod_layout.addWidget(apod_title, row, 0)
        self.apod_slider = QSlider(Qt.Horizontal)
        self.apod_slider.setRange(-200, 200)
        self.apod_slider.setValue(0)
        self.apod_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #b8865f;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #9d714d;
            }
        """)
        self.apod_slider.valueChanged.connect(self.on_apod_changed)
        apod_layout.addWidget(self.apod_slider, row, 1)
        self.apod_spinbox = QDoubleSpinBox()
        self.apod_spinbox.setRange(-2.00, 2.00)
        self.apod_spinbox.setValue(0.00)
        self.apod_spinbox.setSingleStep(0.01)
        self.apod_spinbox.setDecimals(2)
        self.apod_spinbox.setMinimumWidth(80)
        self.apod_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                font-weight: bold;
                color: white;
                background-color: #b8865f;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #9d714d;
                border: none;
                width: 16px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #825c3d;
            }
        """)
        self.apod_spinbox.valueChanged.connect(self.on_apod_spinbox_changed)
        apod_layout.addWidget(self.apod_spinbox, row, 2)
        row += 1
        
        hanning_title = QLabel("Hanning Window:")
        hanning_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        apod_layout.addWidget(hanning_title, row, 0)
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
        apod_layout.addWidget(self.use_hanning, row, 1)
        row += 1
        
        apod_group.setLayout(apod_layout)
        layout.addWidget(apod_group)
        
        layout.addStretch()
        return tab
    
    def create_transform_tab(self):
        """Create transform tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Zero filling
        zf_group = QGroupBox("Zero Filling")
        zf_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
                color: #5e35b1;
            }
        """)
        zf_layout = QGridLayout()
        zf_layout.setSpacing(10)
        zf_layout.setContentsMargins(12, 15, 12, 12)
        
        row = 0
        zf_title = QLabel("Zero Fill Factor:")
        zf_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        zf_layout.addWidget(zf_title, row, 0)
        self.zf_slider = QSlider(Qt.Horizontal)
        self.zf_slider.setRange(0, 1000)
        self.zf_slider.setValue(0)
        self.zf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #7d6b9d;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #685983;
            }
        """)
        self.zf_slider.valueChanged.connect(self.on_zf_changed)
        zf_layout.addWidget(self.zf_slider, row, 1)
        self.zf_spinbox = QDoubleSpinBox()
        self.zf_spinbox.setRange(0.00, 10.00)
        self.zf_spinbox.setValue(0.00)
        self.zf_spinbox.setSingleStep(0.01)
        self.zf_spinbox.setDecimals(2)
        self.zf_spinbox.setMinimumWidth(80)
        self.zf_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                font-weight: bold;
                color: white;
                background-color: #7d6b9d;
                padding: 4px 8px;
                border-radius: 4px;
                border: none;
                font-size: 11px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #685983;
                border: none;
                width: 16px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #544769;
            }
        """)
        self.zf_spinbox.valueChanged.connect(self.on_zf_spinbox_changed)
        zf_layout.addWidget(self.zf_spinbox, row, 2)
        row += 1
        
        zf_hint = QLabel("0 = No filling, 2.7 = 2.7x data length")
        zf_hint.setStyleSheet("font-size: 9px; color: #757575; font-style: italic;")
        zf_layout.addWidget(zf_hint, row, 0, 1, 3)
        
        zf_group.setLayout(zf_layout)
        layout.addWidget(zf_group)
        
        # FFT info
        fft_group = QGroupBox("Fourier Transform (FFT)")
        fft_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
        fft_layout = QVBoxLayout()
        fft_layout.setContentsMargins(12, 10, 12, 12)
        fft_info = QLabel("FFT is automatically applied after all processing steps.\n"
                         "The spectrum is computed using scipy.fft.fft().")
        fft_info.setWordWrap(True)
        fft_info.setStyleSheet("color: #757575; font-size: 10px;")
        fft_layout.addWidget(fft_info)
        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)
        
        # Frequency range controls
        freq_group = QGroupBox("Frequency Display Range")
        freq_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
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
                color: #00897b;
            }
        """)
        freq_layout = QGridLayout()
        freq_layout.setSpacing(10)
        freq_layout.setContentsMargins(12, 15, 12, 12)
        
        low_freq_title = QLabel("Low Frequency View:")
        low_freq_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        freq_layout.addWidget(low_freq_title, 0, 0)
        self.freq_low_min = QDoubleSpinBox()
        self.freq_low_min.setRange(0, 1000)
        self.freq_low_min.setValue(0)
        self.freq_low_min.setSuffix(" Hz")
        self.freq_low_min.setStyleSheet("font-size: 10px; padding: 4px;")
        freq_layout.addWidget(self.freq_low_min, 0, 1)
        self.freq_low_max = QDoubleSpinBox()
        self.freq_low_max.setRange(0, 1000)
        self.freq_low_max.setValue(30)
        self.freq_low_max.setSuffix(" Hz")
        self.freq_low_max.setStyleSheet("font-size: 10px; padding: 4px;")
        freq_layout.addWidget(self.freq_low_max, 0, 2)
        
        high_freq_title = QLabel("High Frequency View:")
        high_freq_title.setStyleSheet("font-size: 10px; color: #424242; font-weight: bold;")
        freq_layout.addWidget(high_freq_title, 1, 0)
        self.freq_high_min = QDoubleSpinBox()
        self.freq_high_min.setRange(0, 1000)
        self.freq_high_min.setValue(100)
        self.freq_high_min.setSuffix(" Hz")
        self.freq_high_min.setStyleSheet("font-size: 10px; padding: 4px;")
        freq_layout.addWidget(self.freq_high_min, 1, 1)
        self.freq_high_max = QDoubleSpinBox()
        self.freq_high_max.setRange(0, 1000)
        self.freq_high_max.setValue(275)
        self.freq_high_max.setSuffix(" Hz")
        self.freq_high_max.setStyleSheet("font-size: 10px; padding: 4px;")
        freq_layout.addWidget(self.freq_high_max, 1, 2)
        
        apply_range_btn = QPushButton("Update View")
        apply_range_btn.setStyleSheet("""
            QPushButton {
                background-color: #00897b;
                color: white;
                padding: 8px;
                font-weight: bold;
                font-size: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00796b;
            }
            QPushButton:pressed {
                background-color: #00695c;
            }
        """)
        apply_range_btn.clicked.connect(self.plot_results)
        freq_layout.addWidget(apply_range_btn, 2, 0, 1, 3)
        
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # SNR Calculation Settings
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
                background-color: #1976d2;
                color: white;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1565c0;
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
                background-color: #1976d2;
                color: white;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1565c0;
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
                background-color: #1976d2;
                color: white;
                padding: 4px 12px;
                font-size: 9px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1565c0;
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
        self.schedule_processing()
    
    @Slot()
    def on_trunc_start_spinbox_changed(self, value):
        self.trunc_start_slider.blockSignals(True)
        self.trunc_start_slider.setValue(int(value))
        self.trunc_start_slider.blockSignals(False)
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
            
            self.scans_label.setText(str(self.scan_count))
            
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
            'apod_t2star': self.apod_slider.value() / 100.0,
            'freq_low_min': self.freq_low_min.value(),
            'freq_low_max': self.freq_low_max.value(),
            'freq_high_min': self.freq_high_min.value(),
            'freq_high_max': self.freq_high_max.value()
        }
        
        # Update params_b if using same parameters
        if self.use_same_params:
            self.params_b = self.params.copy()
        
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
        
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.plot_results()
        self.calculate_metrics()
        self.export_btn.setEnabled(True)
    
    @Slot(object)
    def on_processing_b_finished(self, result):
        """Handle Data B processing finished"""
        self.processed_b = result
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.plot_results()
        self.calculate_metrics()
        self.export_btn.setEnabled(True)
    
    @Slot(str)
    def on_processing_error(self, error):
        """Handle processing error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Processing Error", f"Failed to process:\n{error}")
    
    def plot_results(self):
        """Plot processing results"""
        if self.processed is None:
            return
        
        # Check if in comparison mode with overlay
        if self.comparison_mode and self.processed_b is not None and self.display_overlay.isChecked():
            self.plot_overlay_comparison()
            return
        
        time_data = self.processed['time_data']
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        acq_time = self.processed['acq_time_effective']
        
        # Time domain
        time_axis = np.linspace(0, acq_time, len(time_data))
        self.time_canvas.axes.clear()
        self.time_canvas.axes.plot(time_axis, np.real(time_data), 'b-', linewidth=0.8, alpha=0.8, label='Data A')
        
        # Add Data B if in comparison mode and side-by-side
        if self.comparison_mode and self.processed_b is not None:
            time_data_b = self.processed_b['time_data']
            acq_time_b = self.processed_b['acq_time_effective']
            time_axis_b = np.linspace(0, acq_time_b, len(time_data_b))
            self.time_canvas.axes.plot(time_axis_b, np.real(time_data_b), 'r-', linewidth=0.8, alpha=0.6, label='Data B')
            self.time_canvas.axes.legend(fontsize=9)
        
        self.time_canvas.axes.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        self.time_canvas.axes.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        title = 'Time Domain Signal - Comparison' if self.comparison_mode else 'Time Domain Signal (After Processing)'
        self.time_canvas.axes.set_title(title, fontsize=11, fontweight='bold')
        self.time_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.time_canvas.axes.autoscale(enable=True, axis='y', tight=False)
        self.time_canvas.fig.tight_layout()
        self.time_canvas.draw()
        
        # Frequency domain - low freq
        freq_range_low = [self.freq_low_min.value(), self.freq_low_max.value()]
        
        self.freq1_canvas.axes.clear()
        self.freq1_canvas.axes.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        
        # Add Data B
        if self.comparison_mode and self.processed_b is not None:
            freq_axis_b = self.processed_b['freq_axis']
            spectrum_b = self.processed_b['spectrum']
            self.freq1_canvas.axes.plot(freq_axis_b, np.abs(spectrum_b), 'r-', linewidth=1.0, alpha=0.6, label='Data B')
            self.freq1_canvas.axes.legend(fontsize=9)
        
        self.freq1_canvas.axes.set_xlim(freq_range_low[0], freq_range_low[1])
        idx_visible = (freq_axis >= freq_range_low[0]) & (freq_axis <= freq_range_low[1])
        if np.any(idx_visible):
            y_visible = np.abs(spectrum)[idx_visible]
            y_max = np.max(y_visible)
            self.freq1_canvas.axes.set_ylim(-0.05 * y_max, 1.1 * y_max)
        self.freq1_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq1_canvas.axes.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        title = f'Low Freq Comparison (View: {freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)' if self.comparison_mode \
                else f'Low Frequency Spectrum (View: {freq_range_low[0]:.0f}-{freq_range_low[1]:.0f} Hz)'
        self.freq1_canvas.axes.set_title(title, fontsize=11, fontweight='bold')
        self.freq1_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq1_canvas.fig.tight_layout()
        self.freq1_canvas.draw()
        
        # Frequency domain - high freq
        freq_range_high = [self.freq_high_min.value(), self.freq_high_max.value()]
        
        self.freq2_canvas.axes.clear()
        self.freq2_canvas.axes.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.0, alpha=0.8, label='Data A')
        
        # Add Data B
        if self.comparison_mode and self.processed_b is not None:
            freq_axis_b = self.processed_b['freq_axis']
            spectrum_b = self.processed_b['spectrum']
            self.freq2_canvas.axes.plot(freq_axis_b, np.abs(spectrum_b), 'r-', linewidth=1.0, alpha=0.6, label='Data B')
            self.freq2_canvas.axes.legend(fontsize=9)
        
        self.freq2_canvas.axes.set_xlim(freq_range_high[0], freq_range_high[1])
        idx_visible = (freq_axis >= freq_range_high[0]) & (freq_axis <= freq_range_high[1])
        if np.any(idx_visible):
            y_visible = np.abs(spectrum)[idx_visible]
            y_max = np.max(y_visible)
            self.freq2_canvas.axes.set_ylim(-0.05 * y_max, 1.1 * y_max)
        self.freq2_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.freq2_canvas.axes.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        title = f'High Freq Comparison (View: {freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)' if self.comparison_mode \
                else f'High Frequency Spectrum (View: {freq_range_high[0]:.0f}-{freq_range_high[1]:.0f} Hz)'
        self.freq2_canvas.axes.set_title(title, fontsize=11, fontweight='bold')
        self.freq2_canvas.axes.grid(True, alpha=0.3, linestyle='--')
        self.freq2_canvas.fig.tight_layout()
        self.freq2_canvas.draw()
    
    def plot_overlay_comparison(self):
        """Plot Data A and B overlaid on same axes"""
        # This is called when overlay mode is selected
        # For now, just call plot_results which already handles overlay
        pass
    
    def calculate_metrics(self):
        """Calculate and display metrics"""
        if self.processed is None:
            return
        
        freq_axis = self.processed['freq_axis']
        spectrum = self.processed['spectrum']
        spectrum_abs = np.abs(spectrum)
        
        results = []
        results.append("=== Processing Pipeline ===")
        results.append(f"[OK] Savgol Filter: window={self.params['conv_points']}, poly={self.params['poly_order']}")
        results.append(f"[OK] Truncation: start={self.params['trunc_start']}, end={self.params['trunc_end']}")
        results.append(f"[OK] Apodization T2*: {self.params['apod_t2star']:.2f}")
        results.append(f"[OK] Hanning: {'Yes' if self.params['use_hanning'] else 'No'}")
        results.append(f"[OK] Zero Fill Factor: {self.params['zf_factor']:.2f}")
        
        # SNR calculation
        if HAS_NMRDUINO:
            try:
                # Use user-defined ranges from UI
                frequency_range_snr = [self.signal_range_min.value(), self.signal_range_max.value()]
                noise_range_snr = [self.noise_range_min.value(), self.noise_range_max.value()]
                
                # Calculate peak height from signal region
                signal_idx = (freq_axis >= frequency_range_snr[0]) & (freq_axis <= frequency_range_snr[1])
                if np.any(signal_idx):
                    peak_height = np.max(spectrum_abs[signal_idx])
                else:
                    peak_height = np.max(spectrum_abs)  # Fallback to global max
                self.peak_label.setText(f"{peak_height:.1f}")
                
                snr = nmr_util.snr_calc(freq_axis, spectrum_abs, 
                                       frequency_range_snr, noise_range_snr)
                
                # Calculate noise level
                noise_idx = (freq_axis >= noise_range_snr[0]) & (freq_axis <= noise_range_snr[1])
                noise_level = np.std(spectrum_abs[noise_idx])
                self.noise_label.setText(f"{noise_level:.2f}")
                
                results.append("\n═══ Quality Metrics ═══")
                
                # Display total SNR (from averaged data)
                self.snr_label.setText(f"{snr:.1f}")
                results.append(f"SNR (total, {self.scan_count} scans): {snr:.2f}")
                
                # Calculate and display per-scan SNR
                if self.scan_count > 1:
                    snr_per_scan = snr / np.sqrt(self.scan_count)
                    self.snr_per_scan_label.setText(f"{snr_per_scan:.2f}")
                    results.append(f"SNR (estimated per scan): {snr_per_scan:.2f}")
                else:
                    self.snr_per_scan_label.setText(f"{snr:.2f}")
                    results.append(f"SNR (single scan): {snr:.2f}")
                
                results.append(f"Peak Height: {peak_height:.2f}")
                results.append(f"Noise Level: {noise_level:.2f}")
                results.append(f"Signal Range: {frequency_range_snr}")
                results.append(f"Noise Range: {noise_range_snr}")
                
            except Exception as e:
                self.snr_label.setText("Error")
                self.snr_per_scan_label.setText("--")
                self.noise_label.setText("--")
                # Still show peak height
                peak_height = np.max(spectrum_abs)
                self.peak_label.setText(f"{peak_height:.1f}")
                results.append(f"\n[ERROR] SNR calculation failed: {e}")
        else:
            self.snr_label.setText("N/A")
            self.snr_per_scan_label.setText("--")
            self.noise_label.setText("--")
            # Show global peak height as fallback
            peak_height = np.max(spectrum_abs)
            self.peak_label.setText(f"{peak_height:.1f}")
            results.append("\n[WARNING] nmrduino_util not available for SNR calculation")
        
        # Calculate metrics for Data B if in comparison mode
        if self.comparison_mode and self.processed_b is not None:
            results.append("\n\n═══ Data B Metrics ═══")
            freq_axis_b = self.processed_b['freq_axis']
            spectrum_b = self.processed_b['spectrum']
            spectrum_abs_b = np.abs(spectrum_b)
            
            if HAS_NMRDUINO:
                try:
                    # Calculate peak height from signal region
                    signal_idx_b = (freq_axis_b >= frequency_range_snr[0]) & (freq_axis_b <= frequency_range_snr[1])
                    if np.any(signal_idx_b):
                        peak_height_b = np.max(spectrum_abs_b[signal_idx_b])
                    else:
                        peak_height_b = np.max(spectrum_abs_b)
                    
                    snr_b = nmr_util.snr_calc(freq_axis_b, spectrum_abs_b, 
                                           frequency_range_snr, noise_range_snr)
                    
                    noise_idx_b = (freq_axis_b >= noise_range_snr[0]) & (freq_axis_b <= noise_range_snr[1])
                    noise_level_b = np.std(spectrum_abs_b[noise_idx_b])
                    
                    results.append(f"SNR (total, {self.scan_count_b} scans): {snr_b:.2f}")
                    if self.scan_count_b > 1:
                        snr_per_scan_b = snr_b / np.sqrt(self.scan_count_b)
                        results.append(f"SNR (estimated per scan): {snr_per_scan_b:.2f}")
                    results.append(f"Peak Height: {peak_height_b:.2f}")
                    results.append(f"Noise Level: {noise_level_b:.2f}")
                    
                    # Show comparison
                    results.append("\n═══ Comparison (A vs B) ═══")
                    snr_diff = snr - snr_b
                    snr_ratio = snr / snr_b if snr_b > 0 else float('inf')
                    results.append(f"SNR Difference (A-B): {snr_diff:+.2f}")
                    results.append(f"SNR Ratio (A/B): {snr_ratio:.2f}x")
                    results.append(f"Peak Difference (A-B): {(peak_height - peak_height_b):+.2f}")
                    
                except Exception as e:
                    results.append(f"[ERROR] Data B metrics calculation failed: {e}")
        
        self.results_text.setText('\n'.join(results))
    
    @Slot()
    def save_parameters(self):
        """Save current parameters to JSON"""
        if self.current_path is None:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Parameters",
                "", "JSON files (*.json)"
            )
            if not filepath:
                return
        else:
            filepath = os.path.join(self.current_path, "processing_params.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.params, f, indent=2)
            
            QMessageBox.information(self, "Saved", f"Parameters saved to:\n{filepath}")
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
            
            # Update frequency display range sliders if present
            if 'freq_low_min' in params:
                self.freq_low_min.setValue(float(params['freq_low_min']))
            if 'freq_low_max' in params:
                self.freq_low_max.setValue(float(params['freq_low_max']))
            if 'freq_high_min' in params:
                self.freq_high_min.setValue(float(params['freq_high_min']))
            if 'freq_high_max' in params:
                self.freq_high_max.setValue(float(params['freq_high_max']))
            
            self.params = params
            
            QMessageBox.information(self, "Loaded", f"Parameters loaded from:\n{filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load parameters:\n{e}")
    
    @Slot()
    def export_results(self):
        """Export processing results"""
        if self.processed is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "nmr_processed.npz",
            "NumPy archive (*.npz);;All files (*.*)"
        )
        
        if filename:
            try:
                np.savez(
                    filename,
                    time_data=self.processed['time_data'],
                    freq_axis=self.processed['freq_axis'],
                    spectrum=self.processed['spectrum'],
                    sampling_rate=self.sampling_rate,
                    acquisition_time=self.acq_time,
                    parameters=self.params
                )
                
                QMessageBox.information(self, "Export Successful", f"Results exported to:\n{filename}")
            
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Cannot export:\n{e}")
    
    def export_figures_svg(self):
        """Export all figures as SVG files"""
        if self.processed is None:
            QMessageBox.warning(self, "No Data", "Please process data first before exporting figures.")
            return
        
        # Ask user to select directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for SVG Export",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not directory:
            return
        
        try:
            import os
            
            # Export time domain plot
            time_path = os.path.join(directory, "time_domain.svg")
            self.time_canvas.fig.savefig(time_path, format='svg', bbox_inches='tight', dpi=300)
            
            # Export low frequency plot
            freq1_path = os.path.join(directory, "freq_low.svg")
            self.freq1_canvas.fig.savefig(freq1_path, format='svg', bbox_inches='tight', dpi=300)
            
            # Export high frequency plot
            freq2_path = os.path.join(directory, "freq_high.svg")
            self.freq2_canvas.fig.savefig(freq2_path, format='svg', bbox_inches='tight', dpi=300)
            
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Figures exported to:\n{directory}\n\nFiles:\n- time_domain.svg\n- freq_low.svg\n- freq_high.svg"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Cannot export figures:\n{e}")
    
    def maximize_plot(self, plot_type):
        """Maximize a specific plot in a new window"""
        dialog = QWidget()
        dialog.setWindowTitle(f"Maximized View - {plot_type.upper()}")
        dialog.setGeometry(100, 100, 1200, 800)
        dialog.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create a new canvas with larger size
        canvas = MplCanvas(dialog, width=12, height=8, dpi=100)
        toolbar = NavigationToolbar(canvas, dialog)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Copy the plot
        if plot_type == 'time' and self.processed is not None:
            time_data = self.processed['time_data']
            acq_time = self.processed['acq_time_effective']
            time_axis = np.linspace(0, acq_time, len(time_data))
            
            canvas.axes.clear()
            canvas.axes.plot(time_axis, np.real(time_data), 'k-', linewidth=0.8, alpha=0.8)
            canvas.axes.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            canvas.axes.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            canvas.axes.set_title('Time Domain Signal (Maximized View)', fontsize=14, fontweight='bold')
            canvas.axes.grid(True, alpha=0.3, linestyle='--')
            canvas.axes.autoscale(enable=True, axis='y', tight=False)  # Auto-scale Y axis
            canvas.fig.tight_layout()
            canvas.draw()
            
        elif plot_type == 'freq1' and self.processed is not None:
            freq_axis = self.processed['freq_axis']
            spectrum = self.processed['spectrum']
            freq_range = [self.freq_low_min.value(), self.freq_low_max.value()]
            
            canvas.axes.clear()
            canvas.axes.plot(freq_axis, np.abs(spectrum), 'b-', linewidth=1.2)
            canvas.axes.set_xlim(freq_range[0], freq_range[1])  # Set initial view range
            # Auto-scale Y based on visible data in the X range
            idx_visible = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
            if np.any(idx_visible):
                y_visible = np.abs(spectrum)[idx_visible]
                y_max = np.max(y_visible)
                canvas.axes.set_ylim(-0.05 * y_max, 1.1 * y_max)
            canvas.axes.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            canvas.axes.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            canvas.axes.set_title(f'Low Frequency Spectrum (Initial View: {freq_range[0]:.0f}-{freq_range[1]:.0f} Hz)', 
                                 fontsize=14, fontweight='bold')
            canvas.axes.grid(True, alpha=0.3, linestyle='--')
            canvas.fig.tight_layout()
            canvas.draw()
            
        elif plot_type == 'freq2' and self.processed is not None:
            freq_axis = self.processed['freq_axis']
            spectrum = self.processed['spectrum']
            freq_range = [self.freq_high_min.value(), self.freq_high_max.value()]
            
            canvas.axes.clear()
            canvas.axes.plot(freq_axis, np.abs(spectrum), 'r-', linewidth=1.2)
            canvas.axes.set_xlim(freq_range[0], freq_range[1])  # Set initial view range
            # Auto-scale Y based on visible data in the X range
            idx_visible = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
            if np.any(idx_visible):
                y_visible = np.abs(spectrum)[idx_visible]
                y_max = np.max(y_visible)
                canvas.axes.set_ylim(-0.05 * y_max, 1.1 * y_max)
            canvas.axes.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            canvas.axes.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            canvas.axes.set_title(f'High Frequency Spectrum (Initial View: {freq_range[0]:.0f}-{freq_range[1]:.0f} Hz)', 
                                 fontsize=14, fontweight='bold')
            canvas.axes.grid(True, alpha=0.3, linestyle='--')
            canvas.fig.tight_layout()
            canvas.draw()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.show()
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_maximized_windows'):
            self._maximized_windows = []
        self._maximized_windows.append(dialog)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
            if self.worker.isRunning():
                self.worker.terminate()
        
        # Close all maximized windows
        if hasattr(self, '_maximized_windows'):
            for window in self._maximized_windows:
                window.close()
        
        # Save window state
        self.save_window_state()
        
        event.accept()
    
    def toggle_comparison_mode(self, checked):
        """Toggle between single and comparison mode"""
        self.comparison_mode = checked
        
        if checked:
            self.comparison_mode_action.setText('Disable Comparison Mode')
            # Show comparison UI elements
            if hasattr(self, 'data_b_group'):
                self.data_b_group.setVisible(True)
            if hasattr(self, 'comparison_controls_group'):
                self.comparison_controls_group.setVisible(True)
            QMessageBox.information(self, "Comparison Mode", 
                                   "Comparison mode enabled!\n\n"
                                   "You can now:\n"
                                   "1. Load a second dataset (Data B)\n"
                                   "2. Choose to use same or independent parameters\n"
                                   "3. View side-by-side or overlaid comparison")
        else:
            self.comparison_mode_action.setText('Enable Comparison Mode')
            # Hide comparison UI elements
            if hasattr(self, 'data_b_group'):
                self.data_b_group.setVisible(False)
            if hasattr(self, 'comparison_controls_group'):
                self.comparison_controls_group.setVisible(False)
    
    def save_window_state(self):
        """Save window geometry and splitter states"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('main_splitter', self.main_splitter.saveState())
        self.settings.setValue('plot_splitter', self.plot_splitter.saveState())
    
    def restore_window_state(self):
        """Restore window geometry and splitter states"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        main_splitter_state = self.settings.value('main_splitter')
        if main_splitter_state:
            self.main_splitter.restoreState(main_splitter_state)
        
        plot_splitter_state = self.settings.value('plot_splitter')
        if plot_splitter_state:
            self.plot_splitter.restoreState(plot_splitter_state)
    
    def load_folder_b(self):
        """Load Data B for comparison"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data B Folder")
        if not folder:
            return
        
        try:
            self.current_path_b = folder
            
            if HAS_NMRDUINO:
                # Try to load compiled data
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp_b = np.load(compiled_path)
                    self.sampling_rate_b = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time_b = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                    self.scan_count_b = nmr_util.scan_number_extraction(folder)
                else:
                    # Load and compile
                    compiled = nmr_util.nmrduino_dat_interp(folder, 0)
                    self.halp_b = compiled[0]
                    self.sampling_rate_b = compiled[1]
                    self.acq_time_b = compiled[2]
                    self.scan_count_b = nmr_util.scan_number_extraction(folder)
                    
                    # Save compiled
                    np.save(compiled_path, self.halp_b)
                    np.save(os.path.join(folder, "sampling_rate_compiled.npy"), self.sampling_rate_b)
                    np.save(os.path.join(folder, "acq_time_compiled.npy"), self.acq_time_b)
                    np.save(os.path.join(folder, "scan_count.npy"), self.scan_count_b)
            else:
                # Manual loading
                compiled_path = os.path.join(folder, "halp_compiled.npy")
                if os.path.exists(compiled_path):
                    self.halp_b = np.load(compiled_path)
                    self.sampling_rate_b = np.load(os.path.join(folder, "sampling_rate_compiled.npy"))
                    self.acq_time_b = np.load(os.path.join(folder, "acq_time_compiled.npy"))
                    
                    # Load scan count if available
                    scan_count_path = os.path.join(folder, "scan_count.npy")
                    if os.path.exists(scan_count_path):
                        self.scan_count_b = int(np.load(scan_count_path))
                    else:
                        dat_files = [f for f in os.listdir(folder) if f.endswith('.dat')]
                        self.scan_count_b = len(dat_files) if dat_files else 1
                else:
                    raise FileNotFoundError("No compiled data found. Please compile first or install nmrduino_util.")
            
            self.data_b_info.setText(
                f"<b>Loaded:</b> {os.path.basename(folder)}<br>"
                f"<b>Points:</b> {len(self.halp_b)}<br>"
                f"<b>Sampling:</b> {self.sampling_rate_b:.1f} Hz<br>"
                f"<b>Acq Time:</b> {self.acq_time_b:.3f} s<br>"
                f"<b>Scans:</b> {self.scan_count_b}"
            )
            
            QMessageBox.information(self, "Data B Loaded", 
                                   f"Data B loaded successfully!\n\n"
                                   f"Folder: {os.path.basename(folder)}\n"
                                   f"Points: {len(self.halp_b)}\n"
                                   f"Scans: {self.scan_count_b}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load Data B:\n{e}")
    
    def on_same_params_changed(self, state):
        """Handle same parameters checkbox change"""
        self.use_same_params = (state == Qt.Checked)
        if self.use_same_params:
            # Copy params A to params B
            self.params_b = self.params.copy()
    
    def apply_comparison(self):
        """Apply comparison and update plots"""
        if self.halp is None:
            QMessageBox.warning(self, "No Data", "Please load Data A first!")
            return
        
        if self.halp_b is None:
            QMessageBox.warning(self, "No Data B", "Please load Data B for comparison!")
            return
        
        # Process both datasets
        self.process_data()


def main():
    app = QApplication(sys.argv)
    
    # Set font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Set style
    app.setStyle('Fusion')
    
    window = EnhancedNMRProcessingUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
