import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QRadioButton, QFileDialog, QSplitter,
    QScrollArea, QFrame, QTableWidget, QHeaderView, QTableWidgetItem,
    QTextEdit, QProgressBar, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QObject
from PySide6.QtGui import QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import scipy.signal
from scipy.fft import fft

# Import library components
try:
    from nmr_processing_lib.utils.realtime_monitor import RealtimeDataMonitor
    from nmr_processing_lib.processing.zulf_algorithms import (
        backward_linear_prediction, apply_phase_correction
    )
    from nmr_processing_lib.processing.filtering import svd_denoising
except ImportError:
    # Fallbacks for standalone testing without full lib
    RealtimeDataMonitor = None
    def backward_linear_prediction(data, n, order, train_len=None): return data
    def apply_phase_correction(spec, p0, p1, pivot_index=None): return spec
    def svd_denoising(data, rank): return data

class LiveMonitorSignals(QObject):
    data_received = Signal(object, int)
    error_occurred = Signal(str)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()

class ProcessingWorker(QThread):
    finished = Signal(object)
    
    def __init__(self, halp, sampling_rate, acq_time, params):
        super().__init__()
        self.halp = halp
        self.sampling_rate = sampling_rate
        self.acq_time = acq_time
        self.params = params
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        if not self._running or self.halp is None: return
        
        try:
            # 1. Savgol
            smooth_svd = scipy.signal.savgol_filter(
                self.halp, 
                int(self.params['conv_points']),
                int(self.params['poly_order']), 
                mode="mirror"
            )
            data = self.halp - smooth_svd
            
            # 2. SVD
            if self.params['enable_svd']:
                data = svd_denoising(data, int(self.params['svd_rank']))
                
            # 3. Truncation
            start = int(self.params['trunc_start'])
            end = int(self.params['trunc_end'])
            if start > 0: data = data[start:]
            
            # 4. LP
            n_backward = 0
            lp_train_len = 0
            if self.params['enable_recon']:
                n_backward = int(self.params['recon_points'])
                order = int(self.params['recon_order'])
                train_len = int(self.params['recon_train_len'])
                if n_backward > 0:
                    train_len = min(len(data), train_len)
                    data = backward_linear_prediction(data, n_backward, order, train_len=train_len)
                    lp_train_len = train_len
            
            if end > 0: data = data[:-end]
            
            # 5. Apodization
            acq_time_eff = self.acq_time * (len(data) / len(self.halp))
            t = np.linspace(0, acq_time_eff, len(data))
            data = data * np.exp(-self.params['apod_t2star'] * t)
            
            if self.params['use_hanning']:
                data = data * np.hanning(len(data))
                
            # 6. Zero Fill
            zf = self.params['zf_factor']
            if zf > 0:
                target_len = int(len(data) * (1 + zf))
                data = np.concatenate((data, np.zeros(target_len - len(data))))
                
            # 7. FFT
            yf = fft(data)
            xf = np.linspace(0, self.sampling_rate, len(yf))
            
            # 8. Phase
            pivot = len(yf) // 2
            yf_phased = apply_phase_correction(yf, self.params['phase0'], self.params['phase1'], pivot_index=pivot)
            
            self.finished.emit({
                'time_data': data,
                'freq_axis': xf,
                'spectrum': yf_phased,
                'n_backward': n_backward,
                'lp_train_len': lp_train_len
            })
            
        except Exception as e:
            print(f"Processing error: {e}")

class LiveMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Data Processor")
        self.resize(1400, 900)
        
        self.monitor = None
        self.is_monitoring = False
        self.live_signals = LiveMonitorSignals()
        self.live_signals.data_received.connect(self.handle_live_data)
        
        self.halp = None
        self.sampling_rate = 6000
        self.acq_time = 1.0
        self.scan_count = 0
        self.worker = None
        self.last_result = None
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(380)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Monitor Control
        gb_mon = QGroupBox("Monitor Control")
        gb_mon.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #d32f2f; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #d32f2f; }")
        l_mon = QVBoxLayout()
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        l_mon.addWidget(self.folder_label)
        
        btn_browse = QPushButton("Select Folder")
        btn_browse.clicked.connect(self.select_folder)
        l_mon.addWidget(btn_browse)
        
        h_mode = QHBoxLayout()
        self.rb_avg = QRadioButton("Average")
        self.rb_avg.setChecked(True)
        self.rb_single = QRadioButton("Single")
        self.rb_avg.toggled.connect(self.update_mode)
        h_mode.addWidget(self.rb_avg)
        h_mode.addWidget(self.rb_single)
        l_mon.addLayout(h_mode)
        
        self.chk_existing = QCheckBox("Process Existing Files")
        self.chk_existing.setChecked(True)
        l_mon.addWidget(self.chk_existing)
        
        self.btn_start = QPushButton("START MONITORING")
        self.btn_start.setCheckable(True)
        self.btn_start.setStyleSheet("QPushButton:checked { background-color: #b71c1c; color: white; }")
        self.btn_start.clicked.connect(self.toggle_monitoring)
        self.btn_start.setEnabled(False)
        l_mon.addWidget(self.btn_start)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        l_mon.addWidget(self.lbl_status)
        
        gb_mon.setLayout(l_mon)
        sidebar_layout.addWidget(gb_mon)
        
        # Parameters Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Visualization Settings
        self.create_visualization_group(params_layout)
        
        # Savgol
        self.spin_conv, self.slider_conv = self.add_slider_param(params_layout, "Window", 300, 10, 2000)
        self.spin_poly, self.slider_poly = self.add_slider_param(params_layout, "Poly Order", 2, 1, 10)
        
        # SVD
        gb_svd = QGroupBox("SVD Denoising")
        l_svd = QVBoxLayout()
        self.chk_svd = QCheckBox("Enable SVD")
        l_svd.addWidget(self.chk_svd)
        self.spin_rank, self.slider_rank = self.add_slider_param(l_svd, "Rank", 5, 1, 50, group=False)
        gb_svd.setLayout(l_svd)
        params_layout.addWidget(gb_svd)
        
        # Truncation
        self.spin_trunc_start, self.slider_trunc_start = self.add_slider_param(params_layout, "Trunc Start", 10, 0, 5000)
        self.spin_trunc_end, self.slider_trunc_end = self.add_slider_param(params_layout, "Trunc End", 10, 0, 5000)
        
        # LP
        gb_lp = QGroupBox("Backward LP")
        l_lp = QVBoxLayout()
        self.chk_lp = QCheckBox("Enable LP")
        l_lp.addWidget(self.chk_lp)
        self.spin_lp_points, self.slider_lp_points = self.add_slider_param(l_lp, "Points", 0, 0, 1000, group=False)
        self.spin_lp_order, self.slider_lp_order = self.add_slider_param(l_lp, "Order", 10, 1, 100, group=False)
        self.spin_lp_train, self.slider_lp_train = self.add_slider_param(l_lp, "Train Len", 40, 10, 2000, group=False)
        gb_lp.setLayout(l_lp)
        params_layout.addWidget(gb_lp)
        
        # Apodization
        self.spin_t2, self.slider_t2 = self.add_float_slider_param(params_layout, "T2*", 0.0, 0.0, 10.0, 100)
        self.chk_hanning = QCheckBox("Hanning Window")
        params_layout.addWidget(self.chk_hanning)
        self.spin_zf, self.slider_zf = self.add_float_slider_param(params_layout, "Zero Fill", 0.0, 0.0, 5.0, 10)
        
        # Phase
        self.spin_ph0, self.slider_ph0 = self.add_float_slider_param(params_layout, "Phase 0", 0.0, -360, 360, 1)
        self.spin_ph1, self.slider_ph1 = self.add_float_slider_param(params_layout, "Phase 1", 0.0, -360, 360, 1)
        
        # SNR Calculation Range
        self.create_snr_group(params_layout)
        
        # Results and Metrics
        self.create_results_group(params_layout)
        
        scroll.setWidget(params_widget)
        sidebar_layout.addWidget(scroll)
        main_layout.addWidget(sidebar)
        
        # --- Plots ---
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Time Domain
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        self.canvas_time = MplCanvas(self, width=5, height=4)
        self.toolbar_time = NavigationToolbar(self.canvas_time, self)
        time_layout.addWidget(self.toolbar_time)
        time_layout.addWidget(self.canvas_time)
        self.ax_time = self.canvas_time.axes
        splitter.addWidget(time_widget)
        
        # Frequency Domain
        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        self.canvas_freq = MplCanvas(self, width=5, height=4)
        self.toolbar_freq = NavigationToolbar(self.canvas_freq, self)
        freq_layout.addWidget(self.toolbar_freq)
        freq_layout.addWidget(self.canvas_freq)
        self.ax_freq = self.canvas_freq.axes
        splitter.addWidget(freq_widget)
        
        plot_layout.addWidget(splitter)
        main_layout.addWidget(plot_area)
        
        # Connect signals
        self.timer_process = QTimer()
        self.timer_process.setSingleShot(True)
        self.timer_process.timeout.connect(self.process_data)
        
        for chk in [self.chk_svd, self.chk_lp, self.chk_hanning]:
            chk.stateChanged.connect(self.schedule_process)

    def create_visualization_group(self, layout):
        gb = QGroupBox("Visualization Settings")
        gb.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #00897b; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00897b; }")
        v_layout = QVBoxLayout()
        
        # Display Mode
        h_disp = QHBoxLayout()
        self.view_real = QRadioButton("Real")
        self.view_real.setChecked(True)
        self.view_imag = QRadioButton("Imag")
        self.view_mag = QRadioButton("Mag")
        
        self.view_real.toggled.connect(self.update_plot_view)
        self.view_imag.toggled.connect(self.update_plot_view)
        self.view_mag.toggled.connect(self.update_plot_view)
        
        h_disp.addWidget(self.view_real)
        h_disp.addWidget(self.view_imag)
        h_disp.addWidget(self.view_mag)
        
        self.show_absolute = QCheckBox("Abs")
        self.show_absolute.stateChanged.connect(self.update_plot_view)
        h_disp.addWidget(self.show_absolute)
        
        v_layout.addLayout(h_disp)
        
        # Frequency Range
        grid = QGridLayout()
        grid.addWidget(QLabel("Low Freq (Hz):"), 0, 0)
        self.freq_low_min = QDoubleSpinBox()
        self.freq_low_min.setRange(0, 10000)
        self.freq_low_min.setValue(0)
        grid.addWidget(self.freq_low_min, 0, 1)
        self.freq_low_max = QDoubleSpinBox()
        self.freq_low_max.setRange(0, 10000)
        self.freq_low_max.setValue(30)
        grid.addWidget(self.freq_low_max, 0, 2)
        
        grid.addWidget(QLabel("High Freq (Hz):"), 1, 0)
        self.freq_high_min = QDoubleSpinBox()
        self.freq_high_min.setRange(0, 10000)
        self.freq_high_min.setValue(100)
        grid.addWidget(self.freq_high_min, 1, 1)
        self.freq_high_max = QDoubleSpinBox()
        self.freq_high_max.setRange(0, 10000)
        self.freq_high_max.setValue(275)
        grid.addWidget(self.freq_high_max, 1, 2)
        
        btn_update = QPushButton("Update View Range")
        btn_update.clicked.connect(self.update_plot_view)
        grid.addWidget(btn_update, 2, 0, 1, 3)
        
        v_layout.addLayout(grid)
        gb.setLayout(v_layout)
        layout.addWidget(gb)

    def create_snr_group(self, layout):
        gb = QGroupBox("SNR Calculation Range")
        gb.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #e0e0e0; border-radius: 5px; margin-top: 10px; }")
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Signal (Hz):"), 0, 0)
        self.signal_min = QDoubleSpinBox()
        self.signal_min.setRange(0, 10000)
        self.signal_min.setValue(110)
        self.signal_max = QDoubleSpinBox()
        self.signal_max.setRange(0, 10000)
        self.signal_max.setValue(140)
        
        h_sig = QHBoxLayout()
        h_sig.addWidget(self.signal_min)
        h_sig.addWidget(QLabel("to"))
        h_sig.addWidget(self.signal_max)
        grid.addLayout(h_sig, 0, 1)
        
        grid.addWidget(QLabel("Noise (Hz):"), 1, 0)
        self.noise_min = QDoubleSpinBox()
        self.noise_min.setRange(0, 10000)
        self.noise_min.setValue(350)
        self.noise_max = QDoubleSpinBox()
        self.noise_max.setRange(0, 10000)
        self.noise_max.setValue(400)
        
        h_noise = QHBoxLayout()
        h_noise.addWidget(self.noise_min)
        h_noise.addWidget(QLabel("to"))
        h_noise.addWidget(self.noise_max)
        grid.addLayout(h_noise, 1, 1)
        
        self.signal_min.valueChanged.connect(self.calculate_metrics)
        self.signal_max.valueChanged.connect(self.calculate_metrics)
        self.noise_min.valueChanged.connect(self.calculate_metrics)
        self.noise_max.valueChanged.connect(self.calculate_metrics)
        
        gb.setLayout(grid)
        layout.addWidget(gb)

    def create_results_group(self, layout):
        gb = QGroupBox("Results and Metrics")
        gb.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #e0e0e0; border-radius: 5px; margin-top: 10px; }")
        v_layout = QVBoxLayout()
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setRowCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setMaximumHeight(150)
        
        for row in range(4):
            self.metrics_table.setItem(row, 0, QTableWidgetItem("--"))
            self.metrics_table.setItem(row, 1, QTableWidgetItem("--"))
            
        v_layout.addWidget(self.metrics_table)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(100)
        v_layout.addWidget(self.results_text)
        
        gb.setLayout(v_layout)
        layout.addWidget(gb)

    def add_slider_param(self, layout, label, val, min_val, max_val, group=True):
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 5, 0, 5)
        
        h = QHBoxLayout()
        lbl = QLabel(label)
        sp = QSpinBox()
        sp.setRange(min_val, max_val)
        sp.setValue(val)
        h.addWidget(lbl)
        h.addWidget(sp)
        l.addLayout(h)
        
        sl = QSlider(Qt.Horizontal)
        sl.setRange(min_val, max_val)
        sl.setValue(val)
        l.addWidget(sl)
        
        # Sync
        sp.valueChanged.connect(sl.setValue)
        sl.valueChanged.connect(sp.setValue)
        sp.valueChanged.connect(self.schedule_process)
        
        if group:
            gb = QGroupBox()
            gb.setLayout(l)
            layout.addWidget(gb)
        else:
            layout.addWidget(container)
            
        return sp, sl

    def add_float_slider_param(self, layout, label, val, min_val, max_val, scale_factor):
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 5, 0, 5)
        
        h = QHBoxLayout()
        lbl = QLabel(label)
        sp = QDoubleSpinBox()
        sp.setRange(min_val, max_val)
        sp.setValue(val)
        sp.setSingleStep(1.0/scale_factor)
        h.addWidget(lbl)
        h.addWidget(sp)
        l.addLayout(h)
        
        sl = QSlider(Qt.Horizontal)
        sl.setRange(int(min_val * scale_factor), int(max_val * scale_factor))
        sl.setValue(int(val * scale_factor))
        l.addWidget(sl)
        
        # Sync
        def update_spin(v):
            sp.blockSignals(True)
            sp.setValue(v / scale_factor)
            sp.blockSignals(False)
            self.schedule_process()
            
        def update_slider(v):
            sl.blockSignals(True)
            sl.setValue(int(v * scale_factor))
            sl.blockSignals(False)
            self.schedule_process()
            
        sl.valueChanged.connect(update_spin)
        sp.valueChanged.connect(update_slider)
        
        gb = QGroupBox()
        gb.setLayout(l)
        layout.addWidget(gb)
        return sp, sl

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Monitor Folder")
        if folder:
            self.folder_label.setText(folder)
            self.btn_start.setEnabled(True)

    def toggle_monitoring(self, checked):
        if checked:
            folder = self.folder_label.text()
            if not os.path.exists(folder): return
            
            self.monitor = RealtimeDataMonitor(folder, poll_interval=1.0)
            self.monitor.on_average_updated = lambda d, c: self.live_signals.data_received.emit(d, c)
            self.monitor.on_new_scan = lambda d, c: self.live_signals.data_received.emit(d, c)
            self.monitor.start(
                average_mode=self.rb_avg.isChecked(),
                process_existing=self.chk_existing.isChecked()
            )
            
            self.is_monitoring = True
            self.btn_start.setText("STOP MONITORING")
            self.lbl_status.setText("Monitoring Active...")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            if self.monitor: self.monitor.stop()
            self.is_monitoring = False
            self.btn_start.setText("START MONITORING")
            self.lbl_status.setText("Stopped")
            self.lbl_status.setStyleSheet("color: black;")

    def update_mode(self):
        if self.monitor and self.is_monitoring:
            self.monitor.set_mode(self.rb_avg.isChecked())

    @Slot(object, int)
    def handle_live_data(self, data, count):
        self.halp = data.time_data
        self.sampling_rate = data.sampling_rate
        self.acq_time = data.acquisition_time
        self.scan_count = count
        self.lbl_status.setText(f"Scans: {count}")
        self.process_data()

    def schedule_process(self):
        self.timer_process.start(200)

    def process_data(self):
        if self.halp is None: return
        
        params = {
            'conv_points': self.spin_conv.value(),
            'poly_order': self.spin_poly.value(),
            'enable_svd': self.chk_svd.isChecked(),
            'svd_rank': self.spin_rank.value(),
            'trunc_start': self.spin_trunc_start.value(),
            'trunc_end': self.spin_trunc_end.value(),
            'enable_recon': self.chk_lp.isChecked(),
            'recon_points': self.spin_lp_points.value(),
            'recon_order': self.spin_lp_order.value(),
            'recon_train_len': self.spin_lp_train.value(),
            'apod_t2star': self.spin_t2.value(),
            'use_hanning': self.chk_hanning.isChecked(),
            'zf_factor': self.spin_zf.value(),
            'phase0': self.spin_ph0.value(),
            'phase1': self.spin_ph1.value()
        }
        
        if self.worker:
            self.worker.stop()
            self.worker.wait(500)
            
        self.worker = ProcessingWorker(self.halp, self.sampling_rate, self.acq_time, params)
        self.worker.finished.connect(self.update_plots)
        self.worker.start()

    @Slot(object)
    def update_plots(self, result):
        self.last_result = result
        
        # Time
        self.ax_time.clear()
        self.ax_time.plot(result['time_data'], 'b-', linewidth=0.8)
        if result['n_backward'] > 0:
            self.ax_time.axvspan(0, result['n_backward'], color='r', alpha=0.2)
        self.ax_time.set_title(f"Time Domain (Scans: {self.scan_count})")
        self.ax_time.grid(True, alpha=0.3)
        self.canvas_time.draw()
        
        # Freq - Use Visualization Settings
        self.update_plot_view()

    def update_plot_view(self):
        if self.last_result is None: return
        
        result = self.last_result
        freq_axis = result['freq_axis']
        spectrum = result['spectrum']
        
        # Determine data to plot
        if self.view_real.isChecked():
            data = np.real(spectrum)
            if self.show_absolute.isChecked(): data = np.abs(data)
        elif self.view_imag.isChecked():
            data = np.imag(spectrum)
            if self.show_absolute.isChecked(): data = np.abs(data)
        else: # Mag
            data = np.abs(spectrum)
            
        self.ax_freq.clear()
        self.ax_freq.plot(freq_axis, data, 'k-', linewidth=0.8)
        
        # Set limits
        low_min = self.freq_low_min.value()
        low_max = self.freq_low_max.value()
        high_min = self.freq_high_min.value()
        high_max = self.freq_high_max.value()
        
        # We want to show two regions: low freq and high freq
        # But matplotlib only supports one x-axis per subplot easily.
        # For now, let's just set the view to the full range defined by min(low) and max(high)
        # Or better, just respect the standard NMR convention (high to low)
        # The original UI had specific "Maximize Low/High" actions, but here we have ranges.
        # Let's just set the x-lim to cover the widest range specified if they are disjoint, 
        # or just full range if not specified.
        # Actually, the original UI had a "Update View Range" button that likely set the x-lim.
        # Let's just set xlim to (high_max, low_min) to cover everything, reversed.
        
        self.ax_freq.set_xlim(high_max, low_min) # Reverse NMR convention
        
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.grid(True, alpha=0.3)
        self.canvas_freq.draw()
        
        self.calculate_metrics()

    def calculate_metrics(self):
        if self.last_result is None: return
        
        freq_axis = self.last_result['freq_axis']
        spectrum = self.last_result['spectrum']
        
        # Get data based on view
        if self.view_real.isChecked():
            data = np.real(spectrum)
            mode = "Real"
        elif self.view_imag.isChecked():
            data = np.imag(spectrum)
            mode = "Imag"
        else:
            data = np.abs(spectrum)
            mode = "Mag"
            
        if self.show_absolute.isChecked():
            data = np.abs(data)
            mode = f"|{mode}|"
            
        # SNR
        sig_min = self.signal_min.value()
        sig_max = self.signal_max.value()
        noise_min = self.noise_min.value()
        noise_max = self.noise_max.value()
        
        try:
            sig_mask = (freq_axis >= sig_min) & (freq_axis <= sig_max)
            noise_mask = (freq_axis >= noise_min) & (freq_axis <= noise_max)
            
            if np.any(sig_mask):
                sig_peak = np.max(np.abs(data[sig_mask]))
            else:
                sig_peak = 0
                
            if np.any(noise_mask):
                noise_std = np.std(data[noise_mask])
            else:
                noise_std = 1.0
                
            snr = sig_peak / noise_std if noise_std > 0 else 0
            
            self.metrics_table.item(0, 0).setText(f"SNR Total ({mode})")
            self.metrics_table.item(0, 1).setText(f"{snr:.2f}")
            
            self.metrics_table.item(1, 0).setText("Noise Level")
            self.metrics_table.item(1, 1).setText(f"{noise_std:.2f}")
            
            # FWHM (Simplified)
            if sig_peak > 0:
                half_max = sig_peak / 2
                # Find indices where data > half_max within signal region
                # This is a very rough approximation
                peaks = np.where((data > half_max) & sig_mask)[0]
                if len(peaks) > 1:
                    width_hz = freq_axis[peaks[-1]] - freq_axis[peaks[0]]
                    self.metrics_table.item(2, 0).setText("Linewidth (Hz)")
                    self.metrics_table.item(2, 1).setText(f"{abs(width_hz):.2f}")
            
        except Exception as e:
            print(f"Metrics error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = LiveMonitorUI()
    window.show()
    sys.exit(app.exec())
