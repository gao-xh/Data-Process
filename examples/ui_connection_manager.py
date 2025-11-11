"""
PySide6 UI Integration Example
==============================

Example of integrating ConnectionManager with PySide6 UI,
allowing users to choose between local and cloud servers.
"""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QListWidget, QTextEdit,
    QGroupBox, QRadioButton, QButtonGroup, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
import sys
import time

from nmr_processing_lib.network import (
    ConnectionManager,
    ConnectionProfile,
    ConnectionMode,
    ServerType,
    ConnectionStatus
)


class ConnectionManagerUI(QMainWindow):
    """
    Main window for connection management UI.
    
    Features:
    - Add/remove connection profiles
    - Choose between local and cloud servers
    - Auto-discover local devices
    - Monitor connection status
    - User-friendly interface in Chinese
    """
    
    def __init__(self):
        super().__init__()
        
        self.manager = ConnectionManager()
        self.current_client = None
        
        self.init_ui()
        self.setup_callbacks()
        self.load_profiles()
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("NMR网络连接管理")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Connection mode selection
        mode_group = QGroupBox("连接模式")
        mode_layout = QHBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        self.local_radio = QRadioButton("仅本地设备")
        self.cloud_radio = QRadioButton("仅云端服务器")
        self.hybrid_radio = QRadioButton("智能选择（优先本地）")
        self.auto_radio = QRadioButton("自动选择")
        
        self.hybrid_radio.setChecked(True)  # Default
        
        self.mode_button_group.addButton(self.local_radio, 0)
        self.mode_button_group.addButton(self.cloud_radio, 1)
        self.mode_button_group.addButton(self.hybrid_radio, 2)
        self.mode_button_group.addButton(self.auto_radio, 3)
        
        mode_layout.addWidget(self.local_radio)
        mode_layout.addWidget(self.cloud_radio)
        mode_layout.addWidget(self.hybrid_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_group.setLayout(mode_layout)
        
        main_layout.addWidget(mode_group)
        
        # Server list
        server_group = QGroupBox("可用服务器")
        server_layout = QVBoxLayout()
        
        # Server type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("服务器类型:"))
        self.server_type_combo = QComboBox()
        self.server_type_combo.addItems([
            "NMR采集设备",
            "仿真服务器",
            "数据存储",
            "远程控制"
        ])
        self.server_type_combo.currentIndexChanged.connect(self.on_server_type_changed)
        type_layout.addWidget(self.server_type_combo)
        type_layout.addStretch()
        
        server_layout.addLayout(type_layout)
        
        # Server list widget
        self.server_list = QListWidget()
        self.server_list.itemDoubleClicked.connect(self.on_server_double_clicked)
        server_layout.addWidget(self.server_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.discover_btn = QPushButton("🔍 发现本地设备")
        self.discover_btn.clicked.connect(self.discover_devices)
        button_layout.addWidget(self.discover_btn)
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.connect_server)
        button_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.clicked.connect(self.disconnect_server)
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.disconnect_btn)
        
        server_layout.addLayout(button_layout)
        server_group.setLayout(server_layout)
        
        main_layout.addWidget(server_group)
        
        # Status display
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.connection_info = QTextEdit()
        self.connection_info.setReadOnly(True)
        self.connection_info.setMaximumHeight(100)
        status_layout.addWidget(self.connection_info)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Log
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
    
    def setup_callbacks(self):
        """Setup ConnectionManager callbacks"""
        self.manager.on_connection_changed = self.on_connection_changed
        self.manager.on_connection_error = self.on_connection_error
        self.manager.on_profile_discovered = self.on_profile_discovered
    
    def load_profiles(self):
        """Load saved profiles"""
        self.manager.load_profiles()
        
        # If no profiles, add defaults
        if not self.manager.profiles:
            self.add_default_profiles()
        
        self.refresh_server_list()
        self.log("配置加载完成")
    
    def add_default_profiles(self):
        """Add default connection profiles"""
        # Local device
        self.manager.add_profile(ConnectionProfile(
            name="本地NMR设备（默认）",
            server_type=ServerType.DEVICE,
            mode=ConnectionMode.LOCAL,
            host="192.168.1.100",
            port=5000
        ))
        
        # Cloud simulation
        self.manager.add_profile(ConnectionProfile(
            name="云端Spinach服务器（默认）",
            server_type=ServerType.SIMULATION,
            mode=ConnectionMode.CLOUD,
            host="localhost",
            port=8000
        ))
        
        self.manager.save_profiles()
    
    def get_current_server_type(self) -> ServerType:
        """Get currently selected server type"""
        type_map = {
            0: ServerType.DEVICE,
            1: ServerType.SIMULATION,
            2: ServerType.STORAGE,
            3: ServerType.CONTROL
        }
        return type_map[self.server_type_combo.currentIndex()]
    
    def get_current_connection_mode(self) -> ConnectionMode:
        """Get currently selected connection mode"""
        mode_id = self.mode_button_group.checkedId()
        mode_map = {
            0: ConnectionMode.LOCAL,
            1: ConnectionMode.CLOUD,
            2: ConnectionMode.HYBRID,
            3: ConnectionMode.AUTO
        }
        return mode_map.get(mode_id, ConnectionMode.AUTO)
    
    def refresh_server_list(self):
        """Refresh server list"""
        self.server_list.clear()
        
        server_type = self.get_current_server_type()
        profiles = self.manager.list_profiles(server_type=server_type)
        
        for profile in profiles:
            # Format display text
            mode_icon = "🏠" if profile.mode == ConnectionMode.LOCAL else "☁️"
            status = self.manager.get_status(profile.name)
            status_icon = "✅" if status == ConnectionStatus.CONNECTED else ""
            
            text = f"{status_icon} {mode_icon} {profile.name} - {profile.host}:{profile.port}"
            self.server_list.addItem(text)
            
            # Store profile name in item data
            item = self.server_list.item(self.server_list.count() - 1)
            item.setData(Qt.UserRole, profile.name)
    
    @Slot()
    def on_server_type_changed(self):
        """Handle server type change"""
        self.refresh_server_list()
    
    @Slot()
    def on_server_double_clicked(self, item):
        """Handle double-click on server"""
        self.connect_server()
    
    @Slot()
    def discover_devices(self):
        """Discover local devices"""
        self.log("正在扫描本地网络...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        try:
            server_type = self.get_current_server_type()
            discovered = self.manager.discover_local_devices(
                server_type=server_type,
                timeout=3.0
            )
            
            if discovered:
                self.log(f"发现 {len(discovered)} 个设备")
                self.refresh_server_list()
                self.manager.save_profiles()
            else:
                self.log("未发现设备")
                QMessageBox.information(self, "扫描结果", "未在网络上发现设备")
        
        except Exception as e:
            self.log(f"扫描失败: {e}")
            QMessageBox.warning(self, "扫描失败", f"设备扫描失败:\n{e}")
        
        finally:
            self.progress_bar.setVisible(False)
    
    @Slot()
    def connect_server(self):
        """Connect to selected server"""
        current_item = self.server_list.currentItem()
        
        if not current_item:
            # No selection, try auto-connect
            self.auto_connect()
            return
        
        profile_name = current_item.data(Qt.UserRole)
        
        self.log(f"正在连接到 {profile_name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            self.current_client = self.manager.connect(profile_name)
            
            self.log(f"✓ 已连接到 {profile_name}")
            self.update_connection_info(profile_name)
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            
            self.refresh_server_list()
        
        except Exception as e:
            self.log(f"✗ 连接失败: {e}")
            QMessageBox.critical(self, "连接失败", f"无法连接到服务器:\n{e}")
        
        finally:
            self.progress_bar.setVisible(False)
    
    @Slot()
    def disconnect_server(self):
        """Disconnect current server"""
        # Find connected profile
        for name, client in list(self.manager.connections.items()):
            if client is self.current_client:
                self.log(f"正在断开 {name}...")
                self.manager.disconnect(name)
                self.log(f"✓ 已断开 {name}")
                break
        
        self.current_client = None
        self.status_label.setText("未连接")
        self.connection_info.clear()
        
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        
        self.refresh_server_list()
    
    def auto_connect(self):
        """Auto-connect based on mode"""
        self.log("自动连接...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            server_type = self.get_current_server_type()
            mode = self.get_current_connection_mode()
            
            self.current_client = self.manager.auto_connect(
                server_type=server_type,
                mode=mode if mode != ConnectionMode.AUTO else None
            )
            
            # Find which profile was used
            for name, client in self.manager.connections.items():
                if client is self.current_client:
                    self.log(f"✓ 自动连接成功: {name}")
                    self.update_connection_info(name)
                    break
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.refresh_server_list()
        
        except Exception as e:
            self.log(f"✗ 自动连接失败: {e}")
            QMessageBox.critical(self, "连接失败", f"无法自动连接:\n{e}")
        
        finally:
            self.progress_bar.setVisible(False)
    
    def update_connection_info(self, profile_name: str):
        """Update connection info display"""
        profile = self.manager.get_profile(profile_name)
        
        if profile:
            mode_text = {
                ConnectionMode.LOCAL: "本地",
                ConnectionMode.CLOUD: "云端",
                ConnectionMode.HYBRID: "混合",
                ConnectionMode.AUTO: "自动"
            }[profile.mode]
            
            type_text = {
                ServerType.DEVICE: "NMR采集设备",
                ServerType.SIMULATION: "仿真服务器",
                ServerType.STORAGE: "数据存储",
                ServerType.CONTROL: "远程控制"
            }[profile.server_type]
            
            self.status_label.setText(f"✅ 已连接: {profile.name}")
            
            info_text = f"""
服务器名称: {profile.name}
服务器类型: {type_text}
连接模式: {mode_text}
地址: {profile.host}:{profile.port}
SSL: {'是' if profile.use_ssl else '否'}
            """.strip()
            
            self.connection_info.setText(info_text)
    
    @Slot(str, object)
    def on_connection_changed(self, name: str, status: ConnectionStatus):
        """Handle connection status change"""
        status_text = {
            ConnectionStatus.DISCONNECTED: "断开",
            ConnectionStatus.CONNECTING: "连接中",
            ConnectionStatus.CONNECTED: "已连接",
            ConnectionStatus.RECONNECTING: "重连中",
            ConnectionStatus.FAILED: "失败"
        }[status]
        
        self.log(f"[状态] {name}: {status_text}")
        self.refresh_server_list()
    
    @Slot(str, str)
    def on_connection_error(self, name: str, error: str):
        """Handle connection error"""
        self.log(f"[错误] {name}: {error}")
    
    @Slot(object)
    def on_profile_discovered(self, profile: ConnectionProfile):
        """Handle profile discovered"""
        self.log(f"[发现] {profile.name} @ {profile.host}:{profile.port}")
    
    def log(self, message: str):
        """Add log message"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle window close"""
        # Disconnect all
        self.manager.disconnect_all()
        self.manager.save_profiles()
        event.accept()


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    
    window = ConnectionManagerUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
