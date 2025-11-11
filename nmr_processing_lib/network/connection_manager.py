"""
Connection Manager
==================

Manages connections to both local and cloud servers, allowing users to choose
between different connection modes.

Features:
- Local device discovery and connection
- Cloud server connection with authentication
- Connection profiles management
- Automatic failover
- Connection health monitoring
"""

import json
import os
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

from .device_client import NMRDeviceClient, discover_devices, AcquisitionConfig
from .simulation_client import SpinachServerClient
from .data_transfer import DataTransferClient
from .remote_control import RemoteControlClient


class ConnectionMode(Enum):
    """Connection mode"""
    LOCAL = "local"           # 本地设备
    CLOUD = "cloud"          # 云端服务器
    HYBRID = "hybrid"        # 混合模式（优先本地，失败后云端）
    AUTO = "auto"            # 自动选择


class ServerType(Enum):
    """Server type"""
    DEVICE = "device"               # NMR采集设备
    SIMULATION = "simulation"       # 仿真服务器
    STORAGE = "storage"            # 数据存储服务器
    CONTROL = "control"            # 远程控制服务器


@dataclass
class ConnectionProfile:
    """
    Connection profile configuration.
    
    Attributes:
        name: Profile name (e.g., "Lab Device", "Cloud Server")
        server_type: Type of server
        mode: Connection mode (local/cloud)
        host: Server hostname or IP
        port: Server port
        use_ssl: Whether to use SSL/TLS
        api_key: API key for authentication (optional)
        username: Username for authentication (optional)
        password: Password for authentication (optional)
        timeout: Connection timeout in seconds
        auto_reconnect: Enable automatic reconnection
        metadata: Additional metadata
    """
    name: str
    server_type: ServerType
    mode: ConnectionMode
    host: str
    port: int
    use_ssl: bool = False
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: float = 10.0
    auto_reconnect: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.server_type, str):
            self.server_type = ServerType(self.server_type)
        if isinstance(self.mode, str):
            self.mode = ConnectionMode(self.mode)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['server_type'] = self.server_type.value
        d['mode'] = self.mode.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConnectionProfile':
        """Create from dictionary"""
        return cls(**data)


class ConnectionStatus(Enum):
    """Connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ConnectionManager:
    """
    Manages connections to local and cloud servers.
    
    Features:
    - Load/save connection profiles
    - Auto-discovery of local devices
    - Cloud server authentication
    - Connection health monitoring
    - Automatic reconnection
    - Failover support
    
    Example:
        >>> manager = ConnectionManager()
        >>> 
        >>> # Add local device profile
        >>> local_profile = ConnectionProfile(
        ...     name="Lab NMR Device",
        ...     server_type=ServerType.DEVICE,
        ...     mode=ConnectionMode.LOCAL,
        ...     host="192.168.1.100",
        ...     port=5000
        ... )
        >>> manager.add_profile(local_profile)
        >>> 
        >>> # Add cloud simulation profile
        >>> cloud_profile = ConnectionProfile(
        ...     name="Cloud Spinach Server",
        ...     server_type=ServerType.SIMULATION,
        ...     mode=ConnectionMode.CLOUD,
        ...     host="spinach.example.com",
        ...     port=443,
        ...     use_ssl=True,
        ...     api_key="your-api-key"
        ... )
        >>> manager.add_profile(cloud_profile)
        >>> 
        >>> # Connect to device
        >>> device_client = manager.connect("Lab NMR Device")
        >>> 
        >>> # Or auto-select
        >>> device_client = manager.auto_connect(ServerType.DEVICE)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize connection manager.
        
        Args:
            config_file: Path to configuration file (JSON)
        """
        self.config_file = config_file or os.path.expanduser('~/.nmr_connections.json')
        self.profiles: Dict[str, ConnectionProfile] = {}
        self.connections: Dict[str, Any] = {}  # Active connections
        self.status: Dict[str, ConnectionStatus] = {}
        
        # Callbacks
        self.on_connection_changed: Optional[Callable[[str, ConnectionStatus], None]] = None
        self.on_connection_error: Optional[Callable[[str, str], None]] = None
        self.on_profile_discovered: Optional[Callable[[ConnectionProfile], None]] = None
        
        # Monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._monitor_interval = 10.0  # seconds
        
        # Load profiles
        self.load_profiles()
    
    def add_profile(self, profile: ConnectionProfile) -> None:
        """
        Add a connection profile.
        
        Args:
            profile: Connection profile to add
        """
        self.profiles[profile.name] = profile
        self.status[profile.name] = ConnectionStatus.DISCONNECTED
    
    def remove_profile(self, name: str) -> None:
        """
        Remove a connection profile.
        
        Args:
            name: Profile name
        """
        if name in self.profiles:
            # Disconnect if connected
            if name in self.connections:
                self.disconnect(name)
            del self.profiles[name]
            del self.status[name]
    
    def get_profile(self, name: str) -> Optional[ConnectionProfile]:
        """
        Get connection profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Connection profile or None
        """
        return self.profiles.get(name)
    
    def list_profiles(self, 
                     server_type: Optional[ServerType] = None,
                     mode: Optional[ConnectionMode] = None) -> List[ConnectionProfile]:
        """
        List connection profiles.
        
        Args:
            server_type: Filter by server type (optional)
            mode: Filter by connection mode (optional)
            
        Returns:
            List of connection profiles
        """
        profiles = list(self.profiles.values())
        
        if server_type is not None:
            profiles = [p for p in profiles if p.server_type == server_type]
        
        if mode is not None:
            profiles = [p for p in profiles if p.mode == mode]
        
        return profiles
    
    def discover_local_devices(self, 
                               server_type: ServerType = ServerType.DEVICE,
                               port: int = 5000,
                               timeout: float = 3.0) -> List[ConnectionProfile]:
        """
        Discover local devices on the network.
        
        Args:
            server_type: Type of server to discover
            port: Port to scan
            timeout: Discovery timeout
            
        Returns:
            List of discovered profiles
        """
        discovered_profiles = []
        
        if server_type == ServerType.DEVICE:
            # Discover NMR devices
            devices = discover_devices(port=port, timeout=timeout)
            
            for device in devices:
                profile = ConnectionProfile(
                    name=f"{device['name']} ({device['ip']})",
                    server_type=ServerType.DEVICE,
                    mode=ConnectionMode.LOCAL,
                    host=device['ip'],
                    port=device['port'],
                    metadata={
                        'model': device.get('model', ''),
                        'firmware': device.get('firmware', ''),
                        'discovered': True
                    }
                )
                
                discovered_profiles.append(profile)
                
                # Add to profiles if not exists
                if profile.name not in self.profiles:
                    self.add_profile(profile)
                    
                    # Trigger callback
                    if self.on_profile_discovered:
                        self.on_profile_discovered(profile)
        
        return discovered_profiles
    
    def connect(self, 
                profile_name: str,
                **kwargs) -> Any:
        """
        Connect to a server using profile.
        
        Args:
            profile_name: Name of connection profile
            **kwargs: Additional arguments passed to client
            
        Returns:
            Connected client object
            
        Raises:
            ValueError: If profile not found
            ConnectionError: If connection fails
        """
        profile = self.get_profile(profile_name)
        if profile is None:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        # Update status
        self._update_status(profile_name, ConnectionStatus.CONNECTING)
        
        try:
            # Create client based on server type
            client = self._create_client(profile, **kwargs)
            
            # Connect
            if hasattr(client, 'connect'):
                client.connect()
            
            # Store connection
            self.connections[profile_name] = client
            self._update_status(profile_name, ConnectionStatus.CONNECTED)
            
            return client
            
        except Exception as e:
            self._update_status(profile_name, ConnectionStatus.FAILED)
            if self.on_connection_error:
                self.on_connection_error(profile_name, str(e))
            raise ConnectionError(f"Failed to connect to '{profile_name}': {e}")
    
    def disconnect(self, profile_name: str) -> None:
        """
        Disconnect from a server.
        
        Args:
            profile_name: Name of connection profile
        """
        if profile_name in self.connections:
            client = self.connections[profile_name]
            
            # Disconnect
            try:
                if hasattr(client, 'disconnect'):
                    client.disconnect()
                elif hasattr(client, 'close'):
                    client.close()
            except Exception as e:
                if self.on_connection_error:
                    self.on_connection_error(profile_name, f"Disconnect error: {e}")
            
            # Remove connection
            del self.connections[profile_name]
            self._update_status(profile_name, ConnectionStatus.DISCONNECTED)
    
    def disconnect_all(self) -> None:
        """Disconnect all connections"""
        profile_names = list(self.connections.keys())
        for name in profile_names:
            self.disconnect(name)
    
    def get_connection(self, profile_name: str) -> Optional[Any]:
        """
        Get active connection.
        
        Args:
            profile_name: Name of connection profile
            
        Returns:
            Active client or None
        """
        return self.connections.get(profile_name)
    
    def is_connected(self, profile_name: str) -> bool:
        """
        Check if connected.
        
        Args:
            profile_name: Name of connection profile
            
        Returns:
            True if connected
        """
        return profile_name in self.connections
    
    def get_status(self, profile_name: str) -> ConnectionStatus:
        """
        Get connection status.
        
        Args:
            profile_name: Name of connection profile
            
        Returns:
            Connection status
        """
        return self.status.get(profile_name, ConnectionStatus.DISCONNECTED)
    
    def auto_connect(self,
                    server_type: ServerType,
                    mode: Optional[ConnectionMode] = None,
                    **kwargs) -> Any:
        """
        Auto-connect to best available server.
        
        Priority:
        1. Already connected profile
        2. Local devices (if mode allows)
        3. Cloud servers (if mode allows)
        
        Args:
            server_type: Type of server needed
            mode: Preferred connection mode (optional)
            **kwargs: Additional arguments passed to client
            
        Returns:
            Connected client
            
        Raises:
            ConnectionError: If no suitable server found
        """
        # Check existing connections
        for name, client in self.connections.items():
            profile = self.profiles[name]
            if profile.server_type == server_type:
                if mode is None or profile.mode == mode:
                    return client
        
        # Get suitable profiles
        profiles = self.list_profiles(server_type=server_type, mode=mode)
        
        if not profiles:
            # Try discovery for local devices
            if mode in (None, ConnectionMode.LOCAL, ConnectionMode.AUTO, ConnectionMode.HYBRID):
                profiles = self.discover_local_devices(server_type=server_type)
        
        if not profiles:
            raise ConnectionError(f"No {server_type.value} server available")
        
        # Sort profiles by priority
        profiles = self._sort_profiles_by_priority(profiles)
        
        # Try connecting
        last_error = None
        for profile in profiles:
            try:
                return self.connect(profile.name, **kwargs)
            except Exception as e:
                last_error = e
                continue
        
        # All failed
        raise ConnectionError(f"Failed to connect to any {server_type.value} server: {last_error}")
    
    def start_monitoring(self, interval: float = 10.0) -> None:
        """
        Start connection health monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitor_running:
            return
        
        self._monitor_interval = interval
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop connection health monitoring"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def save_profiles(self, filename: Optional[str] = None) -> None:
        """
        Save profiles to file.
        
        Args:
            filename: Path to save file (uses default if not provided)
        """
        filename = filename or self.config_file
        
        data = {
            'profiles': [p.to_dict() for p in self.profiles.values()]
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_profiles(self, filename: Optional[str] = None) -> None:
        """
        Load profiles from file.
        
        Args:
            filename: Path to load file (uses default if not provided)
        """
        filename = filename or self.config_file
        
        if not os.path.exists(filename):
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for profile_data in data.get('profiles', []):
                profile = ConnectionProfile.from_dict(profile_data)
                self.add_profile(profile)
        
        except Exception as e:
            if self.on_connection_error:
                self.on_connection_error('load', f"Failed to load profiles: {e}")
    
    def _create_client(self, profile: ConnectionProfile, **kwargs):
        """Create client based on profile"""
        # Build URL for HTTP clients
        if profile.use_ssl:
            url = f"https://{profile.host}:{profile.port}"
        else:
            url = f"http://{profile.host}:{profile.port}" if profile.server_type == ServerType.SIMULATION else None
        
        # Create client
        if profile.server_type == ServerType.DEVICE:
            client = NMRDeviceClient(
                host=profile.host,
                port=profile.port,
                timeout=profile.timeout,
                **kwargs
            )
        
        elif profile.server_type == ServerType.SIMULATION:
            client = SpinachServerClient(
                server_url=url,
                api_key=profile.api_key,
                timeout=profile.timeout,
                **kwargs
            )
        
        elif profile.server_type == ServerType.STORAGE:
            client = DataTransferClient(
                host=profile.host,
                port=profile.port,
                **kwargs
            )
        
        elif profile.server_type == ServerType.CONTROL:
            client = RemoteControlClient(
                host=profile.host,
                port=profile.port,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown server type: {profile.server_type}")
        
        return client
    
    def _update_status(self, profile_name: str, status: ConnectionStatus) -> None:
        """Update connection status and trigger callback"""
        self.status[profile_name] = status
        
        if self.on_connection_changed:
            self.on_connection_changed(profile_name, status)
    
    def _sort_profiles_by_priority(self, profiles: List[ConnectionProfile]) -> List[ConnectionProfile]:
        """Sort profiles by connection priority"""
        def priority(profile):
            # Local > Cloud
            mode_priority = {
                ConnectionMode.LOCAL: 0,
                ConnectionMode.HYBRID: 1,
                ConnectionMode.AUTO: 2,
                ConnectionMode.CLOUD: 3
            }
            return mode_priority.get(profile.mode, 99)
        
        return sorted(profiles, key=priority)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitor_running:
            try:
                # Check each connection
                for name, client in list(self.connections.items()):
                    profile = self.profiles[name]
                    
                    # Check health
                    is_healthy = self._check_health(client)
                    
                    if not is_healthy:
                        # Connection lost
                        self._update_status(name, ConnectionStatus.RECONNECTING)
                        
                        # Try reconnect if enabled
                        if profile.auto_reconnect:
                            try:
                                if hasattr(client, 'connect'):
                                    client.connect()
                                self._update_status(name, ConnectionStatus.CONNECTED)
                            except Exception as e:
                                self._update_status(name, ConnectionStatus.FAILED)
                                if self.on_connection_error:
                                    self.on_connection_error(name, f"Reconnect failed: {e}")
                        else:
                            # Remove connection
                            del self.connections[name]
                            self._update_status(name, ConnectionStatus.DISCONNECTED)
            
            except Exception as e:
                if self.on_connection_error:
                    self.on_connection_error('monitor', f"Monitoring error: {e}")
            
            # Sleep
            time.sleep(self._monitor_interval)
    
    def _check_health(self, client: Any) -> bool:
        """Check if connection is healthy"""
        try:
            if isinstance(client, SpinachServerClient):
                return client.health_check()
            elif isinstance(client, NMRDeviceClient):
                # Try getting status
                status = client.get_status()
                return status is not None
            elif hasattr(client, '_socket'):
                # Check socket connection
                return client._socket is not None
            else:
                # Assume healthy if no check method
                return True
        except:
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect_all()
        self.stop_monitoring()
