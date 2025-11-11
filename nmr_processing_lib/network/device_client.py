"""
NMR Device/Setup Host Client
=============================

Client for communicating with NMR acquisition setup/device host.

Supports:
- Connection management
- Command sending (start/stop acquisition, set parameters)
- Status monitoring
- Real-time data streaming
"""

import socket
import json
import time
import threading
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
import struct


class DeviceStatus(Enum):
    """Device status codes"""
    IDLE = "idle"
    ACQUIRING = "acquiring"
    PROCESSING = "processing"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class CommandType(Enum):
    """Command types"""
    START_ACQUISITION = "start_acq"
    STOP_ACQUISITION = "stop_acq"
    SET_PARAMETERS = "set_params"
    GET_STATUS = "get_status"
    GET_DATA = "get_data"
    CALIBRATE = "calibrate"
    RESET = "reset"


@dataclass
class AcquisitionConfig:
    """Acquisition configuration parameters"""
    num_scans: int = 1
    sampling_rate: float = 8333.0
    num_points: int = 65515
    pulse_length: float = 10.0  # microseconds
    delay_time: float = 1.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AcquisitionConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class DeviceConnectionError(Exception):
    """Connection error"""
    pass


class DeviceCommandError(Exception):
    """Command execution error"""
    pass


class NMRDeviceClient:
    """
    Client for NMR device/setup host communication.
    
    Features:
    - TCP/IP socket communication
    - JSON-based command protocol
    - Binary data transfer
    - Status monitoring
    - Event callbacks
    
    Example:
        >>> client = NMRDeviceClient('192.168.1.100', 5000)
        >>> client.connect()
        >>> 
        >>> # Set acquisition parameters
        >>> config = AcquisitionConfig(num_scans=16, sampling_rate=10000)
        >>> client.set_parameters(config)
        >>> 
        >>> # Start acquisition with callback
        >>> def on_data(scan_data):
        ...     print(f"Received scan: {len(scan_data)} points")
        >>> 
        >>> client.on_scan_received = on_data
        >>> client.start_acquisition()
        >>> 
        >>> # ... wait for acquisition ...
        >>> 
        >>> client.stop_acquisition()
        >>> client.disconnect()
    """
    
    def __init__(
        self, 
        host: str, 
        port: int = 5000,
        timeout: float = 10.0
    ):
        """
        Initialize device client.
        
        Args:
            host: Device host IP address
            port: Communication port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._status = DeviceStatus.DISCONNECTED
        
        # Background threads
        self._monitor_thread: Optional[threading.Thread] = None
        self._data_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_status_changed: Optional[Callable[[DeviceStatus], None]] = None
        self.on_scan_received: Optional[Callable[[np.ndarray], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_message: Optional[Callable[[str], None]] = None
    
    def connect(self) -> bool:
        """
        Connect to device host.
        
        Returns:
            True if connection successful
        
        Raises:
            DeviceConnectionError: If connection fails
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self._connected = True
            
            # Start monitoring thread
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                daemon=True
            )
            self._monitor_thread.start()
            
            # Get initial status
            self._update_status(DeviceStatus.IDLE)
            
            if self.on_message:
                self.on_message(f"Connected to {self.host}:{self.port}")
            
            return True
            
        except Exception as e:
            self._connected = False
            raise DeviceConnectionError(f"Connection failed: {e}")
    
    def disconnect(self):
        """Disconnect from device"""
        self._stop_event.set()
        self._connected = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        if self._data_thread:
            self._data_thread.join(timeout=2.0)
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
        
        self._update_status(DeviceStatus.DISCONNECTED)
        
        if self.on_message:
            self.on_message("Disconnected")
    
    def send_command(
        self, 
        command_type: CommandType, 
        parameters: Optional[Dict[str, Any]] = None,
        expect_response: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Send command to device.
        
        Args:
            command_type: Type of command
            parameters: Command parameters
            expect_response: Wait for response
        
        Returns:
            Response data if expect_response=True
        
        Raises:
            DeviceCommandError: If command fails
        """
        if not self._connected:
            raise DeviceCommandError("Not connected to device")
        
        # Build command message
        message = {
            'command': command_type.value,
            'parameters': parameters or {},
            'timestamp': time.time()
        }
        
        try:
            # Send JSON message
            json_data = json.dumps(message).encode('utf-8')
            message_length = struct.pack('!I', len(json_data))
            
            self._socket.sendall(message_length + json_data)
            
            if expect_response:
                # Receive response
                response_length = struct.unpack('!I', self._recv_exactly(4))[0]
                response_data = self._recv_exactly(response_length)
                response = json.loads(response_data.decode('utf-8'))
                
                if response.get('status') == 'error':
                    raise DeviceCommandError(response.get('message', 'Unknown error'))
                
                return response
            
            return None
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Command failed: {e}")
            raise DeviceCommandError(f"Command failed: {e}")
    
    def start_acquisition(self) -> bool:
        """
        Start data acquisition.
        
        Returns:
            True if started successfully
        """
        try:
            response = self.send_command(CommandType.START_ACQUISITION)
            
            if response and response.get('status') == 'ok':
                self._update_status(DeviceStatus.ACQUIRING)
                
                # Start data receiving thread
                self._data_thread = threading.Thread(
                    target=self._data_receiving_loop,
                    daemon=True
                )
                self._data_thread.start()
                
                if self.on_message:
                    self.on_message("Acquisition started")
                
                return True
            
            return False
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to start acquisition: {e}")
            return False
    
    def stop_acquisition(self) -> bool:
        """
        Stop data acquisition.
        
        Returns:
            True if stopped successfully
        """
        try:
            response = self.send_command(CommandType.STOP_ACQUISITION)
            
            if response and response.get('status') == 'ok':
                self._update_status(DeviceStatus.IDLE)
                
                if self.on_message:
                    self.on_message("Acquisition stopped")
                
                return True
            
            return False
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to stop acquisition: {e}")
            return False
    
    def set_parameters(self, config: AcquisitionConfig) -> bool:
        """
        Set acquisition parameters.
        
        Args:
            config: Acquisition configuration
        
        Returns:
            True if parameters set successfully
        """
        try:
            response = self.send_command(
                CommandType.SET_PARAMETERS,
                parameters=config.to_dict()
            )
            
            if response and response.get('status') == 'ok':
                if self.on_message:
                    self.on_message("Parameters updated")
                return True
            
            return False
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to set parameters: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get device status.
        
        Returns:
            Status dictionary
        """
        try:
            response = self.send_command(CommandType.GET_STATUS)
            return response.get('data', {})
        except:
            return {'status': 'error', 'message': 'Failed to get status'}
    
    def calibrate(self) -> bool:
        """
        Run device calibration.
        
        Returns:
            True if calibration successful
        """
        try:
            response = self.send_command(CommandType.CALIBRATE)
            return response and response.get('status') == 'ok'
        except:
            return False
    
    def reset(self) -> bool:
        """
        Reset device.
        
        Returns:
            True if reset successful
        """
        try:
            response = self.send_command(CommandType.RESET)
            return response and response.get('status') == 'ok'
        except:
            return False
    
    # Private methods
    
    def _recv_exactly(self, n_bytes: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n_bytes:
            chunk = self._socket.recv(n_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
    
    def _monitor_loop(self):
        """Background status monitoring loop"""
        while not self._stop_event.is_set():
            try:
                if self._connected:
                    status = self.get_status()
                    
                    # Update status if changed
                    device_status = DeviceStatus(status.get('status', 'idle'))
                    if device_status != self._status:
                        self._update_status(device_status)
                
                time.sleep(1.0)  # Poll every second
                
            except Exception as e:
                if self.on_error and self._connected:
                    self.on_error(f"Monitoring error: {e}")
                time.sleep(2.0)
    
    def _data_receiving_loop(self):
        """Background data receiving loop"""
        while not self._stop_event.is_set() and self._status == DeviceStatus.ACQUIRING:
            try:
                # Receive data packet header
                header_data = self._recv_exactly(12)
                packet_type, data_length = struct.unpack('!II', header_data[:8])
                
                if packet_type == 1:  # Scan data packet
                    # Receive binary data
                    raw_data = self._recv_exactly(data_length)
                    
                    # Convert to complex numpy array
                    scan_data = np.frombuffer(raw_data, dtype=np.complex64)
                    
                    # Callback
                    if self.on_scan_received:
                        self.on_scan_received(scan_data)
                
            except Exception as e:
                if self._connected:
                    if self.on_error:
                        self.on_error(f"Data receiving error: {e}")
                break
    
    def _update_status(self, new_status: DeviceStatus):
        """Update device status and trigger callback"""
        if new_status != self._status:
            self._status = new_status
            if self.on_status_changed:
                self.on_status_changed(new_status)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected
    
    @property
    def current_status(self) -> DeviceStatus:
        """Get current status"""
        return self._status


# Utility functions

def discover_devices(
    port: int = 5000,
    timeout: float = 2.0,
    broadcast_address: str = '255.255.255.255'
) -> List[Dict[str, Any]]:
    """
    Discover NMR devices on local network.
    
    Args:
        port: Port to scan
        timeout: Discovery timeout
        broadcast_address: Broadcast address
    
    Returns:
        List of discovered devices
    
    Example:
        >>> devices = discover_devices()
        >>> for device in devices:
        ...     print(f"Found: {device['ip']} - {device['name']}")
    """
    discovered = []
    
    try:
        # Create UDP socket for broadcast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        
        # Send discovery message
        message = json.dumps({'type': 'discovery', 'timestamp': time.time()})
        sock.sendto(message.encode('utf-8'), (broadcast_address, port))
        
        # Collect responses
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                
                if response.get('type') == 'device_info':
                    device_info = {
                        'ip': addr[0],
                        'port': response.get('port', port),
                        'name': response.get('name', 'Unknown'),
                        'model': response.get('model', 'Unknown'),
                        'firmware': response.get('firmware', 'Unknown')
                    }
                    discovered.append(device_info)
                    
            except socket.timeout:
                break
            except Exception:
                continue
        
        sock.close()
        
    except Exception as e:
        print(f"Discovery error: {e}")
    
    return discovered
