"""
Network Communication Module
============================

Network interfaces for:
1. NMR Device/Setup Host Communication
2. Spinach Simulation Server Communication
3. Data Transfer and Remote Control
4. Connection Management (Local/Cloud)

This module provides both client and server implementations.
"""

from .device_client import (
    NMRDeviceClient,
    DeviceConnectionError,
    DeviceCommandError
)

from .simulation_client import (
    SpinachServerClient,
    SimulationRequest,
    SimulationResponse,
    SimulationError
)

from .data_transfer import (
    DataTransferClient,
    DataTransferServer,
    TransferProtocol
)

from .remote_control import (
    RemoteControlClient,
    RemoteControlServer,
    CommandMessage,
    StatusMessage
)

from .connection_manager import (
    ConnectionManager,
    ConnectionProfile,
    ConnectionMode,
    ServerType,
    ConnectionStatus
)

__all__ = [
    # Device communication
    'NMRDeviceClient',
    'DeviceConnectionError',
    'DeviceCommandError',
    
    # Simulation server
    'SpinachServerClient',
    'SimulationRequest',
    'SimulationResponse',
    'SimulationError',
    
    # Data transfer
    'DataTransferClient',
    'DataTransferServer',
    'TransferProtocol',
    
    # Remote control
    'RemoteControlClient',
    'RemoteControlServer',
    'CommandMessage',
    'StatusMessage',
    
    # Connection management
    'ConnectionManager',
    'ConnectionProfile',
    'ConnectionMode',
    'ServerType',
    'ConnectionStatus',
]
