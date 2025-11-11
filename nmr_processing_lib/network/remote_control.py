"""
Remote Control Module
=====================

Remote control for NMR devices and servers.

Supports:
- Command/response protocol
- Status monitoring
- Event notifications
- Remote parameter adjustment
"""

import socket
import json
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue


class MessageType(Enum):
    """Message types"""
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    EVENT = "event"
    HEARTBEAT = "heartbeat"


@dataclass
class CommandMessage:
    """Command message"""
    command: str
    parameters: Dict[str, Any]
    message_id: Optional[str] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class StatusMessage:
    """Status message"""
    device_id: str
    status: str
    parameters: Dict[str, Any]
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class RemoteControlClient:
    """
    Client for remote control of devices/servers.
    
    Example:
        >>> client = RemoteControlClient('192.168.1.100', 7000)
        >>> client.connect()
        >>> 
        >>> # Set status callback
        >>> def on_status(status_msg):
        ...     print(f"Device status: {status_msg.status}")
        >>> 
        >>> client.on_status_update = on_status
        >>> 
        >>> # Send command
        >>> response = client.send_command('set_parameter', {'gain': 10})
        >>> print(f"Response: {response}")
        >>> 
        >>> client.disconnect()
    """
    
    def __init__(
        self,
        host: str,
        port: int = 7000,
        timeout: float = 10.0
    ):
        """
        Initialize remote control client.
        
        Args:
            host: Server host
            port: Server port
            timeout: Socket timeout
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
        
        # Background threads
        self._receive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Message queues
        self._response_queue = Queue()
        self._pending_commands: Dict[str, Queue] = {}
        
        # Callbacks
        self.on_status_update: Optional[Callable[[StatusMessage], None]] = None
        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Message ID counter
        self._message_id = 0
    
    def connect(self) -> bool:
        """Connect to remote server"""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self._connected = True
            
            # Start receive thread
            self._stop_event.clear()
            self._receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            self._receive_thread.start()
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self._stop_event.set()
        self._connected = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
    
    def send_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send command and wait for response.
        
        Args:
            command: Command name
            parameters: Command parameters
            timeout: Response timeout
        
        Returns:
            Response data
        """
        if not self._connected:
            raise ConnectionError("Not connected")
        
        # Create command message
        msg_id = self._get_next_message_id()
        cmd_msg = CommandMessage(
            command=command,
            parameters=parameters or {},
            message_id=msg_id,
            timestamp=time.time()
        )
        
        # Create response queue
        response_queue = Queue()
        self._pending_commands[msg_id] = response_queue
        
        try:
            # Send message
            self._send_message(MessageType.COMMAND, cmd_msg.to_dict())
            
            # Wait for response
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except:
                return None
                
        finally:
            # Clean up
            self._pending_commands.pop(msg_id, None)
    
    def send_command_async(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Send command asynchronously.
        
        Args:
            command: Command name
            parameters: Command parameters
            callback: Response callback
        """
        def _async_send():
            response = self.send_command(command, parameters)
            if callback and response:
                callback(response)
        
        thread = threading.Thread(target=_async_send, daemon=True)
        thread.start()
    
    def _send_message(self, msg_type: MessageType, data: Dict[str, Any]):
        """Send message to server"""
        import struct
        
        message = {
            'type': msg_type.value,
            'data': data
        }
        
        json_data = json.dumps(message).encode('utf-8')
        
        # Send length prefix + data
        self._socket.sendall(struct.pack('!I', len(json_data)))
        self._socket.sendall(json_data)
    
    def _receive_loop(self):
        """Background message receiving loop"""
        import struct
        
        while not self._stop_event.is_set() and self._connected:
            try:
                # Receive message length
                length_data = self._recv_exactly(4)
                if not length_data:
                    break
                
                msg_length = struct.unpack('!I', length_data)[0]
                
                # Receive message
                msg_data = self._recv_exactly(msg_length)
                if not msg_data:
                    break
                
                message = json.loads(msg_data.decode('utf-8'))
                
                # Handle message by type
                msg_type = MessageType(message['type'])
                data = message['data']
                
                if msg_type == MessageType.RESPONSE:
                    self._handle_response(data)
                elif msg_type == MessageType.STATUS:
                    self._handle_status(data)
                elif msg_type == MessageType.EVENT:
                    self._handle_event(data)
                    
            except Exception as e:
                if self._connected:
                    if self.on_error:
                        self.on_error(f"Receive error: {e}")
                break
    
    def _handle_response(self, data: Dict[str, Any]):
        """Handle response message"""
        msg_id = data.get('message_id')
        
        if msg_id in self._pending_commands:
            queue = self._pending_commands[msg_id]
            queue.put(data)
    
    def _handle_status(self, data: Dict[str, Any]):
        """Handle status message"""
        if self.on_status_update:
            status_msg = StatusMessage.from_dict(data)
            self.on_status_update(status_msg)
    
    def _handle_event(self, data: Dict[str, Any]):
        """Handle event message"""
        if self.on_event:
            self.on_event(data)
    
    def _recv_exactly(self, n_bytes: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n_bytes:
            chunk = self._socket.recv(n_bytes - len(data))
            if not chunk:
                return b''
            data += chunk
        return data
    
    def _get_next_message_id(self) -> str:
        """Get next message ID"""
        self._message_id += 1
        return f"msg_{self._message_id}"
    
    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected


class RemoteControlServer:
    """
    Server for remote control.
    
    Example:
        >>> def handle_command(cmd, params):
        ...     print(f"Command: {cmd}, Params: {params}")
        ...     return {'status': 'ok', 'result': 'done'}
        >>> 
        >>> server = RemoteControlServer(port=7000)
        >>> server.on_command_received = handle_command
        >>> server.start()
        >>> # ... server running ...
        >>> server.stop()
    """
    
    def __init__(self, port: int = 7000):
        """
        Initialize remote control server.
        
        Args:
            port: Server port
        """
        self.port = port
        
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._clients: List[socket.socket] = []
        
        # Callbacks
        self.on_command_received: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None
        self.on_client_connected: Optional[Callable[[str], None]] = None
        self.on_client_disconnected: Optional[Callable[[str], None]] = None
    
    def start(self):
        """Start server"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(('0.0.0.0', self.port))
        self._server_socket.listen(10)
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        
        print(f"Remote control server started on port {self.port}")
    
    def stop(self):
        """Stop server"""
        self._running = False
        
        # Close all client connections
        for client in self._clients:
            try:
                client.close()
            except:
                pass
        
        if self._server_socket:
            self._server_socket.close()
        
        if self._server_thread:
            self._server_thread.join(timeout=5.0)
        
        print("Remote control server stopped")
    
    def broadcast_status(self, status_msg: StatusMessage):
        """
        Broadcast status to all clients.
        
        Args:
            status_msg: Status message
        """
        message = {
            'type': MessageType.STATUS.value,
            'data': status_msg.to_dict()
        }
        
        self._broadcast_message(message)
    
    def broadcast_event(self, event_data: Dict[str, Any]):
        """
        Broadcast event to all clients.
        
        Args:
            event_data: Event data
        """
        message = {
            'type': MessageType.EVENT.value,
            'data': event_data
        }
        
        self._broadcast_message(message)
    
    def _server_loop(self):
        """Main server loop"""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                
                self._clients.append(client_socket)
                
                if self.on_client_connected:
                    self.on_client_connected(f"{addr[0]}:{addr[1]}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self._running:
                    print(f"Server error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle client connection"""
        import struct
        
        try:
            while self._running:
                # Receive message
                length_data = self._recv_exactly(client_socket, 4)
                if not length_data:
                    break
                
                msg_length = struct.unpack('!I', length_data)[0]
                msg_data = self._recv_exactly(client_socket, msg_length)
                
                if not msg_data:
                    break
                
                message = json.loads(msg_data.decode('utf-8'))
                
                # Handle command
                if message['type'] == MessageType.COMMAND.value:
                    self._handle_command(client_socket, message['data'])
                    
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            try:
                self._clients.remove(client_socket)
                client_socket.close()
            except:
                pass
            
            if self.on_client_disconnected:
                self.on_client_disconnected(f"{addr[0]}:{addr[1]}")
    
    def _handle_command(self, client_socket: socket.socket, data: Dict[str, Any]):
        """Handle command from client"""
        cmd_msg = CommandMessage.from_dict(data)
        
        # Execute command
        if self.on_command_received:
            result = self.on_command_received(cmd_msg.command, cmd_msg.parameters)
        else:
            result = {'status': 'error', 'message': 'No command handler'}
        
        # Send response
        response = {
            'type': MessageType.RESPONSE.value,
            'data': {
                'message_id': cmd_msg.message_id,
                'result': result,
                'timestamp': time.time()
            }
        }
        
        self._send_message(client_socket, response)
    
    def _send_message(self, sock: socket.socket, message: Dict[str, Any]):
        """Send message to client"""
        import struct
        
        json_data = json.dumps(message).encode('utf-8')
        sock.sendall(struct.pack('!I', len(json_data)))
        sock.sendall(json_data)
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all clients"""
        disconnected = []
        
        for client in self._clients:
            try:
                self._send_message(client, message)
            except:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            try:
                self._clients.remove(client)
                client.close()
            except:
                pass
    
    def _recv_exactly(self, sock: socket.socket, n_bytes: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n_bytes:
            chunk = sock.recv(n_bytes - len(data))
            if not chunk:
                return b''
            data += chunk
        return data
