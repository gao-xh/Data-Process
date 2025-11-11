"""
Data Transfer Module
===================

High-performance data transfer between devices and servers.

Supports:
- File transfer
- Streaming data transfer
- Compression
- Progress callbacks
"""

import socket
import os
import hashlib
import struct
import zlib
import threading
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from enum import Enum
import time


class TransferProtocol(Enum):
    """Transfer protocol types"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"


class DataTransferClient:
    """
    Client for transferring data to/from server.
    
    Example:
        >>> client = DataTransferClient('192.168.1.100', 6000)
        >>> client.connect()
        >>> 
        >>> # Upload file with progress callback
        >>> def progress(sent, total):
        ...     print(f"Progress: {sent}/{total} ({100*sent/total:.1f}%)")
        >>> 
        >>> client.upload_file('experiment_data.dat', progress_callback=progress)
        >>> 
        >>> # Download file
        >>> client.download_file('result.npy', 'local_result.npy')
        >>> 
        >>> client.disconnect()
    """
    
    def __init__(
        self,
        host: str,
        port: int = 6000,
        buffer_size: int = 65536,
        compress: bool = True
    ):
        """
        Initialize data transfer client.
        
        Args:
            host: Server host
            port: Server port
            buffer_size: Transfer buffer size
            compress: Enable compression
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.compress = compress
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
    
    def connect(self):
        """Connect to server"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))
        self._connected = True
    
    def disconnect(self):
        """Disconnect from server"""
        if self._socket:
            self._socket.close()
        self._connected = False
    
    def upload_file(
        self,
        filepath: str,
        remote_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Upload file to server.
        
        Args:
            filepath: Local file path
            remote_path: Remote destination path (defaults to filename)
            progress_callback: Progress callback(bytes_sent, total_bytes)
        
        Returns:
            True if upload successful
        """
        if not self._connected:
            raise ConnectionError("Not connected")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get file info
        file_size = filepath.stat().st_size
        remote_name = remote_path or filepath.name
        
        # Send upload request
        request = {
            'command': 'upload',
            'filename': remote_name,
            'size': file_size,
            'compress': self.compress
        }
        
        self._send_json(request)
        
        # Wait for acknowledgment
        response = self._recv_json()
        if response.get('status') != 'ready':
            return False
        
        # Transfer file
        with open(filepath, 'rb') as f:
            bytes_sent = 0
            
            while True:
                chunk = f.read(self.buffer_size)
                if not chunk:
                    break
                
                # Compress if enabled
                if self.compress:
                    chunk = zlib.compress(chunk)
                
                # Send chunk size then data
                self._socket.sendall(struct.pack('!I', len(chunk)))
                self._socket.sendall(chunk)
                
                bytes_sent += len(chunk)
                
                if progress_callback:
                    progress_callback(bytes_sent, file_size)
        
        # Send end marker
        self._socket.sendall(struct.pack('!I', 0))
        
        # Wait for confirmation
        response = self._recv_json()
        return response.get('status') == 'ok'
    
    def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download file from server.
        
        Args:
            remote_path: Remote file path
            local_path: Local destination path
            progress_callback: Progress callback(bytes_received, total_bytes)
        
        Returns:
            True if download successful
        """
        if not self._connected:
            raise ConnectionError("Not connected")
        
        # Send download request
        request = {
            'command': 'download',
            'filename': remote_path,
            'compress': self.compress
        }
        
        self._send_json(request)
        
        # Receive file info
        response = self._recv_json()
        if response.get('status') != 'ok':
            return False
        
        file_size = response['size']
        
        # Receive file data
        with open(local_path, 'wb') as f:
            bytes_received = 0
            
            while True:
                # Receive chunk size
                size_data = self._recv_exactly(4)
                chunk_size = struct.unpack('!I', size_data)[0]
                
                if chunk_size == 0:  # End marker
                    break
                
                # Receive chunk
                chunk = self._recv_exactly(chunk_size)
                
                # Decompress if needed
                if self.compress:
                    chunk = zlib.decompress(chunk)
                
                f.write(chunk)
                bytes_received += len(chunk)
                
                if progress_callback:
                    progress_callback(bytes_received, file_size)
        
        return True
    
    def _send_json(self, data: Dict[str, Any]):
        """Send JSON message"""
        import json
        json_data = json.dumps(data).encode('utf-8')
        self._socket.sendall(struct.pack('!I', len(json_data)))
        self._socket.sendall(json_data)
    
    def _recv_json(self) -> Dict[str, Any]:
        """Receive JSON message"""
        import json
        size = struct.unpack('!I', self._recv_exactly(4))[0]
        data = self._recv_exactly(size)
        return json.loads(data.decode('utf-8'))
    
    def _recv_exactly(self, n_bytes: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n_bytes:
            chunk = self._socket.recv(n_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data


class DataTransferServer:
    """
    Server for receiving data transfers.
    
    Example:
        >>> server = DataTransferServer(port=6000, storage_dir='/data')
        >>> server.start()
        >>> # ... server running ...
        >>> server.stop()
    """
    
    def __init__(
        self,
        port: int = 6000,
        storage_dir: str = './data',
        buffer_size: int = 65536
    ):
        """
        Initialize data transfer server.
        
        Args:
            port: Server port
            storage_dir: Directory to store received files
            buffer_size: Transfer buffer size
        """
        self.port = port
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start server"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(('0.0.0.0', self.port))
        self._server_socket.listen(5)
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        
        print(f"Data transfer server started on port {self.port}")
    
    def stop(self):
        """Stop server"""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        if self._server_thread:
            self._server_thread.join(timeout=5.0)
        
        print("Data transfer server stopped")
    
    def _server_loop(self):
        """Main server loop"""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                print(f"Client connected: {addr}")
                
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
        try:
            # Receive request
            request = self._recv_json(client_socket)
            command = request.get('command')
            
            if command == 'upload':
                self._handle_upload(client_socket, request)
            elif command == 'download':
                self._handle_download(client_socket, request)
            else:
                self._send_json(client_socket, {'status': 'error', 'message': 'Unknown command'})
                
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
    
    def _handle_upload(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle file upload"""
        filename = request['filename']
        file_size = request['size']
        compress = request.get('compress', False)
        
        # Send ready
        self._send_json(client_socket, {'status': 'ready'})
        
        # Receive file
        filepath = self.storage_dir / filename
        with open(filepath, 'wb') as f:
            while True:
                size_data = self._recv_exactly(client_socket, 4)
                chunk_size = struct.unpack('!I', size_data)[0]
                
                if chunk_size == 0:
                    break
                
                chunk = self._recv_exactly(client_socket, chunk_size)
                
                if compress:
                    chunk = zlib.decompress(chunk)
                
                f.write(chunk)
        
        # Send confirmation
        self._send_json(client_socket, {'status': 'ok'})
        print(f"File received: {filename}")
    
    def _handle_download(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle file download"""
        filename = request['filename']
        compress = request.get('compress', False)
        
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            self._send_json(client_socket, {'status': 'error', 'message': 'File not found'})
            return
        
        file_size = filepath.stat().st_size
        
        # Send file info
        self._send_json(client_socket, {'status': 'ok', 'size': file_size})
        
        # Send file
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(self.buffer_size)
                if not chunk:
                    break
                
                if compress:
                    chunk = zlib.compress(chunk)
                
                client_socket.sendall(struct.pack('!I', len(chunk)))
                client_socket.sendall(chunk)
        
        # Send end marker
        client_socket.sendall(struct.pack('!I', 0))
        print(f"File sent: {filename}")
    
    def _send_json(self, sock: socket.socket, data: Dict[str, Any]):
        """Send JSON message"""
        import json
        json_data = json.dumps(data).encode('utf-8')
        sock.sendall(struct.pack('!I', len(json_data)))
        sock.sendall(json_data)
    
    def _recv_json(self, sock: socket.socket) -> Dict[str, Any]:
        """Receive JSON message"""
        import json
        size = struct.unpack('!I', self._recv_exactly(sock, 4))[0]
        data = self._recv_exactly(sock, size)
        return json.loads(data.decode('utf-8'))
    
    def _recv_exactly(self, sock: socket.socket, n_bytes: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n_bytes:
            chunk = sock.recv(n_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
