"""
Spinach Simulation Server Client
=================================

Client for communicating with Spinach simulation server.

Supports:
- Simulation request submission
- Result retrieval
- Parameter optimization
- Batch simulations
"""

import requests
import json
import time
import numpy as np
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import base64


class SimulationType(Enum):
    """Simulation types"""
    LIQUID = "liquid"
    SOLID = "solid"
    ZULF = "zulf"
    CUSTOM = "custom"


class SimulationStatus(Enum):
    """Simulation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpinSystem:
    """Spin system definition"""
    isotopes: List[str] = field(default_factory=list)  # e.g., ['1H', '13C']
    spins: List[float] = field(default_factory=list)  # e.g., [0.5, 0.5]
    
    # Interaction parameters
    j_couplings: Dict[str, float] = field(default_factory=dict)  # e.g., {'1-2': 100.0}
    chemical_shifts: List[float] = field(default_factory=list)  # Hz
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpinSystem':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class SimulationRequest:
    """
    Simulation request parameters.
    
    Example:
        >>> system = SpinSystem(
        ...     isotopes=['1H', '1H'],
        ...     spins=[0.5, 0.5],
        ...     j_couplings={'1-2': 10.0},
        ...     chemical_shifts=[0.0, 100.0]
        ... )
        >>> 
        >>> request = SimulationRequest(
        ...     simulation_type=SimulationType.LIQUID,
        ...     spin_system=system,
        ...     magnet_field=0.0,  # ZULF
        ...     sampling_rate=5000,
        ...     num_points=10000
        ... )
    """
    simulation_type: SimulationType = SimulationType.LIQUID
    spin_system: Optional[SpinSystem] = None
    
    # Experimental parameters
    magnet_field: float = 0.0  # Tesla (0 for ZULF)
    sampling_rate: float = 5000.0  # Hz
    num_points: int = 10000
    
    # Pulse sequence
    pulse_sequence: str = "zg"  # Pulse program name
    pulse_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Relaxation
    t1_relaxation: Optional[float] = None  # seconds
    t2_relaxation: Optional[float] = None  # seconds
    
    # Other options
    add_noise: bool = False
    noise_level: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.simulation_type:
            data['simulation_type'] = self.simulation_type.value
        if self.spin_system:
            data['spin_system'] = self.spin_system.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationRequest':
        if 'simulation_type' in data:
            data['simulation_type'] = SimulationType(data['simulation_type'])
        if 'spin_system' in data:
            data['spin_system'] = SpinSystem.from_dict(data['spin_system'])
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class SimulationResponse:
    """Simulation response"""
    job_id: str
    status: SimulationStatus
    
    # Results (if completed)
    time_data: Optional[np.ndarray] = None
    freq_data: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None
    freq_axis: Optional[np.ndarray] = None
    
    # Metadata
    parameters: Optional[Dict[str, Any]] = None
    computation_time: float = 0.0
    error_message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResponse':
        # Decode numpy arrays from base64
        if 'time_data' in data and data['time_data']:
            data['time_data'] = np.frombuffer(
                base64.b64decode(data['time_data']), 
                dtype=np.complex128
            )
        
        if 'freq_data' in data and data['freq_data']:
            data['freq_data'] = np.frombuffer(
                base64.b64decode(data['freq_data']), 
                dtype=np.complex128
            )
        
        if 'time_axis' in data and data['time_axis']:
            data['time_axis'] = np.frombuffer(
                base64.b64decode(data['time_axis']), 
                dtype=np.float64
            )
        
        if 'freq_axis' in data and data['freq_axis']:
            data['freq_axis'] = np.frombuffer(
                base64.b64decode(data['freq_axis']), 
                dtype=np.float64
            )
        
        if 'status' in data:
            data['status'] = SimulationStatus(data['status'])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class SimulationError(Exception):
    """Simulation error"""
    pass


class SpinachServerClient:
    """
    Client for Spinach simulation server.
    
    Features:
    - Submit simulation requests
    - Poll for results
    - Download simulation data
    - Batch processing
    
    Example:
        >>> client = SpinachServerClient('http://localhost:8000')
        >>> 
        >>> # Define spin system
        >>> system = SpinSystem(
        ...     isotopes=['1H', '1H'],
        ...     spins=[0.5, 0.5],
        ...     j_couplings={'1-2': 10.0},
        ...     chemical_shifts=[0.0, 100.0]
        ... )
        >>> 
        >>> # Create simulation request
        >>> request = SimulationRequest(
        ...     simulation_type=SimulationType.ZULF,
        ...     spin_system=system,
        ...     magnet_field=0.0,
        ...     sampling_rate=5000,
        ...     num_points=10000
        ... )
        >>> 
        >>> # Submit and wait for results
        >>> response = client.submit_and_wait(request, timeout=60)
        >>> 
        >>> if response.status == SimulationStatus.COMPLETED:
        ...     print(f"Simulation completed in {response.computation_time:.2f}s")
        ...     # Use response.time_data, response.freq_data
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Spinach server client.
        
        Args:
            server_url: Server URL (e.g., 'http://localhost:8000')
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Session for persistent connections
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def submit_simulation(self, request: SimulationRequest) -> str:
        """
        Submit simulation request.
        
        Args:
            request: Simulation request
        
        Returns:
            Job ID
        
        Raises:
            SimulationError: If submission fails
        """
        try:
            response = self.session.post(
                f"{self.server_url}/api/simulations/submit",
                json=request.to_dict(),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                return data['job_id']
            else:
                raise SimulationError(data.get('message', 'Submission failed'))
                
        except requests.RequestException as e:
            raise SimulationError(f"Failed to submit simulation: {e}")
    
    def get_status(self, job_id: str) -> SimulationStatus:
        """
        Get simulation status.
        
        Args:
            job_id: Job ID
        
        Returns:
            Simulation status
        """
        try:
            response = self.session.get(
                f"{self.server_url}/api/simulations/{job_id}/status",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            return SimulationStatus(data['status'])
            
        except Exception as e:
            raise SimulationError(f"Failed to get status: {e}")
    
    def get_result(self, job_id: str) -> SimulationResponse:
        """
        Get simulation result.
        
        Args:
            job_id: Job ID
        
        Returns:
            Simulation response with results
        
        Raises:
            SimulationError: If retrieval fails
        """
        try:
            response = self.session.get(
                f"{self.server_url}/api/simulations/{job_id}/result",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            return SimulationResponse.from_dict(data)
            
        except Exception as e:
            raise SimulationError(f"Failed to get result: {e}")
    
    def submit_and_wait(
        self,
        request: SimulationRequest,
        timeout: float = 300.0,
        poll_interval: float = 2.0
    ) -> SimulationResponse:
        """
        Submit simulation and wait for completion.
        
        Args:
            request: Simulation request
            timeout: Maximum wait time in seconds
            poll_interval: Status polling interval in seconds
        
        Returns:
            Simulation response
        
        Raises:
            SimulationError: If simulation fails or times out
        """
        # Submit
        job_id = self.submit_simulation(request)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(job_id)
            
            if status == SimulationStatus.COMPLETED:
                return self.get_result(job_id)
            
            elif status == SimulationStatus.FAILED:
                result = self.get_result(job_id)
                raise SimulationError(f"Simulation failed: {result.error_message}")
            
            elif status == SimulationStatus.CANCELLED:
                raise SimulationError("Simulation was cancelled")
            
            time.sleep(poll_interval)
        
        # Timeout
        raise SimulationError(f"Simulation timed out after {timeout}s")
    
    def cancel_simulation(self, job_id: str) -> bool:
        """
        Cancel running simulation.
        
        Args:
            job_id: Job ID
        
        Returns:
            True if cancelled successfully
        """
        try:
            response = self.session.post(
                f"{self.server_url}/api/simulations/{job_id}/cancel",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data.get('status') == 'ok'
            
        except Exception:
            return False
    
    def list_simulations(
        self,
        status_filter: Optional[SimulationStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List simulations.
        
        Args:
            status_filter: Filter by status
            limit: Maximum number of results
        
        Returns:
            List of simulation info
        """
        try:
            params = {'limit': limit}
            if status_filter:
                params['status'] = status_filter.value
            
            response = self.session.get(
                f"{self.server_url}/api/simulations",
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data.get('simulations', [])
            
        except Exception as e:
            raise SimulationError(f"Failed to list simulations: {e}")
    
    def batch_submit(
        self,
        requests: List[SimulationRequest]
    ) -> List[str]:
        """
        Submit batch of simulations.
        
        Args:
            requests: List of simulation requests
        
        Returns:
            List of job IDs
        """
        job_ids = []
        for request in requests:
            try:
                job_id = self.submit_simulation(request)
                job_ids.append(job_id)
            except Exception as e:
                print(f"Failed to submit request: {e}")
        
        return job_ids
    
    def batch_wait(
        self,
        job_ids: List[str],
        timeout: float = 600.0,
        poll_interval: float = 5.0
    ) -> List[SimulationResponse]:
        """
        Wait for batch of simulations to complete.
        
        Args:
            job_ids: List of job IDs
            timeout: Maximum wait time
            poll_interval: Status polling interval
        
        Returns:
            List of simulation responses
        """
        results = []
        start_time = time.time()
        remaining = set(job_ids)
        
        while remaining and (time.time() - start_time < timeout):
            for job_id in list(remaining):
                try:
                    status = self.get_status(job_id)
                    
                    if status in [SimulationStatus.COMPLETED, 
                                  SimulationStatus.FAILED,
                                  SimulationStatus.CANCELLED]:
                        result = self.get_result(job_id)
                        results.append(result)
                        remaining.remove(job_id)
                        
                except Exception as e:
                    print(f"Error checking {job_id}: {e}")
            
            if remaining:
                time.sleep(poll_interval)
        
        return results
    
    def health_check(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if server is responding
        """
        try:
            response = self.session.get(
                f"{self.server_url}/api/health",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information.
        
        Returns:
            Server info dictionary
        """
        try:
            response = self.session.get(
                f"{self.server_url}/api/info",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """Close session"""
        self.session.close()


# Utility functions

def create_simple_system(
    isotope: str = '1H',
    num_spins: int = 2,
    j_coupling: float = 10.0,
    chemical_shift_range: float = 100.0
) -> SpinSystem:
    """
    Create a simple spin system.
    
    Args:
        isotope: Isotope type
        num_spins: Number of spins
        j_coupling: J-coupling constant (Hz)
        chemical_shift_range: Chemical shift range (Hz)
    
    Returns:
        SpinSystem object
    
    Example:
        >>> system = create_simple_system('1H', 2, 10.0, 100.0)
        >>> # Two-spin 1H system with 10 Hz J-coupling
    """
    system = SpinSystem(
        isotopes=[isotope] * num_spins,
        spins=[0.5] * num_spins,
        chemical_shifts=np.linspace(0, chemical_shift_range, num_spins).tolist()
    )
    
    # Add J-couplings between all pairs
    for i in range(num_spins):
        for j in range(i + 1, num_spins):
            system.j_couplings[f'{i+1}-{j+1}'] = j_coupling
    
    return system
