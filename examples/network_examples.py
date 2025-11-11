"""
Network Module Examples
=======================

Examples of using network communication modules for:
1. NMR Device Communication
2. Spinach Server Simulation
3. Data Transfer
4. Remote Control
"""

import numpy as np
import time
from nmr_processing_lib.network import (
    NMRDeviceClient,
    AcquisitionConfig,
    SpinachServerClient,
    SpinSystem,
    SimulationRequest,
    SimulationType,
    DataTransferClient,
    RemoteControlClient,
    CommandMessage
)


# =============================================================================
# Example 1: NMR Device Communication
# =============================================================================

def example_device_communication():
    """
    Example: Connect to NMR device and control acquisition.
    """
    print("\n=== Example 1: NMR Device Communication ===\n")
    
    # Create client
    client = NMRDeviceClient(
        host='192.168.1.100',  # Device IP address
        port=5000,
        timeout=10.0
    )
    
    # Set up callbacks
    def on_status_changed(status):
        print(f"[STATUS] Device status: {status.value}")
    
    def on_scan_received(scan_data):
        print(f"[DATA] Received scan: {len(scan_data)} points, peak={np.max(np.abs(scan_data)):.2e}")
    
    def on_message(msg):
        print(f"[INFO] {msg}")
    
    def on_error(msg):
        print(f"[ERROR] {msg}")
    
    client.on_status_changed = on_status_changed
    client.on_scan_received = on_scan_received
    client.on_message = on_message
    client.on_error = on_error
    
    try:
        # Connect
        print("Connecting to device...")
        client.connect()
        
        # Configure acquisition
        config = AcquisitionConfig(
            num_scans=16,
            sampling_rate=10000.0,
            num_points=50000,
            pulse_length=10.0,
            delay_time=1.0
        )
        
        print("\nSetting acquisition parameters...")
        client.set_parameters(config)
        
        # Start acquisition
        print("\nStarting acquisition...")
        client.start_acquisition()
        
        # Wait for acquisition
        print("Acquiring data... (press Ctrl+C to stop)")
        time.sleep(30)  # Acquire for 30 seconds
        
        # Stop acquisition
        print("\nStopping acquisition...")
        client.stop_acquisition()
        
        # Get status
        status = client.get_status()
        print(f"\nFinal status: {status}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Disconnect
        print("\nDisconnecting...")
        client.disconnect()


# =============================================================================
# Example 2: Spinach Server Simulation
# =============================================================================

def example_spinach_simulation():
    """
    Example: Submit simulation to Spinach server and get results.
    """
    print("\n=== Example 2: Spinach Server Simulation ===\n")
    
    # Create client
    client = SpinachServerClient(
        server_url='http://localhost:8000',
        api_key=None,  # Or your API key
        timeout=30.0
    )
    
    # Check server health
    print("Checking server health...")
    if not client.health_check():
        print("Server is not responding!")
        return
    
    print("Server is healthy!")
    
    # Get server info
    info = client.get_server_info()
    print(f"Server info: {info}")
    
    # Define spin system
    print("\nDefining spin system...")
    system = SpinSystem(
        isotopes=['1H', '1H'],
        spins=[0.5, 0.5],
        j_couplings={'1-2': 10.0},  # 10 Hz J-coupling
        chemical_shifts=[0.0, 50.0]  # 0 and 50 Hz chemical shifts
    )
    
    # Create simulation request
    request = SimulationRequest(
        simulation_type=SimulationType.ZULF,
        spin_system=system,
        magnet_field=0.0,  # ZULF
        sampling_rate=5000.0,
        num_points=10000,
        pulse_sequence="zg",
        add_noise=True,
        noise_level=0.01
    )
    
    try:
        # Submit and wait for results
        print("\nSubmitting simulation...")
        print("(This may take a while...)")
        
        response = client.submit_and_wait(
            request,
            timeout=300.0,  # 5 minutes
            poll_interval=2.0
        )
        
        print(f"\nSimulation completed!")
        print(f"Job ID: {response.job_id}")
        print(f"Computation time: {response.computation_time:.2f} seconds")
        print(f"Data points: {len(response.time_data)}")
        
        # Process results
        if response.time_data is not None:
            print(f"\nTime domain data:")
            print(f"  Shape: {response.time_data.shape}")
            print(f"  Max amplitude: {np.max(np.abs(response.time_data)):.2e}")
            
            # You can now use this data with nmr_processing_lib
            from nmr_processing_lib import DataInterface
            
            nmr_data = DataInterface.from_arrays(
                response.time_data,
                sampling_rate=request.sampling_rate,
                acquisition_time=len(response.time_data) / request.sampling_rate
            )
            
            print(f"\nNMRData object created successfully")
            print(f"  Can now process with nmr_processing_lib functions")
        
    except Exception as e:
        print(f"\nSimulation failed: {e}")
    
    finally:
        client.close()


# =============================================================================
# Example 3: Batch Simulations
# =============================================================================

def example_batch_simulations():
    """
    Example: Submit multiple simulations in batch.
    """
    print("\n=== Example 3: Batch Simulations ===\n")
    
    client = SpinachServerClient('http://localhost:8000')
    
    # Create multiple simulation requests
    requests = []
    
    for j_coupling in [5.0, 10.0, 15.0, 20.0]:
        system = SpinSystem(
            isotopes=['1H', '1H'],
            spins=[0.5, 0.5],
            j_couplings={'1-2': j_coupling},
            chemical_shifts=[0.0, 50.0]
        )
        
        request = SimulationRequest(
            simulation_type=SimulationType.ZULF,
            spin_system=system,
            magnet_field=0.0,
            sampling_rate=5000.0,
            num_points=10000
        )
        
        requests.append(request)
    
    print(f"Submitting {len(requests)} simulations...")
    
    # Submit batch
    job_ids = client.batch_submit(requests)
    print(f"Submitted {len(job_ids)} jobs")
    
    # Wait for all results
    print("\nWaiting for results...")
    results = client.batch_wait(job_ids, timeout=600.0)
    
    print(f"\nReceived {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\nSimulation {i+1}:")
        print(f"  Status: {result.status.value}")
        print(f"  Computation time: {result.computation_time:.2f}s")
        if result.time_data is not None:
            print(f"  Data points: {len(result.time_data)}")
    
    client.close()


# =============================================================================
# Example 4: Data Transfer
# =============================================================================

def example_data_transfer():
    """
    Example: Transfer files to/from server.
    """
    print("\n=== Example 4: Data Transfer ===\n")
    
    # Create client
    client = DataTransferClient(
        host='192.168.1.100',
        port=6000,
        compress=True  # Enable compression
    )
    
    try:
        # Connect
        print("Connecting to server...")
        client.connect()
        
        # Upload file with progress
        def upload_progress(sent, total):
            percent = 100 * sent / total
            print(f"\rUpload progress: {percent:.1f}% ({sent}/{total} bytes)", end='')
        
        print("\nUploading file...")
        success = client.upload_file(
            'experiment_data.dat',
            remote_path='remote_data.dat',
            progress_callback=upload_progress
        )
        
        if success:
            print("\n✓ Upload successful!")
        else:
            print("\n✗ Upload failed!")
        
        # Download file with progress
        def download_progress(received, total):
            percent = 100 * received / total
            print(f"\rDownload progress: {percent:.1f}% ({received}/{total} bytes)", end='')
        
        print("\nDownloading file...")
        success = client.download_file(
            'results.npy',
            'local_results.npy',
            progress_callback=download_progress
        )
        
        if success:
            print("\n✓ Download successful!")
        else:
            print("\n✗ Download failed!")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        client.disconnect()


# =============================================================================
# Example 5: Remote Control
# =============================================================================

def example_remote_control():
    """
    Example: Remote control of device/server.
    """
    print("\n=== Example 5: Remote Control ===\n")
    
    # Create client
    client = RemoteControlClient(
        host='192.168.1.100',
        port=7000
    )
    
    # Set up callbacks
    def on_status(status_msg):
        print(f"[STATUS] Device: {status_msg.device_id}, Status: {status_msg.status}")
        print(f"         Parameters: {status_msg.parameters}")
    
    def on_event(event_data):
        print(f"[EVENT] {event_data}")
    
    client.on_status_update = on_status
    client.on_event = on_event
    
    try:
        # Connect
        print("Connecting...")
        client.connect()
        
        # Send commands
        print("\nSending command: set_parameter")
        response = client.send_command(
            'set_parameter',
            parameters={'gain': 10, 'offset': 0.5}
        )
        print(f"Response: {response}")
        
        print("\nSending command: get_configuration")
        response = client.send_command('get_configuration')
        print(f"Configuration: {response}")
        
        print("\nSending command: start_monitoring")
        response = client.send_command('start_monitoring')
        print(f"Response: {response}")
        
        # Wait for status updates
        print("\nWaiting for status updates... (10 seconds)")
        time.sleep(10)
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        client.disconnect()


# =============================================================================
# Example 6: Device Discovery
# =============================================================================

def example_device_discovery():
    """
    Example: Discover NMR devices on network.
    """
    print("\n=== Example 6: Device Discovery ===\n")
    
    from nmr_processing_lib.network.device_client import discover_devices
    
    print("Discovering devices on network...")
    print("(This may take a few seconds...)")
    
    devices = discover_devices(
        port=5000,
        timeout=3.0
    )
    
    if devices:
        print(f"\nFound {len(devices)} device(s):")
        for i, device in enumerate(devices, 1):
            print(f"\nDevice {i}:")
            print(f"  IP: {device['ip']}")
            print(f"  Port: {device['port']}")
            print(f"  Name: {device['name']}")
            print(f"  Model: {device['model']}")
            print(f"  Firmware: {device['firmware']}")
    else:
        print("\nNo devices found")


# =============================================================================
# Example 7: Integrated Workflow
# =============================================================================

def example_integrated_workflow():
    """
    Example: Complete workflow - simulation, comparison with experiment, transfer results.
    """
    print("\n=== Example 7: Integrated Workflow ===\n")
    
    # Step 1: Run simulation
    print("Step 1: Running Spinach simulation...")
    sim_client = SpinachServerClient('http://localhost:8000')
    
    system = SpinSystem(
        isotopes=['1H', '1H'],
        spins=[0.5, 0.5],
        j_couplings={'1-2': 10.0},
        chemical_shifts=[0.0, 50.0]
    )
    
    sim_request = SimulationRequest(
        simulation_type=SimulationType.ZULF,
        spin_system=system,
        magnet_field=0.0,
        sampling_rate=5000.0,
        num_points=10000
    )
    
    try:
        sim_response = sim_client.submit_and_wait(sim_request, timeout=120.0)
        print(f"✓ Simulation completed in {sim_response.computation_time:.2f}s")
        
        # Step 2: Get experimental data from device
        print("\nStep 2: Getting experimental data...")
        device_client = NMRDeviceClient('192.168.1.100', 5000)
        
        experimental_data = []
        
        def on_scan(scan_data):
            experimental_data.append(scan_data)
        
        device_client.on_scan_received = on_scan
        device_client.connect()
        
        config = AcquisitionConfig(num_scans=1, sampling_rate=5000.0, num_points=10000)
        device_client.set_parameters(config)
        device_client.start_acquisition()
        
        time.sleep(5)  # Wait for acquisition
        
        device_client.stop_acquisition()
        device_client.disconnect()
        
        print(f"✓ Acquired {len(experimental_data)} scan(s)")
        
        # Step 3: Compare simulation vs experiment
        if experimental_data and sim_response.time_data is not None:
            print("\nStep 3: Comparing simulation vs experiment...")
            
            from nmr_processing_lib import DataInterface, apply_fft
            from nmr_processing_lib.quality import compare_snr
            
            # Process both datasets
            sim_data = DataInterface.from_arrays(sim_response.time_data, 5000.0)
            exp_data = DataInterface.from_arrays(experimental_data[0], 5000.0)
            
            freq_ax_sim, spec_sim = apply_fft(sim_data.time_data, 5000.0)
            freq_ax_exp, spec_exp = apply_fft(exp_data.time_data, 5000.0)
            
            # Compare SNR
            comparison = compare_snr(
                freq_ax_exp, spec_exp, spec_sim,
                peak_range=(-50, 50),
                noise_range=(200, 400)
            )
            
            print(f"\nSNR Comparison:")
            print(f"  Experimental SNR: {comparison['experimental_snr']:.1f}")
            print(f"  Simulated SNR: {comparison['simulated_snr']:.1f}")
            print(f"  Ratio: {comparison['ratio']:.2f}")
        
        # Step 4: Transfer results to analysis server
        print("\nStep 4: Transferring results...")
        
        # Save results locally
        import numpy as np
        np.save('simulation_result.npy', sim_response.time_data)
        if experimental_data:
            np.save('experimental_result.npy', experimental_data[0])
        
        # Upload to server
        transfer_client = DataTransferClient('192.168.1.200', 6000)
        transfer_client.connect()
        
        transfer_client.upload_file('simulation_result.npy')
        print("✓ Simulation results uploaded")
        
        if experimental_data:
            transfer_client.upload_file('experimental_result.npy')
            print("✓ Experimental results uploaded")
        
        transfer_client.disconnect()
        
        print("\n✓ Workflow completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
    
    finally:
        sim_client.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("NMR Processing Library - Network Module Examples")
    print("=" * 60)
    
    print("\nAvailable examples:")
    print("  1. Device Communication")
    print("  2. Spinach Simulation")
    print("  3. Batch Simulations")
    print("  4. Data Transfer")
    print("  5. Remote Control")
    print("  6. Device Discovery")
    print("  7. Integrated Workflow")
    
    choice = input("\nEnter example number (1-7): ").strip()
    
    examples = {
        '1': example_device_communication,
        '2': example_spinach_simulation,
        '3': example_batch_simulations,
        '4': example_data_transfer,
        '5': example_remote_control,
        '6': example_device_discovery,
        '7': example_integrated_workflow
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")
