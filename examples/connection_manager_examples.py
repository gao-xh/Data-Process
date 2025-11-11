"""
Connection Manager Examples
===========================

Examples of using ConnectionManager for local and cloud server connections.
"""

from nmr_processing_lib.network import (
    ConnectionManager,
    ConnectionProfile,
    ConnectionMode,
    ServerType,
    ConnectionStatus
)
import time


# =============================================================================
# Example 1: Basic Setup with Profiles
# =============================================================================

def example_basic_setup():
    """
    Example: Set up connection profiles for local and cloud servers.
    """
    print("\n=== Example 1: Basic Setup ===\n")
    
    manager = ConnectionManager()
    
    # Add local device profile
    local_device = ConnectionProfile(
        name="实验室NMR设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.LOCAL,
        host="192.168.1.100",
        port=5000,
        auto_reconnect=True
    )
    manager.add_profile(local_device)
    print(f"✓ Added local device profile: {local_device.name}")
    
    # Add cloud simulation server
    cloud_sim = ConnectionProfile(
        name="云端Spinach服务器",
        server_type=ServerType.SIMULATION,
        mode=ConnectionMode.CLOUD,
        host="spinach.example.com",
        port=443,
        use_ssl=True,
        api_key="your-api-key-here",
        timeout=30.0
    )
    manager.add_profile(cloud_sim)
    print(f"✓ Added cloud simulation profile: {cloud_sim.name}")
    
    # Add local simulation server (for testing)
    local_sim = ConnectionProfile(
        name="本地Spinach测试服务器",
        server_type=ServerType.SIMULATION,
        mode=ConnectionMode.LOCAL,
        host="localhost",
        port=8000,
        use_ssl=False
    )
    manager.add_profile(local_sim)
    print(f"✓ Added local simulation profile: {local_sim.name}")
    
    # Add cloud storage
    cloud_storage = ConnectionProfile(
        name="云端数据存储",
        server_type=ServerType.STORAGE,
        mode=ConnectionMode.CLOUD,
        host="storage.example.com",
        port=6000,
        username="user",
        password="pass"
    )
    manager.add_profile(cloud_storage)
    print(f"✓ Added cloud storage profile: {cloud_storage.name}")
    
    # Save profiles
    manager.save_profiles()
    print(f"\n✓ Saved {len(manager.profiles)} profiles to {manager.config_file}")
    
    # List profiles
    print("\n--- All Profiles ---")
    for profile in manager.list_profiles():
        print(f"  {profile.name}")
        print(f"    Type: {profile.server_type.value}")
        print(f"    Mode: {profile.mode.value}")
        print(f"    Host: {profile.host}:{profile.port}")
    
    return manager


# =============================================================================
# Example 2: Auto-Discovery of Local Devices
# =============================================================================

def example_auto_discovery():
    """
    Example: Automatically discover local devices on the network.
    """
    print("\n=== Example 2: Auto-Discovery ===\n")
    
    manager = ConnectionManager()
    
    # Set up discovery callback
    def on_discovered(profile):
        print(f"✓ Discovered: {profile.name}")
        print(f"  Host: {profile.host}:{profile.port}")
        print(f"  Model: {profile.metadata.get('model', 'Unknown')}")
    
    manager.on_profile_discovered = on_discovered
    
    # Discover devices
    print("Scanning network for NMR devices...")
    discovered = manager.discover_local_devices(
        server_type=ServerType.DEVICE,
        port=5000,
        timeout=3.0
    )
    
    if discovered:
        print(f"\n✓ Found {len(discovered)} device(s)")
        manager.save_profiles()
    else:
        print("\n✗ No devices found")
    
    return manager


# =============================================================================
# Example 3: Connect with User Choice
# =============================================================================

def example_user_choice():
    """
    Example: Let user choose between local and cloud connections.
    """
    print("\n=== Example 3: User Choice ===\n")
    
    manager = ConnectionManager()
    
    # Add profiles
    manager.add_profile(ConnectionProfile(
        name="本地设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.LOCAL,
        host="192.168.1.100",
        port=5000
    ))
    
    manager.add_profile(ConnectionProfile(
        name="云端设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.CLOUD,
        host="cloud.example.com",
        port=5000,
        use_ssl=True,
        api_key="cloud-api-key"
    ))
    
    # List available devices
    print("Available NMR Devices:")
    device_profiles = manager.list_profiles(server_type=ServerType.DEVICE)
    
    for i, profile in enumerate(device_profiles, 1):
        mode_text = "本地" if profile.mode == ConnectionMode.LOCAL else "云端"
        print(f"  {i}. {profile.name} ({mode_text})")
    
    # Simulate user choice
    print("\n用户选择: 1 (本地设备)")
    choice = 0  # User chose first option
    
    # Connect to selected profile
    selected_profile = device_profiles[choice]
    print(f"\nConnecting to {selected_profile.name}...")
    
    try:
        client = manager.connect(selected_profile.name)
        print(f"✓ Connected successfully")
        print(f"  Mode: {selected_profile.mode.value}")
        print(f"  Host: {selected_profile.host}")
        
        # Use the client
        # ... do something with client ...
        
        # Disconnect
        manager.disconnect(selected_profile.name)
        print(f"✓ Disconnected")
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
    
    return manager


# =============================================================================
# Example 4: Auto-Connect with Fallback
# =============================================================================

def example_auto_connect_fallback():
    """
    Example: Auto-connect with fallback from local to cloud.
    """
    print("\n=== Example 4: Auto-Connect with Fallback ===\n")
    
    manager = ConnectionManager()
    
    # Add hybrid mode profile (tries local first, then cloud)
    manager.add_profile(ConnectionProfile(
        name="混合模式设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.HYBRID,
        host="192.168.1.100",  # Local first
        port=5000
    ))
    
    # Add cloud backup
    manager.add_profile(ConnectionProfile(
        name="云端备份设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.CLOUD,
        host="cloud.example.com",
        port=5000,
        use_ssl=True
    ))
    
    # Auto-connect (will try local first, then cloud)
    print("Auto-connecting to NMR device...")
    print("Priority: Local > Cloud")
    
    try:
        client = manager.auto_connect(ServerType.DEVICE)
        
        # Find which profile was used
        for name, conn in manager.connections.items():
            if conn is client:
                profile = manager.get_profile(name)
                print(f"\n✓ Connected to: {profile.name}")
                print(f"  Mode: {profile.mode.value}")
                print(f"  Host: {profile.host}")
                break
        
    except Exception as e:
        print(f"\n✗ All connection attempts failed: {e}")
    
    finally:
        manager.disconnect_all()
    
    return manager


# =============================================================================
# Example 5: Connection Monitoring
# =============================================================================

def example_connection_monitoring():
    """
    Example: Monitor connection health and auto-reconnect.
    """
    print("\n=== Example 5: Connection Monitoring ===\n")
    
    manager = ConnectionManager()
    
    # Set up callbacks
    def on_status_changed(name, status):
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {status.value}")
    
    def on_error(name, error):
        print(f"[ERROR] {name}: {error}")
    
    manager.on_connection_changed = on_status_changed
    manager.on_connection_error = on_error
    
    # Add profile with auto-reconnect
    manager.add_profile(ConnectionProfile(
        name="监控设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.LOCAL,
        host="192.168.1.100",
        port=5000,
        auto_reconnect=True  # Enable auto-reconnect
    ))
    
    # Start monitoring
    print("Starting connection monitoring...")
    manager.start_monitoring(interval=5.0)  # Check every 5 seconds
    
    # Connect
    print("\nConnecting to device...")
    try:
        client = manager.connect("监控设备")
        print("✓ Connected")
        
        # Simulate running for a while
        print("\nMonitoring connection... (Ctrl+C to stop)")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        manager.stop_monitoring()
        manager.disconnect_all()
        print("✓ Monitoring stopped")
    
    return manager


# =============================================================================
# Example 6: Multiple Connections (Local + Cloud)
# =============================================================================

def example_multiple_connections():
    """
    Example: Connect to both local device and cloud simulation server.
    """
    print("\n=== Example 6: Multiple Connections ===\n")
    
    manager = ConnectionManager()
    
    # Add profiles
    manager.add_profile(ConnectionProfile(
        name="本地NMR设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.LOCAL,
        host="192.168.1.100",
        port=5000
    ))
    
    manager.add_profile(ConnectionProfile(
        name="云端仿真服务器",
        server_type=ServerType.SIMULATION,
        mode=ConnectionMode.CLOUD,
        host="spinach.example.com",
        port=443,
        use_ssl=True,
        api_key="spinach-api-key"
    ))
    
    try:
        # Connect to local device
        print("Connecting to local NMR device...")
        device_client = manager.connect("本地NMR设备")
        print("✓ Device connected")
        
        # Connect to cloud simulation
        print("\nConnecting to cloud simulation server...")
        sim_client = manager.connect("云端仿真服务器")
        print("✓ Simulation server connected")
        
        # Now you can use both
        print("\n--- Active Connections ---")
        for name in manager.connections:
            profile = manager.get_profile(name)
            status = manager.get_status(name)
            print(f"  {name}")
            print(f"    Type: {profile.server_type.value}")
            print(f"    Mode: {profile.mode.value}")
            print(f"    Status: {status.value}")
        
        # Workflow: Get experimental data from device,
        # compare with simulation from cloud
        print("\n✓ Ready for integrated workflow:")
        print("  - Acquire data from local device")
        print("  - Run simulation on cloud server")
        print("  - Compare results")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        manager.disconnect_all()
        print("\n✓ All connections closed")
    
    return manager


# =============================================================================
# Example 7: Context Manager Usage
# =============================================================================

def example_context_manager():
    """
    Example: Use ConnectionManager as context manager.
    """
    print("\n=== Example 7: Context Manager ===\n")
    
    # Create profiles
    profiles = [
        ConnectionProfile(
            name="设备A",
            server_type=ServerType.DEVICE,
            mode=ConnectionMode.LOCAL,
            host="192.168.1.100",
            port=5000
        ),
        ConnectionProfile(
            name="仿真服务器",
            server_type=ServerType.SIMULATION,
            mode=ConnectionMode.CLOUD,
            host="localhost",
            port=8000
        )
    ]
    
    # Use with context manager
    print("Using ConnectionManager with context manager...")
    
    with ConnectionManager() as manager:
        # Add profiles
        for profile in profiles:
            manager.add_profile(profile)
        
        # Connect
        try:
            device = manager.auto_connect(ServerType.DEVICE)
            print("✓ Connected to device")
            
            sim = manager.auto_connect(ServerType.SIMULATION)
            print("✓ Connected to simulation server")
            
            # Do work...
            print("\n... working ...")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Auto-disconnected on exit
    print("\n✓ All connections auto-closed")


# =============================================================================
# Example 8: Load/Save Configuration
# =============================================================================

def example_load_save_config():
    """
    Example: Save and load connection configurations.
    """
    print("\n=== Example 8: Load/Save Configuration ===\n")
    
    # Create manager with custom config file
    config_file = "my_nmr_connections.json"
    manager = ConnectionManager(config_file=config_file)
    
    # Add some profiles
    manager.add_profile(ConnectionProfile(
        name="实验室主设备",
        server_type=ServerType.DEVICE,
        mode=ConnectionMode.LOCAL,
        host="192.168.1.100",
        port=5000,
        metadata={'location': '3楼实验室', 'user': '张三'}
    ))
    
    manager.add_profile(ConnectionProfile(
        name="云端计算集群",
        server_type=ServerType.SIMULATION,
        mode=ConnectionMode.CLOUD,
        host="hpc.example.com",
        port=443,
        use_ssl=True,
        api_key="hpc-token-123",
        metadata={'region': 'us-west', 'tier': 'premium'}
    ))
    
    # Save
    print("Saving configuration...")
    manager.save_profiles(config_file)
    print(f"✓ Saved to {config_file}")
    
    # Create new manager and load
    print("\nLoading configuration...")
    new_manager = ConnectionManager(config_file=config_file)
    
    print(f"✓ Loaded {len(new_manager.profiles)} profile(s):")
    for profile in new_manager.list_profiles():
        print(f"  - {profile.name} ({profile.server_type.value})")
        if profile.metadata:
            print(f"    Metadata: {profile.metadata}")
    
    return new_manager


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Connection Manager Examples")
    print("=" * 60)
    
    print("\nAvailable examples:")
    print("  1. Basic Setup with Profiles")
    print("  2. Auto-Discovery of Local Devices")
    print("  3. Connect with User Choice")
    print("  4. Auto-Connect with Fallback")
    print("  5. Connection Monitoring")
    print("  6. Multiple Connections (Local + Cloud)")
    print("  7. Context Manager Usage")
    print("  8. Load/Save Configuration")
    
    choice = input("\nEnter example number (1-8): ").strip()
    
    examples = {
        '1': example_basic_setup,
        '2': example_auto_discovery,
        '3': example_user_choice,
        '4': example_auto_connect_fallback,
        '5': example_connection_monitoring,
        '6': example_multiple_connections,
        '7': example_context_manager,
        '8': example_load_save_config
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")
