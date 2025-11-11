# Network Communication Module

网络通信模块，用于连接NMR设备主机和Spinach仿真服务器。

## 模块概述

该模块提供五个主要功能：

1. **NMR设备通信** (`device_client.py`) - 与NMR设备主机的TCP/IP通信
2. **Spinach仿真服务器** (`simulation_client.py`) - 与Spinach仿真服务器的HTTP REST API通信
3. **数据传输** (`data_transfer.py`) - 高性能文件传输（支持压缩）
4. **远程控制** (`remote_control.py`) - 双向远程控制协议
5. **连接管理** (`connection_manager.py`) - **本地/云端连接管理（推荐使用）**

## 快速开始

### 0. 连接管理器（推荐方式）

**让用户选择本地或云端服务器：**

```python
from nmr_processing_lib.network import (
    ConnectionManager,
    ConnectionProfile,
    ConnectionMode,
    ServerType
)

# 创建管理器
manager = ConnectionManager()

# 添加本地设备配置
local_device = ConnectionProfile(
    name="实验室NMR设备",
    server_type=ServerType.DEVICE,
    mode=ConnectionMode.LOCAL,
    host="192.168.1.100",
    port=5000
)
manager.add_profile(local_device)

# 添加云端设备配置
cloud_device = ConnectionProfile(
    name="云端NMR设备",
    server_type=ServerType.DEVICE,
    mode=ConnectionMode.CLOUD,
    host="cloud.example.com",
    port=5000,
    use_ssl=True,
    api_key="your-api-key"
)
manager.add_profile(cloud_device)

# 方式1: 让用户选择
profiles = manager.list_profiles(server_type=ServerType.DEVICE)
for i, p in enumerate(profiles):
    print(f"{i+1}. {p.name} ({'本地' if p.mode == ConnectionMode.LOCAL else '云端'})")

choice = int(input("选择设备: ")) - 1
client = manager.connect(profiles[choice].name)

# 方式2: 自动选择（优先本地，失败后云端）
client = manager.auto_connect(ServerType.DEVICE)

# 方式3: 自动发现本地设备
discovered = manager.discover_local_devices()
if discovered:
    client = manager.connect(discovered[0].name)

# 保存配置供下次使用
manager.save_profiles()
```

### 1. NMR设备通信

```python
from nmr_processing_lib.network import NMRDeviceClient, AcquisitionConfig

# 创建客户端
client = NMRDeviceClient(host='192.168.1.100', port=5000)

# 设置回调
def on_scan_received(scan_data):
    print(f"Received scan: {len(scan_data)} points")

client.on_scan_received = on_scan_received

# 连接和采集
client.connect()
config = AcquisitionConfig(
    num_scans=16,
    sampling_rate=10000.0,
    num_points=50000
)
client.set_parameters(config)
client.start_acquisition()

# ... 等待数据采集 ...

client.stop_acquisition()
client.disconnect()
```

### 2. Spinach仿真

```python
from nmr_processing_lib.network import SpinachServerClient, SpinSystem, SimulationRequest

# 创建客户端
client = SpinachServerClient('http://localhost:8000')

# 定义自旋系统
system = SpinSystem(
    isotopes=['1H', '1H'],
    spins=[0.5, 0.5],
    j_couplings={'1-2': 10.0},
    chemical_shifts=[0.0, 50.0]
)

# 提交仿真
request = SimulationRequest(
    simulation_type=SimulationType.ZULF,
    spin_system=system,
    magnet_field=0.0,
    sampling_rate=5000.0,
    num_points=10000
)

response = client.submit_and_wait(request, timeout=300.0)
print(f"Simulation completed: {len(response.time_data)} points")

# 转换为NMRData对象
from nmr_processing_lib import DataInterface
nmr_data = DataInterface.from_arrays(response.time_data, request.sampling_rate)
```

### 3. 设备发现

```python
from nmr_processing_lib.network.device_client import discover_devices

# 自动发现网络上的设备
devices = discover_devices(port=5000, timeout=3.0)

for device in devices:
    print(f"Found: {device['name']} at {device['ip']}")
```

## API文档

### ConnectionManager（推荐使用）

连接管理器是**推荐的连接方式**，它统一管理本地和云端服务器连接。

**主要方法：**
- `add_profile(profile)` - 添加连接配置
- `remove_profile(name)` - 删除连接配置
- `list_profiles(server_type, mode)` - 列出所有配置
- `connect(profile_name)` - 连接到指定服务器
- `disconnect(profile_name)` - 断开连接
- `auto_connect(server_type, mode)` - 自动连接（智能选择）
- `discover_local_devices()` - 自动发现本地设备
- `save_profiles()` / `load_profiles()` - 保存/加载配置
- `start_monitoring()` - 开启连接健康监控
- `stop_monitoring()` - 停止监控

**连接模式：**
```python
class ConnectionMode(Enum):
    LOCAL = "local"      # 本地设备
    CLOUD = "cloud"      # 云端服务器
    HYBRID = "hybrid"    # 混合（优先本地，失败后云端）
    AUTO = "auto"        # 自动选择
```

**服务器类型：**
```python
class ServerType(Enum):
    DEVICE = "device"           # NMR采集设备
    SIMULATION = "simulation"   # 仿真服务器
    STORAGE = "storage"        # 数据存储
    CONTROL = "control"        # 远程控制
```

**配置数据类：**
```python
@dataclass
class ConnectionProfile:
    name: str                    # 配置名称
    server_type: ServerType      # 服务器类型
    mode: ConnectionMode         # 连接模式
    host: str                    # 主机地址
    port: int                    # 端口号
    use_ssl: bool = False       # 是否使用SSL
    api_key: str = None         # API密钥（云端）
    username: str = None        # 用户名
    password: str = None        # 密码
    timeout: float = 10.0       # 超时时间
    auto_reconnect: bool = True # 自动重连
    metadata: dict = None       # 额外元数据
```

**回调函数：**
```python
manager.on_connection_changed = lambda name, status: print(f"{name}: {status}")
manager.on_connection_error = lambda name, error: print(f"Error: {error}")
manager.on_profile_discovered = lambda profile: print(f"Found: {profile.name}")
```

**使用场景：**

1. **用户选择本地或云端：**
```python
manager = ConnectionManager()

# 列出所有设备
devices = manager.list_profiles(server_type=ServerType.DEVICE)
for i, dev in enumerate(devices):
    mode_text = "本地" if dev.mode == ConnectionMode.LOCAL else "云端"
    print(f"{i+1}. {dev.name} ({mode_text})")

# 用户选择
choice = int(input("选择: ")) - 1
client = manager.connect(devices[choice].name)
```

2. **自动发现并连接：**
```python
manager = ConnectionManager()

# 发现本地设备
discovered = manager.discover_local_devices()

if discovered:
    # 连接第一个发现的设备
    client = manager.connect(discovered[0].name)
else:
    # 本地没有设备，使用云端
    client = manager.auto_connect(ServerType.DEVICE, mode=ConnectionMode.CLOUD)
```

3. **混合模式（优先本地，自动切换云端）：**
```python
manager = ConnectionManager()

# 添加混合模式配置
manager.add_profile(ConnectionProfile(
    name="智能设备",
    server_type=ServerType.DEVICE,
    mode=ConnectionMode.HYBRID,
    host="192.168.1.100",  # 优先尝试本地
    port=5000
))

manager.add_profile(ConnectionProfile(
    name="云端备份",
    server_type=ServerType.DEVICE,
    mode=ConnectionMode.CLOUD,
    host="cloud.example.com",
    port=5000,
    use_ssl=True
))

# 自动选择（本地失败自动切换云端）
client = manager.auto_connect(ServerType.DEVICE)
```

4. **保存配置供下次使用：**
```python
manager = ConnectionManager()

# 添加配置...
manager.add_profile(...)
manager.add_profile(...)

# 保存
manager.save_profiles()  # 默认保存到 ~/.nmr_connections.json

# 下次直接加载
new_manager = ConnectionManager()  # 自动加载已保存的配置
```

5. **连接健康监控：**
```python
manager = ConnectionManager()

# 设置回调
manager.on_connection_changed = lambda name, status: print(f"{name}: {status.value}")

# 连接
client = manager.connect("设备A")

# 开启监控（每5秒检查一次连接状态）
manager.start_monitoring(interval=5.0)

# 如果连接断开，会自动尝试重连（如果 auto_reconnect=True）
```

---

### NMRDeviceClient

**主要方法：**
- `connect()` - 连接到设备
- `disconnect()` - 断开连接
- `start_acquisition()` - 开始采集
- `stop_acquisition()` - 停止采集
- `set_parameters(config: AcquisitionConfig)` - 设置采集参数
- `calibrate()` - 校准设备
- `reset()` - 重置设备

**回调函数：**
- `on_status_changed(status)` - 状态变化时调用
- `on_scan_received(scan_data)` - 接收到扫描数据时调用
- `on_message(message)` - 接收到消息时调用
- `on_error(error)` - 发生错误时调用

**数据类：**
```python
@dataclass
class AcquisitionConfig:
    num_scans: int = 1
    sampling_rate: float = 10000.0
    num_points: int = 50000
    pulse_length: float = 10.0
    delay_time: float = 1.0
```

### SpinachServerClient

**主要方法：**
- `submit_simulation(request)` - 提交仿真任务
- `get_status(job_id)` - 查询任务状态
- `get_result(job_id)` - 获取仿真结果
- `submit_and_wait(request, timeout)` - 提交并等待完成
- `cancel_simulation(job_id)` - 取消仿真
- `batch_submit(requests)` - 批量提交
- `batch_wait(job_ids, timeout)` - 批量等待

**工具函数：**
- `health_check()` - 检查服务器健康状态
- `get_server_info()` - 获取服务器信息
- `create_simple_system(...)` - 创建简单自旋系统

**数据类：**
```python
@dataclass
class SpinSystem:
    isotopes: list[str]
    spins: list[float]
    j_couplings: dict[str, float]
    chemical_shifts: list[float]

@dataclass
class SimulationRequest:
    simulation_type: SimulationType
    spin_system: SpinSystem
    magnet_field: float
    sampling_rate: float
    num_points: int
    pulse_sequence: str = "zg"
    # ... 更多参数 ...

@dataclass
class SimulationResponse:
    job_id: str
    status: SimulationStatus
    time_data: Optional[np.ndarray]
    freq_data: Optional[np.ndarray]
    computation_time: float
```

### DataTransferClient

**主要方法：**
- `connect()` - 连接到服务器
- `disconnect()` - 断开连接
- `upload_file(local_path, remote_path, progress_callback)` - 上传文件
- `download_file(remote_path, local_path, progress_callback)` - 下载文件

**进度回调：**
```python
def progress_callback(bytes_transferred: int, total_bytes: int):
    percent = 100 * bytes_transferred / total_bytes
    print(f"Progress: {percent:.1f}%")
```

### RemoteControlClient

**主要方法：**
- `connect()` - 连接到服务器
- `disconnect()` - 断开连接
- `send_command(command, parameters, timeout)` - 发送命令（阻塞）
- `send_command_async(command, parameters, callback)` - 发送命令（异步）

**回调函数：**
- `on_status_update(status_message)` - 状态更新时调用
- `on_event(event_data)` - 接收到事件时调用

## 服务器端实现

如果需要实现服务器端，可以使用提供的服务器类：

### DataTransferServer

```python
from nmr_processing_lib.network import DataTransferServer

server = DataTransferServer(
    host='0.0.0.0',
    port=6000,
    storage_dir='./storage'
)

server.start()
# ... 服务器运行中 ...
server.stop()
```

### RemoteControlServer

```python
from nmr_processing_lib.network import RemoteControlServer

def handle_command(command_msg):
    # 处理命令
    if command_msg.command == 'get_status':
        return {'status': 'running', 'uptime': 3600}
    return {'error': 'Unknown command'}

server = RemoteControlServer(
    host='0.0.0.0',
    port=7000,
    command_handler=handle_command
)

server.start()
# ... 服务器运行中 ...
server.stop()
```

## 网络协议

### 设备通信协议 (TCP/IP)

**消息格式（JSON）：**
```json
{
    "type": "command",
    "command": "start_acquisition",
    "parameters": {...},
    "timestamp": 1234567890.123
}
```

**命令类型：**
- `START_ACQUISITION` - 开始采集
- `STOP_ACQUISITION` - 停止采集
- `SET_PARAMETERS` - 设置参数
- `GET_STATUS` - 获取状态
- `GET_DATA` - 获取数据
- `CALIBRATE` - 校准
- `RESET` - 重置

**二进制数据格式：**
```
Header: [4 bytes: data length] + [4 bytes: num points]
Body: [numpy array bytes]
```

### Spinach服务器协议 (HTTP REST API)

**端点：**
- `POST /api/simulations` - 提交仿真
- `GET /api/simulations/{job_id}` - 查询状态
- `GET /api/simulations/{job_id}/result` - 获取结果
- `DELETE /api/simulations/{job_id}` - 取消任务
- `GET /health` - 健康检查
- `GET /info` - 服务器信息

**请求示例：**
```json
{
    "simulation_type": "zulf",
    "spin_system": {
        "isotopes": ["1H", "1H"],
        "spins": [0.5, 0.5],
        "j_couplings": {"1-2": 10.0},
        "chemical_shifts": [0.0, 50.0]
    },
    "magnet_field": 0.0,
    "sampling_rate": 5000.0,
    "num_points": 10000
}
```

**响应示例：**
```json
{
    "job_id": "abc123",
    "status": "completed",
    "time_data": "<base64 encoded numpy array>",
    "computation_time": 12.34
}
```

## 错误处理

所有网络操作都应使用try-except处理异常：

```python
from nmr_processing_lib.network import NMRDeviceClient

client = NMRDeviceClient('192.168.1.100', 5000)

try:
    client.connect()
    client.start_acquisition()
    # ... 操作 ...
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Operation timed out: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    client.disconnect()
```

## 线程安全

- 所有客户端类都是线程安全的
- 回调函数在后台线程中执行，需要注意线程同步
- 服务器类支持多客户端并发连接

## 性能优化

### 数据传输优化

```python
# 启用压缩（适合文本/数值数据）
client = DataTransferClient(host='...', compress=True)

# 调整缓冲区大小
client.upload_file('large_file.dat', buffer_size=1024*1024)  # 1MB buffer
```

### 批量仿真优化

```python
# 批量提交避免多次网络往返
job_ids = sim_client.batch_submit(requests)
results = sim_client.batch_wait(job_ids, timeout=600.0)
```

## 安全注意事项

1. **身份验证**：在生产环境中使用API密钥或其他身份验证机制
2. **加密**：敏感数据传输应使用SSL/TLS
3. **输入验证**：服务器端应验证所有输入参数
4. **超时设置**：设置合理的超时避免死锁
5. **错误日志**：记录所有错误用于调试

## 示例用法

查看 `examples/network_examples.py` 获取完整示例：

1. 设备通信示例
2. Spinach仿真示例
3. 批量仿真示例
4. 数据传输示例
5. 远程控制示例
6. 设备发现示例
7. 集成工作流示例

## 集成到UI

将网络模块集成到PySide6 UI：

```python
from PySide6.QtCore import QThread, Signal
from nmr_processing_lib.network import NMRDeviceClient

class AcquisitionThread(QThread):
    scan_received = Signal(object)  # 发射扫描数据
    status_changed = Signal(str)    # 发射状态变化
    
    def __init__(self, host, port):
        super().__init__()
        self.client = NMRDeviceClient(host, port)
        self.client.on_scan_received = self.scan_received.emit
        self.client.on_status_changed = lambda s: self.status_changed.emit(s.value)
    
    def run(self):
        self.client.connect()
        self.client.start_acquisition()
        # ... 采集逻辑 ...
        self.client.disconnect()
```

## 常见问题

**Q: 如何找到网络上的设备？**
A: 使用 `discover_devices()` 函数自动发现。

**Q: 仿真超时怎么办？**
A: 增加 `timeout` 参数或检查服务器负载。

**Q: 如何同时采集和仿真？**
A: 在不同线程中运行设备客户端和仿真客户端。

**Q: 数据传输很慢？**
A: 启用压缩 (`compress=True`) 或增加缓冲区大小。

**Q: 如何处理网络中断？**
A: 实现重连逻辑，捕获 `ConnectionError` 并重试。

## 依赖要求

```
numpy>=1.20.0
requests>=2.25.0  # 用于HTTP客户端
```

## 许可证

与nmr_processing_lib主库相同。
