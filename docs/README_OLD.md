# NMR Data Processing Library

一个模块化的NMR数据处理函数库，设计用于与PySide6 UI无缝整合。

## 🎯 项目目标

1. ✅ **模块化设计**: 核心算法与UI完全分离
2. ✅ **多数据源支持**: 文件、实时采集、内存数组
3. ✅ **灵活参数管理**: 类型安全的参数系统
4. ✅ **易于整合**: 提供清晰的UI接口
5. ✅ **向后兼容**: 支持现有Notebook工作流

## 📁 项目结构

```
DUI_10_8/
├── nmr_processing_lib/          # 核心函数库
│   ├── __init__.py             # 包导出
│   ├── core/                   # 核心模块
│   │   ├── data_io.py         # 数据I/O接口 ✅
│   │   ├── parameters.py      # 参数管理 ✅
│   │   └── transforms.py      # FFT/相位校正 ✅
│   ├── processing/             # 数据处理
│   │   ├── filtering.py       # Savgol/窗函数 (待实现)
│   │   ├── preprocessing.py   # 截断/零填充 (待实现)
│   │   └── postprocessing.py  # 线展宽/基线 (待实现)
│   ├── quality/                # 质量控制
│   │   ├── snr.py            # SNR计算 (待实现)
│   │   └── scan_selection.py # 扫描筛选 (待实现)
│   ├── analysis/               # 高级分析
│   │   ├── fitting.py        # Lorentzian拟合 (待实现)
│   │   └── decomposition.py  # SVD/Matrix Pencil (待实现)
│   └── utils/                  # 工具函数
│       └── helpers.py         (待实现)
│
├── nmr_ui/                      # UI程序 (待开发)
│   ├── main_window.py          # 主窗口
│   ├── widgets/                # 自定义控件
│   └── dialogs/                # 对话框
│
├── examples/                    # 示例代码
│   └── usage_examples.py       # 使用示例 ✅
│
├── tests/                       # 单元测试 (待开发)
│
├── ARCHITECTURE.md              # 架构文档 ✅
└── README.md                    # 本文件
```

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib PySide6
```

### 基本用法

```python
from nmr_processing_lib import DataInterface, ParameterManager

# 1. 加载数据
data = DataInterface.from_nmrduino_folder("path/to/experiment")

# 2. 设置参数
manager = ParameterManager()
manager.processing.savgol_window = 301
manager.processing.zero_fill_factor = 2.7

# 3. 处理数据 (完整流程待实现)
# processed_data = process_pipeline(data, manager.processing)

# 4. 保存参数
manager.save_all("my_parameters.json")
```

## 📚 核心概念

### 1. 数据接口 (DataInterface)

提供统一的数据加载接口，支持多种数据源：

```python
# 从文件加载
data = DataInterface.from_nmrduino_folder(folder_path, scans=0)

# 从实时采集加载（未来扩展）
data = DataInterface.from_live_acquisition(time_data, sr, acq_time)

# 从内存数组加载
data = DataInterface.from_arrays(numpy_array, sampling_rate)
```

### 2. NMRData 对象

核心数据容器：

```python
@dataclass
class NMRData:
    time_data: np.ndarray        # 时域数据
    sampling_rate: float          # 采样率 (Hz)
    acquisition_time: float       # 采集时间 (s)
    freq_data: np.ndarray        # 频域数据（FFT后）
    freq_axis: np.ndarray        # 频率轴
    source: DataSource           # 数据来源
    processing_steps: List[str]  # 处理历史
```

### 3. 参数管理

类型安全的参数系统：

```python
@dataclass
class ProcessingParameters:
    savgol_window: int = 300
    savgol_order: int = 2
    trunc_start: int = 10
    trunc_end: int = 10
    apodization_t2: float = 0.0
    zero_fill_factor: float = 0.0
    gaussian_fwhm: float = 0.0
    # ... 更多参数
```

支持验证、序列化和预设：

```python
manager = ParameterManager()

# 验证参数
errors = manager.validate_current()

# 加载预设
manager.load_preset("high_resolution")

# 保存/加载
manager.save_all("params.json")
manager.load_all("params.json")
```

## 🔌 UI整合

### 数据绑定示例

```python
# PySide6 UI中的使用
class ProcessingPanel(QWidget):
    def __init__(self):
        self.param_manager = ParameterManager()
        
        # 双向绑定SpinBox
        self.savgol_spinbox.valueChanged.connect(
            lambda v: setattr(
                self.param_manager.processing, 
                'savgol_window', 
                v
            )
        )
        
        # 从参数初始化UI
        self.savgol_spinbox.setValue(
            self.param_manager.processing.savgol_window
        )
```

### 处理流程示例

```python
def on_run_processing(self):
    # 获取参数
    params = self.param_manager.processing
    
    # 验证
    errors = params.validate()
    if errors:
        QMessageBox.warning(self, "Invalid", "\n".join(errors))
        return
    
    # 在Worker线程中处理
    self.worker = ProcessingWorker(self.data, params)
    self.worker.finished.connect(self.update_plot)
    self.worker.start()
```

## 📖 详细文档

- **架构设计**: 查看 [ARCHITECTURE.md](ARCHITECTURE.md)
- **使用示例**: 查看 [examples/usage_examples.py](examples/usage_examples.py)
- **API文档**: (待生成)

## 🛠️ 开发状态

### ✅ 已完成
- [x] 数据I/O接口
- [x] 参数管理系统
- [x] FFT/相位校正
- [x] 架构文档
- [x] 使用示例

### 🚧 进行中
- [ ] Savgol滤波模块
- [ ] 预处理模块（截断、零填充、apodization）
- [ ] 后处理模块（线展宽、基线校正）

### 📋 待开发
- [ ] SNR计算
- [ ] 扫描筛选（整合fid_select）
- [ ] Lorentzian拟合
- [ ] SVD/Matrix Pencil分析
- [ ] PySide6 UI组件
- [ ] 单元测试

## 🤝 与现有代码的关系

### 从 nmrduino_util.py 提取的功能
- ✅ `nmrduino_dat_interp` → `load_nmrduino_data`
- ✅ `select_folder` → `select_folder_dialog`
- ⏳ `snr_calc` → `quality/snr.py` (待迁移)
- ⏳ `bandpass_data` → `transforms.py` (已实现部分)

### 从 fid_select.py 提取的功能
- ⏳ `calculate_difference_sums` → `scan_selection.py` (待迁移)
- ⏳ `filter_scans_by_threshold` → `scan_selection.py` (待迁移)
- ⏳ 交互式筛选UI → `nmr_ui/dialogs/scan_filter.py` (待开发)

### 从 Notebook 提取的功能
- ⏳ Savgol滤波 → `filtering.py` (待迁移)
- ⏳ 时域处理 → `preprocessing.py` (待迁移)
- ⏳ 高斯展宽 → `postprocessing.py` (待迁移)

## 💡 设计原则

1. **分离关注点**: 算法与UI完全分离
2. **接口优先**: 提供清晰的数据接口
3. **类型安全**: 使用dataclass和类型注解
4. **易于测试**: 纯函数设计，易于单元测试
5. **向后兼容**: 支持现有工作流

## 🎯 下一步计划

1. **完成核心处理模块** (本周)
   - [ ] `processing/filtering.py`
   - [ ] `processing/preprocessing.py`
   - [ ] `processing/postprocessing.py`

2. **实现质量控制** (下周)
   - [ ] `quality/snr.py`
   - [ ] `quality/scan_selection.py`

3. **开发UI组件** (后续)
   - [ ] ProcessingPanel
   - [ ] 整合到Spinach UI

4. **高级功能** (可选)
   - [ ] Lorentzian拟合
   - [ ] SVD滤波
   - [ ] 批处理模式

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Email: your.email@example.com

## 📄 许可证

MIT License

---

**最后更新**: 2025-10-08  
**版本**: 1.0.0-alpha
