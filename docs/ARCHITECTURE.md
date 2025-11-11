# NMR数据处理系统架构规划
## Architecture Planning for NMR Processing System

**日期**: 2025-10-08  
**版本**: 1.0  
**目标**: 创建可与PySide6 UI整合的模块化NMR数据处理库

---

## 📋 总体架构

```
项目结构:
├── nmr_processing_lib/          # 核心函数库（纯Python，无UI依赖）
│   ├── core/                    # 核心模块
│   ├── processing/              # 数据处理
│   ├── quality/                 # 质量控制
│   ├── analysis/                # 高级分析
│   └── utils/                   # 工具函数
│
├── nmr_ui/                      # UI程序（基于PySide6）
│   ├── main_window.py           # 主窗口（整合到Spinach UI）
│   ├── widgets/                 # 自定义控件
│   └── dialogs/                 # 对话框
│
└── tests/                       # 单元测试
```

---

## 🎯 模块职责划分

### 1. **函数库 (nmr_processing_lib/)** - 核心算法与数据处理

#### ✅ 应该包含的内容：

1. **数据I/O** (`core/data_io.py`)
   - ✓ 从文件加载数据 (`load_nmrduino_data`)
   - ✓ 从内存/实时数据创建NMRData对象 (`from_live_acquisition`, `from_arrays`)
   - ✓ 数据缓存机制
   - ✓ 导出功能 (`save_spectrum`)
   - **接口设计**: `DataInterface` 提供统一的数据获取方式

2. **参数管理** (`core/parameters.py`)
   - ✓ 参数数据类 (`ProcessingParameters`, `AcquisitionParameters`)
   - ✓ 参数验证
   - ✓ JSON序列化/反序列化
   - ✓ 预设模板（高分辨率/高灵敏度等）
   - ✓ 与Notebook格式兼容

3. **变换** (`core/transforms.py`)
   - ✓ FFT/IFFT
   - ✓ 相位校正
   - ✓ 频率轴生成
   - ✓ 频率范围提取
   - ✓ 带通滤波
   - ✓ 多谱图组合（用于multi-system）

4. **预处理** (`processing/preprocessing.py`)
   - 时域截断
   - 零填充
   - Apodization（指数衰减）
   - DC offset移除

5. **滤波** (`processing/filtering.py`)
   - Savgol滤波
   - 窗函数（Hanning, Hamming, Blackman等）
   - 自适应滤波

6. **后处理** (`processing/postprocessing.py`)
   - 高斯线展宽
   - 基线校正
   - 归一化

7. **质量控制** (`quality/`)
   - SNR计算 (`snr.py`)
   - 坏扫描筛选 (`scan_selection.py` - 整合fid_select功能)
   - 峰检测

8. **高级分析** (`analysis/`)
   - Lorentzian拟合 (`fitting.py`)
   - SVD滤波 (`decomposition.py`)
   - Matrix Pencil方法 (`decomposition.py`)

#### ❌ 不应该包含的内容：
- 任何GUI代码（QWidget, QPushButton等）
- 用户交互逻辑
- 具体的UI布局
- matplotlib的交互式工具（如sliders）

---

### 2. **UI程序 (nmr_ui/)** - 用户界面与交互

#### ✅ 应该包含的内容：

1. **主窗口** (`main_window.py`)
   - 整合到现有Spinach UI的Tab系统
   - 菜单栏（File, View, Help）
   - 状态栏
   - 日志显示

2. **数据处理面板** (`widgets/processing_panel.py`)
   - 参数输入控件（SpinBox, Slider等）
   - 实时参数调整
   - "Run Processing"按钮
   - 进度显示

3. **质量控制面板** (`widgets/quality_panel.py`)
   - 扫描筛选界面
   - SNR显示
   - 坏扫描标记

4. **绘图控件** (`widgets/plot_widget.py`)
   - 复用现有的 `PlotWidget` 类
   - 添加NMR特定功能（频率范围选择、峰标记等）
   - 多谱图叠加显示

5. **对话框** (`dialogs/`)
   - 文件/文件夹选择
   - 扫描筛选（交互式阈值调整）
   - 参数预设选择
   - 导出选项

#### ❌ 不应该包含的内容：
- 核心算法实现
- 数据处理逻辑
- 文件格式解析

---

## 🔌 关键接口设计

### 接口1: 数据输入接口

```python
# 函数库提供多种数据源接口
class DataInterface:
    @staticmethod
    def from_nmrduino_folder(folder_path, scans=0) -> NMRData
    
    @staticmethod
    def from_live_acquisition(time_data, sampling_rate, acq_time) -> NMRData
    
    @staticmethod
    def from_arrays(time_data, sampling_rate) -> NMRData

# UI层调用示例
# 方式1: 从文件
data = DataInterface.from_nmrduino_folder(selected_folder, scans=0)

# 方式2: 从实时采集（未来扩展）
data = DataInterface.from_live_acquisition(live_buffer, sr, acq_t)

# 方式3: 从内存数组
data = DataInterface.from_arrays(numpy_array, 8333.0)
```

**优势**: 
- UI不关心数据来源
- 函数库提供统一的`NMRData`对象
- 易于扩展新的数据源

---

### 接口2: 参数绑定接口

```python
# 函数库提供参数类
params = ProcessingParameters()

# UI控件双向绑定
# 方式1: 直接绑定
spinbox.valueChanged.connect(lambda v: setattr(params, 'savgol_window', v))
spinbox.setValue(params.savgol_window)

# 方式2: 使用ParameterManager
manager = ParameterManager()
manager.processing.savgol_window = spinbox.value()

# 加载/保存
manager.save_all("parameters.json")
manager.load_all("parameters.json")
```

---

### 接口3: 处理流水线接口

```python
# 函数库提供处理函数
from nmr_processing_lib import (
    savgol_filter_nmr,
    truncate_time_domain,
    apply_apodization,
    zero_filling,
    apply_fft
)

# UI层组织处理流程
def process_data(data: NMRData, params: ProcessingParameters):
    # 1. Savgol滤波
    if params.savgol_enabled:
        data.time_data = savgol_filter_nmr(
            data.time_data, 
            params.savgol_window, 
            params.savgol_order
        )
    
    # 2. 截断
    data.time_data = truncate_time_domain(
        data.time_data,
        params.trunc_start,
        params.trunc_end
    )
    
    # 3. Apodization
    if params.apodization_t2 > 0:
        data.time_data = apply_apodization(
            data.time_data,
            data.acquisition_time,
            params.apodization_t2
        )
    
    # 4. 零填充
    if params.zero_fill_factor > 0:
        data.time_data = zero_filling(
            data.time_data,
            params.zero_fill_factor
        )
    
    # 5. FFT
    freq_axis, freq_data = apply_fft(data)
    
    return data
```

---

## 🎨 UI整合方案

### 方案1: 在Spinach UI中添加新Tab

```python
# 在DualSystemWindow中添加
class DualSystemWindow(QMainWindow):
    def setup_ui(self):
        # ... 现有代码 ...
        
        # 添加NMR Processing Tab
        nmr_tab = NMRProcessingWidget()  # 新建的处理控件
        self.main_tabs.addTab(nmr_tab, "NMR Processing")
```

### 方案2: 独立窗口（通过菜单打开）

```python
# 在菜单栏添加
def setup_menu(self):
    # ... 现有菜单 ...
    
    tools_menu = menubar.addMenu("Tools")
    nmr_action = QAction("NMR Data Processing", self)
    nmr_action.triggered.connect(self.open_nmr_window)
    tools_menu.addAction(nmr_action)

def open_nmr_window(self):
    self.nmr_window = NMRProcessingWindow()
    self.nmr_window.show()
```

---

## 📊 数据流示意图

```
用户操作 → UI控件 → 参数对象 → 处理函数 → NMRData对象 → 绘图控件
   ↓                                      ↓
文件选择                               更新显示
   ↓                                      ↓
DataInterface.from_folder()         PlotWidget.draw()
   ↓
NMRData对象
```

---

## ✅ 下一步开发计划

### 阶段1: 核心函数库（本次）
- [x] 数据I/O接口 (`data_io.py`)
- [x] 参数管理 (`parameters.py`)
- [x] 变换模块 (`transforms.py`)
- [ ] 预处理模块 (`preprocessing.py`)
- [ ] 滤波模块 (`filtering.py`)
- [ ] 后处理模块 (`postprocessing.py`)
- [ ] SNR计算 (`snr.py`)
- [ ] 扫描筛选 (`scan_selection.py`)

### 阶段2: 基础UI控件
- [ ] ProcessingPanel (参数调整面板)
- [ ] PlotWidget扩展 (NMR专用绘图)
- [ ] 文件选择对话框

### 阶段3: 整合到Spinach UI
- [ ] 添加NMR Processing Tab
- [ ] 数据接口对接
- [ ] 参数保存/加载整合

### 阶段4: 高级功能
- [ ] 实时数据接入
- [ ] 批处理模式
- [ ] Lorentzian拟合UI
- [ ] SVD/Matrix Pencil分析

---

## 🤔 需要讨论的问题

1. **UI整合方式偏好**
   - [ ] 方案A: 在Spinach UI中添加新Tab（推荐，统一界面）
   - [ ] 方案B: 独立窗口（独立性强，但界面分散）

2. **参数绑定策略**
   - [ ] 实时绑定（每次改变立即更新参数对象）
   - [ ] 手动应用（点击"Apply"按钮后更新）

3. **处理流程**
   - [ ] 自动处理（参数改变自动重新处理）
   - [ ] 手动触发（点击"Run"按钮）

4. **绘图更新**
   - [ ] 实时更新（slider拖动时更新）
   - [ ] 释放后更新（slider释放后更新，性能更好）

5. **高级功能优先级**
   - 你最需要哪些功能？
     - [ ] Lorentzian拟合
     - [ ] SVD滤波
     - [ ] Matrix Pencil
     - [ ] 批处理多个实验
     - [ ] 实时数据采集接入

---

## 💡 优势总结

### 函数库的优势
1. ✅ **独立性**: 可独立测试、调试
2. ✅ **可复用**: 可用于Jupyter Notebook、命令行脚本、UI
3. ✅ **易维护**: 算法更新不影响UI
4. ✅ **易扩展**: 添加新功能无需修改UI

### UI程序的优势
1. ✅ **专注交互**: 只关心用户体验
2. ✅ **灵活布局**: 可随时调整界面
3. ✅ **易集成**: 与Spinach UI无缝整合
4. ✅ **实时反馈**: 参数调整立即可视化

---

## 📝 代码示例：完整工作流

```python
# UI层代码示例
class NMRProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # 数据和参数
        self.data = None
        self.param_manager = ParameterManager()
        
        # 创建UI
        self.setup_ui()
    
    def load_data_from_file(self):
        """从文件加载"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.data = DataInterface.from_nmrduino_folder(folder)
            self.log(f"Loaded {len(self.data.time_data)} points")
    
    def load_data_from_setup(self, time_data, sr, acq_t):
        """从采集装置加载（未来扩展）"""
        self.data = DataInterface.from_live_acquisition(
            time_data, sr, acq_t
        )
        self.log("Loaded from acquisition")
    
    def run_processing(self):
        """执行处理"""
        if self.data is None:
            return
        
        # 获取UI参数
        params = self.get_parameters_from_ui()
        
        # 验证参数
        errors = params.validate()
        if errors:
            QMessageBox.warning(self, "Invalid Parameters", "\n".join(errors))
            return
        
        # 处理数据
        processed_data = process_pipeline(self.data.copy(), params)
        
        # 更新显示
        self.plot_widget.draw(
            processed_data.freq_axis,
            np.abs(processed_data.freq_data),
            xlabel="Frequency (Hz)"
        )
```

---

**准备好了吗？我们可以继续实现剩余的模块！**

请告诉我：
1. 你对这个架构有什么意见？
2. 你希望优先实现哪些功能？
3. UI整合倾向哪个方案？
