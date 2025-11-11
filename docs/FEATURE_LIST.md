# NMR Processing Library - Complete Feature List

## 📦 版本 1.0.0 - 功能完成状态

### ✅ 已完成的核心功能

#### 1. 数据输入输出 (core/data_io.py)
- [x] **多源数据接口**
  - `DataInterface.from_nmrduino_folder()` - 从NMRduino文件夹加载
  - `DataInterface.from_live_acquisition()` - 实时采集数据接入
  - `DataInterface.from_arrays()` - 从内存数组创建
- [x] **NMRData数据类**
  - 时域/频域数据存储
  - 处理历史记录追踪
  - 采集参数管理
- [x] **文件操作**
  - `load_nmrduino_data()` - 优化的.dat文件读取
  - `save_spectrum()` - 导出处理后的谱图
  - `get_available_scans()` - 扫描文件夹获取可用scan

#### 2. 参数管理 (core/parameters.py)
- [x] **ProcessingParameters** - 数据类参数系统
  - Savgol滤波参数 (window, polyorder)
  - 截断参数 (truncation_start, truncation_end)
  - 窗函数参数 (apodization_t2, window_type)
  - Zero filling参数 (zero_fill_factor, fill_value)
  - 相位校正参数 (phase0, phase1)
  - 展宽参数 (broadening_hz, broadening_type)
- [x] **AcquisitionParameters** - 采集参数
  - 采样率、采集时间、脉冲参数
- [x] **ParameterManager** - 参数管理器
  - JSON序列化存储/加载
  - 参数验证
  - 预设模板 (high_resolution, high_sensitivity, fast_preview)

#### 3. 傅里叶变换与频域操作 (core/transforms.py)
- [x] `apply_fft()` - 快速傅里叶变换
- [x] `apply_ifft()` - 逆傅里叶变换
- [x] `apply_phase_correction()` - 零阶+一阶相位校正
- [x] `frequency_axis()` - 频率轴生成
- [x] `bandpass_filter()` - 带通滤波器
- [x] `combine_spectra()` - 多谱图合并（为multi-system准备）

#### 4. 信号滤波 (processing/filtering.py)
- [x] **Savgol滤波**
  - `savgol_filter_nmr()` - 返回baseline用于减法
  - 可配置window长度和多项式阶数
- [x] **窗函数**
  - `apply_window_function()` - 统一窗函数接口
  - Hanning, Hamming, Blackman, Kaiser窗
  - WindowType枚举
- [x] **频域滤波器**
  - `lowpass_filter()` - 低通滤波
  - `highpass_filter()` - 高通滤波
  - `notch_filter()` - 陷波滤波

#### 5. 预处理 (processing/preprocessing.py)
- [x] `truncate_time_domain()` - 时域截断
- [x] `apply_apodization()` - 线形增宽
  - Exponential (指数)
  - Gaussian (高斯)
  - Lorentzian (洛伦兹)
- [x] `zero_filling()` - 零填充
- [x] `remove_dc_offset()` - 直流偏置去除
- [x] `apply_first_point_correction()` - 首点修正

#### 6. 后处理 (processing/postprocessing.py)
- [x] **展宽函数**
  - `gaussian_broadening()` - 高斯展宽
  - `lorentzian_broadening()` - 洛伦兹展宽
  - FWHM参数控制
- [x] **基线校正**
  - `baseline_correction()` - 多种方法
  - 多项式拟合
  - 中值滤波
  - 区域拟合
  - `asymmetric_least_squares_baseline()` - ALS基线校正
- [x] **归一化**
  - `normalize_spectrum()` - 谱图归一化
  - Max, area, internal standard归一化

#### 7. 质量控制 (quality/snr.py)
- [x] `calculate_snr()` - 信噪比计算
  - 详细模式返回peak, noise, peak_position
  - 峰和噪声区域可指定
- [x] `find_peak_in_range()` - 区域内寻峰
- [x] `estimate_noise()` - 噪声估计
  - 支持baseline校正
  - RMS噪声计算
- [x] `dynamic_snr_monitor()` - 实时SNR监控
  - 进度追踪
  - 停止条件判断
- [x] `compare_snr()` - 模拟vs实验SNR对比

#### 8. 扫描筛选 (quality/scan_selection.py)
- [x] **ScanSelector类** - 完整的bad scan筛选系统
  - `calculate_residuals()` - 残差计算
    - Squared, absolute, max差异方法
  - `filter_by_threshold()` - 阈值筛选
  - `auto_threshold_suggestion()` - 自动阈值推荐
    - Percentile, sigma, median方法
  - `get_statistics()` - 统计信息
  - `save_selected_scans()` / `load_selected_scans()` - 结果持久化
- [x] `calculate_scan_residuals()` - 独立残差计算函数
- [x] `filter_scans_by_threshold()` - 独立阈值筛选函数

#### 9. 实时监控 (utils/realtime_monitor.py) ⭐ 新增
- [x] **RealtimeDataMonitor类** - 文件夹实时监控
  - 自动检测新.dat文件
  - 单次扫描模式
  - 累积平均模式
  - 线程安全运行
  - 回调系统 (on_new_scan, on_average_updated, on_scan_count_changed, on_error)
- [x] **MonitorState数据类** - 监控状态管理
- [x] `quick_monitor_start()` - 快速启动监控
- [x] **主要方法**
  - `start()` / `stop()` - 启动/停止监控
  - `set_mode()` - 切换单次/平均模式
  - `reset_average()` - 重置累积平均
  - `get_current_average()` - 获取当前平均结果
  - `get_status()` - 获取监控状态

---

## 📋 功能使用场景

### 场景1: 基本数据处理流程
```python
# 1. 加载数据
data = DataInterface.from_nmrduino_folder("path/to/data", scans=[1,2,3])

# 2. 设置参数
params = ProcessingParameters(savgol_window=51, apodization_t2=0.05)

# 3. 处理流程
filtered = savgol_filter_nmr(data.time_data, params.savgol_window)
truncated = truncate_time_domain(filtered, params.truncation_start, params.truncation_end)
apodized = apply_apodization(truncated, params.apodization_t2)
zero_filled = zero_filling(apodized, params.zero_fill_factor)

# 4. FFT
freq_axis, spectrum = apply_fft(zero_filled, data.sampling_rate)

# 5. 后处理
final = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
```

### 场景2: 实时监控与平均 ⭐
```python
from nmr_processing_lib import quick_monitor_start

def on_new_data(nmr_data, scan_count):
    print(f"Total scans: {scan_count}")
    # 处理数据...
    # 更新UI图表...

# 启动实时监控（累积平均模式）
monitor = quick_monitor_start(
    folder_path="/path/to/experiment",
    on_data_callback=on_new_data,
    average_mode=True,
    poll_interval=1.0
)

# ... 采集运行中 ...

monitor.stop()
```

### 场景3: Bad Scan筛选
```python
from nmr_processing_lib.quality import ScanSelector

# 创建筛选器
selector = ScanSelector("/path/to/experiment")

# 自动推荐阈值
threshold = selector.auto_threshold_suggestion(method='percentile', percentile=75)

# 筛选
good_scans, bad_scans = selector.filter_by_threshold(threshold)

# 查看统计
stats = selector.get_statistics()
print(f"Good: {stats['num_good']}, Bad: {stats['num_bad']}")

# 保存结果
selector.save_selected_scans("selected.json", good_scans)
```

### 场景4: SNR监控与比较
```python
from nmr_processing_lib.quality import calculate_snr, compare_snr

# 计算实验SNR
snr_exp = calculate_snr(
    freq_axis, 
    experimental_spectrum,
    peak_range=(-50, 50),
    noise_range=(200, 400),
    detailed=True
)

# 与模拟比较
comparison = compare_snr(
    freq_axis,
    experimental_spectrum,
    simulated_spectrum,
    peak_range=(-50, 50),
    noise_range=(200, 400)
)

print(f"Experimental SNR: {comparison['experimental_snr']:.1f}")
print(f"Simulated SNR: {comparison['simulated_snr']:.1f}")
```

---

## 🎯 与Spinach UI整合的接口设计

### 数据接口
```python
# Spinach模拟结果 -> NMR处理库
simulated_data = DataInterface.from_arrays(
    time_data=spinach_fid,
    sampling_rate=spinach_params['sampling_rate'],
    acquisition_time=spinach_params['acq_time']
)

# 实验数据加载
experimental_data = DataInterface.from_nmrduino_folder(folder_path)

# 两者可以使用相同的处理流程
```

### 参数系统集成
```python
# ProcessingParameters可以直接绑定到UI spinbox/slider
params = ProcessingParameters()

# UI绑定示例（PySide6）
self.savgol_spinbox.setValue(params.savgol_window)
self.savgol_spinbox.valueChanged.connect(
    lambda val: setattr(params, 'savgol_window', val)
)

# 参数保存/加载
manager = ParameterManager(params)
manager.save_all("user_settings.json")
```

### 实时监控集成
```python
class ExperimentalUI:
    def __init__(self):
        self.monitor = RealtimeDataMonitor(folder_path)
        
        # 连接UI更新回调
        self.monitor.on_average_updated = self.update_plot
        self.monitor.on_scan_count_changed = self.update_counter
    
    def update_plot(self, nmr_data, scan_count):
        # 处理数据
        processed = self.process_pipeline(nmr_data)
        
        # 更新图表widget
        self.plot_widget.update_spectrum(processed.freq_axis, processed.freq_data)
        
        # 如果需要与Spinach模拟对比
        if self.simulation_data is not None:
            self.comparison_plot.update_both(processed, self.simulation_data)
```

---

## 🔄 待实现功能（下一阶段）

### 高级分析
- [ ] Lorentzian拟合
- [ ] SVD分析
- [ ] Matrix Pencil方法
- [ ] 多峰分解

### UI集成
- [ ] PySide6图形界面
- [ ] 与Spinach UI合并
- [ ] 实时参数调节widget
- [ ] 模拟vs实验对比显示
- [ ] 批处理界面

### 测试
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能基准测试

---

## 📖 文档状态

- ✅ `README.md` - 项目说明和快速开始
- ✅ `ARCHITECTURE.md` - 架构设计文档
- ✅ `examples/usage_examples.py` - 基础用法示例
- ✅ `examples/realtime_monitor_examples.py` - 实时监控示例 ⭐ 新增

---

## 🚀 下一步建议

1. **立即可做：测试功能库**
   - 使用您的实际NMR数据测试各个模块
   - 检查是否有bug或需要调整的参数
   - 测试实时监控功能

2. **准备UI整合**
   - 当您准备好Spinach UI代码后，我可以帮您整合
   - 实时监控已经提供了UI集成的回调接口
   - 参数系统设计已考虑UI绑定

3. **性能优化**
   - 如发现处理速度慢的部分，可以针对性优化
   - 实时监控的poll_interval可根据需要调整

---

## 📞 使用反馈

测试过程中如发现任何问题，请记录：
- 问题描述
- 输入数据特征
- 错误信息
- 期望的行为

这样我可以快速修复并完善功能库！

---

**当前版本: 1.0.0 - Core Library Complete** ✅
**实时监控功能已添加** ⭐
