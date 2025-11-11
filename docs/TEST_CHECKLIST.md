# 功能测试清单

## 📋 测试准备

- [ ] 准备NMR数据文件夹（包含多个scan的.dat文件）
- [ ] 安装所有依赖: `pip install numpy scipy matplotlib PySide6`
- [ ] 确认Python环境正常

---

## ✅ 模块测试清单

### 1. 数据I/O测试 (core/data_io.py)

```python
from nmr_processing_lib import DataInterface

# 测试1: 加载单个scan
folder = r"C:\Your\NMR\Data\Path"
data = DataInterface.from_nmrduino_folder(folder, scans=1)
print(f"✓ 数据点数: {len(data.time_data)}")
print(f"✓ 采样率: {data.sampling_rate} Hz")
```

- [ ] 能成功加载单个scan
- [ ] 能成功加载多个scan并平均
- [ ] 采样率和采集时间正确
- [ ] 数据形状正确（复数数组）

```python
# 测试2: 从数组创建
import numpy as np
test_data = np.random.randn(1000) + 1j*np.random.randn(1000)
data2 = DataInterface.from_arrays(test_data, 5000, 0.2)
print(f"✓ 从数组创建成功")
```

- [ ] 从数组创建成功
- [ ] 数据正确存储

---

### 2. 参数管理测试 (core/parameters.py)

```python
from nmr_processing_lib import ProcessingParameters, ParameterManager

# 测试1: 创建参数
params = ProcessingParameters(
    savgol_window=51,
    truncation_start=100,
    apodization_t2=0.05
)
print(f"✓ 参数创建成功")
print(f"  Savgol window: {params.savgol_window}")

# 测试2: 保存和加载
manager = ParameterManager(params)
manager.save_all("test_params.json")
print(f"✓ 参数保存成功")

manager2 = ParameterManager()
manager2.load_all("test_params.json")
print(f"✓ 参数加载成功")
assert manager2.processing.savgol_window == 51

# 测试3: 预设
manager.load_preset("high_resolution")
print(f"✓ 预设加载成功: {manager.processing.savgol_window}")
```

- [ ] 参数创建正常
- [ ] 能保存到JSON
- [ ] 能从JSON加载
- [ ] 预设功能正常
- [ ] 参数验证工作

---

### 3. 完整处理流程测试

```python
from nmr_processing_lib import (
    DataInterface,
    ProcessingParameters,
    savgol_filter_nmr,
    truncate_time_domain,
    apply_apodization,
    zero_filling,
    apply_fft,
    gaussian_broadening
)
import matplotlib.pyplot as plt

# 加载数据
folder = r"C:\Your\NMR\Data\Path"
data = DataInterface.from_nmrduino_folder(folder, scans=[1,2,3])
params = ProcessingParameters()

# 处理流程
print("开始处理...")
filtered = savgol_filter_nmr(data.time_data, params.savgol_window)
print("✓ Savgol滤波完成")

truncated = truncate_time_domain(filtered, params.truncation_start, params.truncation_end)
print("✓ 截断完成")

apodized = apply_apodization(truncated, params.apodization_t2)
print("✓ 窗函数完成")

zero_filled = zero_filling(apodized, params.zero_fill_factor)
print("✓ Zero filling完成")

freq_axis, spectrum = apply_fft(zero_filled, data.sampling_rate)
print("✓ FFT完成")

final = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
print("✓ 展宽完成")

# 绘图检查
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
time_axis = np.arange(len(data.time_data)) / data.sampling_rate
plt.plot(time_axis, data.time_data.real)
plt.title('Time Domain (Original)')
plt.xlabel('Time (s)')

plt.subplot(1, 2, 2)
plt.plot(freq_axis, np.abs(final))
plt.title('Frequency Domain (Processed)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('test_processing.png')
print("✓ 图像保存到 test_processing.png")
plt.show()
```

- [ ] Savgol滤波正常工作
- [ ] 截断功能正常
- [ ] 窗函数应用正确
- [ ] Zero filling正确
- [ ] FFT结果正确
- [ ] 展宽效果明显
- [ ] 图像正常显示

---

### 4. SNR计算测试 (quality/snr.py)

```python
from nmr_processing_lib.quality import calculate_snr

# 使用上面处理好的spectrum
snr_simple = calculate_snr(
    freq_axis,
    final,
    peak_range=(-50, 50),
    noise_range=(200, 400)
)
print(f"✓ SNR (简单): {snr_simple:.1f}")

# 详细模式
snr_detail = calculate_snr(
    freq_axis,
    final,
    peak_range=(-50, 50),
    noise_range=(200, 400),
    detailed=True
)
print(f"✓ SNR (详细):")
print(f"  SNR: {snr_detail['snr']:.1f}")
print(f"  Peak: {snr_detail['peak']:.2f}")
print(f"  Noise: {snr_detail['noise']:.2f}")
print(f"  Peak位置: {snr_detail['peak_position']:.2f} Hz")
```

- [ ] 简单SNR计算正确
- [ ] 详细模式返回完整信息
- [ ] Peak和noise值合理
- [ ] Peak位置检测正确

---

### 5. Scan筛选测试 (quality/scan_selection.py)

```python
from nmr_processing_lib.quality import ScanSelector

# 创建筛选器
selector = ScanSelector(folder)
print(f"✓ ScanSelector创建成功")

# 计算残差
residuals = selector.calculate_residuals(reference_scan=1, method='squared')
print(f"✓ 残差计算完成: {len(residuals)} scans")
print(f"  残差范围: {min(residuals.values()):.2e} - {max(residuals.values()):.2e}")

# 自动阈值
threshold_p = selector.auto_threshold_suggestion(method='percentile', percentile=75)
print(f"✓ 自动阈值(75%): {threshold_p:.2e}")

threshold_s = selector.auto_threshold_suggestion(method='sigma', sigma_multiplier=2)
print(f"✓ 自动阈值(2σ): {threshold_s:.2e}")

# 筛选
good_scans, bad_scans = selector.filter_by_threshold(threshold_p)
print(f"✓ 筛选完成:")
print(f"  Good scans: {len(good_scans)}")
print(f"  Bad scans: {len(bad_scans)}")

# 统计
stats = selector.get_statistics()
print(f"✓ 统计信息:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# 保存结果
selector.save_selected_scans("test_selected.json", good_scans)
print(f"✓ 结果保存成功")
```

- [ ] 残差计算正常
- [ ] 自动阈值推荐合理
- [ ] 筛选结果正确
- [ ] 统计信息完整
- [ ] 能保存和加载结果

---

### 6. 实时监控测试 ⭐ (utils/realtime_monitor.py)

**测试前准备**: 
1. 准备一个实验文件夹，确保有几个scan文件
2. 准备手动添加新scan文件来模拟采集

```python
from nmr_processing_lib import RealtimeDataMonitor
import time

# 创建监控器
test_folder = r"C:\Your\NMR\Data\Path"
monitor = RealtimeDataMonitor(test_folder, poll_interval=2.0)
print(f"✓ Monitor创建成功")

# 设置回调
def on_single(data, scan_num):
    print(f"  [单次] Scan #{scan_num}: {len(data.time_data)} 点")

def on_average(data, count):
    print(f"  [平均] 累积 {count} 个scan")

def on_count(count):
    print(f"  [计数] 总文件数: {count}")

def on_error(msg):
    print(f"  [错误] {msg}")

monitor.on_new_scan = on_single
monitor.on_average_updated = on_average
monitor.on_scan_count_changed = on_count
monitor.on_error = on_error
print(f"✓ 回调设置完成")

# 测试状态查询
status = monitor.get_status()
print(f"✓ 初始状态: {status}")

# 启动监控（平均模式）
print("\n开始监控（平均模式）...")
monitor.start(average_mode=True)
print("  监控运行中，请手动复制一个新scan文件到文件夹...")
print("  等待10秒...")

# 等待10秒看是否检测到新文件
time.sleep(10)

# 切换到单次模式
print("\n切换到单次模式...")
monitor.set_mode(average_mode=False)
print("  请再复制一个新scan文件...")
time.sleep(10)

# 停止
print("\n停止监控...")
monitor.stop()
final_status = monitor.get_status()
print(f"✓ 最终状态: {final_status}")
```

- [ ] Monitor创建成功
- [ ] 回调系统工作
- [ ] 能检测到新文件
- [ ] 单次模式正常
- [ ] 平均模式正常
- [ ] 模式切换正常
- [ ] 停止功能正常
- [ ] 状态查询正确

**快速测试版本**:

```python
from nmr_processing_lib import quick_monitor_start

def quick_callback(data, count):
    print(f"✓ 检测到数据! Scans: {count}, 数据点: {len(data.time_data)}")

# 快速启动
monitor = quick_monitor_start(
    folder_path=test_folder,
    on_data_callback=quick_callback,
    average_mode=True
)

print("监控启动，等待15秒...")
time.sleep(15)
monitor.stop()
print("✓ 快速监控测试完成")
```

- [ ] quick_monitor_start工作正常

---

### 7. 实时监控与处理整合测试 ⭐

```python
from nmr_processing_lib import (
    RealtimeDataMonitor,
    ProcessingParameters,
    savgol_filter_nmr,
    truncate_time_domain,
    apply_apodization,
    zero_filling,
    apply_fft,
    gaussian_broadening
)
from nmr_processing_lib.quality import calculate_snr
import matplotlib.pyplot as plt

# 准备参数
params = ProcessingParameters(
    savgol_window=51,
    truncation_start=100,
    truncation_end=-100,
    apodization_t2=0.05,
    zero_fill_factor=2,
    broadening_hz=5.0
)

# 准备绘图
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax1.plot([], [], 'b-')
line2, = ax2.plot([], [], 'r-')
ax1.set_title('Time Domain')
ax1.set_xlabel('Time (s)')
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)')

# 完整处理+绘图回调
def process_and_plot(nmr_data, scan_count):
    print(f"\n处理 {scan_count} 个平均scan...")
    
    # 完整处理流程
    filtered = savgol_filter_nmr(nmr_data.time_data, params.savgol_window)
    truncated = truncate_time_domain(filtered, params.truncation_start, params.truncation_end)
    apodized = apply_apodization(truncated, params.apodization_t2)
    zero_filled = zero_filling(apodized, params.zero_fill_factor)
    freq_axis, spectrum = apply_fft(zero_filled, nmr_data.sampling_rate)
    final = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
    
    # 计算SNR
    try:
        snr = calculate_snr(freq_axis, final, peak_range=(-50,50), noise_range=(200,400))
        print(f"  SNR: {snr:.1f} (理论提升: {np.sqrt(scan_count):.1f}x)")
    except:
        pass
    
    # 更新图表
    time_axis = np.arange(len(nmr_data.time_data)) / nmr_data.sampling_rate
    line1.set_data(time_axis, nmr_data.time_data.real)
    ax1.relim()
    ax1.autoscale_view()
    
    line2.set_data(freq_axis, np.abs(final))
    ax2.relim()
    ax2.autoscale_view()
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("  ✓ 图表已更新")

# 启动监控
monitor = RealtimeDataMonitor(test_folder, poll_interval=2.0)
monitor.on_average_updated = process_and_plot
monitor.start(average_mode=True)

print("\n完整流程监控启动!")
print("请手动添加新scan文件测试...")
print("等待30秒...")

time.sleep(30)

monitor.stop()
plt.close()
print("\n✓ 完整流程测试完成!")
```

- [ ] 实时监控工作
- [ ] 处理流程正常执行
- [ ] SNR随scan数增加
- [ ] 图表实时更新
- [ ] 无报错

---

## 🎯 集成测试

### 完整工作流测试

```python
"""
完整的NMR数据处理工作流测试
从数据加载 → 处理 → 质量控制 → 结果保存
"""

import numpy as np
import matplotlib.pyplot as plt
from nmr_processing_lib import *
from nmr_processing_lib.quality import calculate_snr, ScanSelector
from nmr_processing_lib.processing.postprocessing import baseline_correction, normalize_spectrum

# 1. 设置
folder = r"C:\Your\NMR\Data\Path"
print("=== 完整工作流测试 ===\n")

# 2. Bad scan筛选
print("1. Scan筛选...")
selector = ScanSelector(folder)
threshold = selector.auto_threshold_suggestion('percentile', 75)
good_scans, bad_scans = selector.filter_by_threshold(threshold)
print(f"   Good: {len(good_scans)}, Bad: {len(bad_scans)}")

# 3. 加载good scans
print("\n2. 加载数据...")
data = DataInterface.from_nmrduino_folder(folder, scans=good_scans)
print(f"   数据点: {len(data.time_data)}")

# 4. 参数设置
print("\n3. 设置参数...")
params = ProcessingParameters(
    savgol_window=51,
    truncation_start=100,
    truncation_end=-100,
    apodization_t2=0.05,
    zero_fill_factor=2,
    broadening_hz=5.0
)

# 5. 完整处理
print("\n4. 数据处理...")
filtered = savgol_filter_nmr(data.time_data, params.savgol_window)
truncated = truncate_time_domain(filtered, params.truncation_start, params.truncation_end)
apodized = apply_apodization(truncated, params.apodization_t2)
zero_filled = zero_filling(apodized, params.zero_fill_factor)
freq_axis, spectrum = apply_fft(zero_filled, data.sampling_rate)
broadened = gaussian_broadening(spectrum, freq_axis, params.broadening_hz)
corrected = baseline_correction(broadened, method='polynomial', order=2)
final = normalize_spectrum(corrected, method='max')
print("   处理完成!")

# 6. 质量评估
print("\n5. 质量评估...")
snr_result = calculate_snr(
    freq_axis, final,
    peak_range=(-50, 50),
    noise_range=(200, 400),
    detailed=True
)
print(f"   SNR: {snr_result['snr']:.1f}")
print(f"   Peak: {snr_result['peak']:.2e}")
print(f"   Noise: {snr_result['noise']:.2e}")

# 7. 绘图
print("\n6. 绘图...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 时域原始
time_axis = np.arange(len(data.time_data)) / data.sampling_rate
axes[0,0].plot(time_axis, data.time_data.real)
axes[0,0].set_title('Time Domain (Original)')
axes[0,0].set_xlabel('Time (s)')

# 时域处理后
time_axis_proc = np.arange(len(zero_filled)) / data.sampling_rate
axes[0,1].plot(time_axis_proc, zero_filled.real)
axes[0,1].set_title('Time Domain (Processed)')
axes[0,1].set_xlabel('Time (s)')

# 频域（处理后）
axes[1,0].plot(freq_axis, np.abs(broadened))
axes[1,0].set_title('Frequency Domain (Broadened)')
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].axvspan(-50, 50, alpha=0.2, color='green', label='Peak Region')
axes[1,0].legend()

# 频域（最终）
axes[1,1].plot(freq_axis, np.abs(final))
axes[1,1].set_title(f'Final Spectrum (SNR={snr_result["snr"]:.1f})')
axes[1,1].set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('test_complete_workflow.png', dpi=150)
print("   图像保存到: test_complete_workflow.png")
plt.show()

# 8. 保存结果
print("\n7. 保存结果...")
ParameterManager(params).save_all('test_workflow_params.json')
selector.save_selected_scans('test_workflow_scans.json', good_scans)
print("   参数和scan列表已保存")

print("\n✓ 完整工作流测试成功!")
```

- [ ] Scan筛选正常
- [ ] 数据加载正确
- [ ] 完整处理流程无错
- [ ] SNR计算合理
- [ ] 图像正常显示
- [ ] 结果保存成功

---

## ✅ 测试总结

### 通过标准

- [ ] 所有核心功能无报错
- [ ] 处理结果合理（谱图、SNR等）
- [ ] 实时监控能检测新文件
- [ ] 参数保存/加载正常
- [ ] 图像输出正常

### 发现的问题记录

1. 问题描述：
   - 输入数据：
   - 错误信息：
   - 期望行为：

2. 问题描述：
   ...

---

## 📝 测试完成后

测试完成请反馈：
1. 哪些功能工作正常 ✅
2. 哪些功能有问题 ❌
3. 需要调整的参数
4. 性能表现如何
5. 还需要哪些功能

这样我可以快速修复bug并完善功能库！
