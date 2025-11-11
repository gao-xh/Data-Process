"""
示例: 如何使用 nmr_processing_lib 函数库
==========================================

这个文件展示了函数库的基本用法，包括：
1. 从不同数据源加载数据
2. 参数管理
3. 数据处理流程
4. 与UI整合的接口

作者: NMR Processing Team
日期: 2025-10-08
"""

import numpy as np
import matplotlib.pyplot as plt
from nmr_processing_lib.core.data_io import DataInterface, NMRData
from nmr_processing_lib.core.parameters import (
    ProcessingParameters,
    ParameterManager
)


# ============================================================================
# 示例1: 从文件加载数据
# ============================================================================
def example_load_from_file():
    """从NMRduino文件夹加载数据"""
    print("=" * 60)
    print("示例1: 从文件加载数据")
    print("=" * 60)
    
    # 方式1: 使用DataInterface（推荐）
    folder_path = "path/to/your/experiment"  # 替换为实际路径
    
    try:
        # 加载所有扫描
        data = DataInterface.from_nmrduino_folder(folder_path, scans=0)
        
        print(f"✓ 成功加载数据:")
        print(f"  - 采样点数: {len(data.time_data)}")
        print(f"  - 采样率: {data.sampling_rate} Hz")
        print(f"  - 采集时间: {data.acquisition_time:.3f} s")
        print(f"  - 扫描数: {data.num_scans}")
        print(f"  - 数据来源: {data.source.value}")
        
        return data
    
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


# ============================================================================
# 示例2: 从实时采集加载数据（未来扩展接口）
# ============================================================================
def example_load_from_live():
    """模拟从实时采集加载数据"""
    print("\n" + "=" * 60)
    print("示例2: 从实时采集加载数据（模拟）")
    print("=" * 60)
    
    # 模拟采集数据
    sampling_rate = 8333.0
    acquisition_time = 8.0
    num_points = int(sampling_rate * acquisition_time)
    
    # 生成模拟FID信号（衰减正弦波）
    t = np.linspace(0, acquisition_time, num_points)
    time_data = np.exp(-t / 2.0) * np.sin(2 * np.pi * 150 * t) + \
                0.1 * np.random.randn(num_points)
    
    # 使用DataInterface创建NMRData对象
    data = DataInterface.from_live_acquisition(
        time_data=time_data,
        sampling_rate=sampling_rate,
        acquisition_time=acquisition_time,
        scan_number=1
    )
    
    print(f"✓ 创建模拟数据:")
    print(f"  - 采样点数: {len(data.time_data)}")
    print(f"  - 采样率: {data.sampling_rate} Hz")
    print(f"  - 采集时间: {data.acquisition_time:.3f} s")
    print(f"  - 数据来源: {data.source.value}")
    
    return data


# ============================================================================
# 示例3: 参数管理
# ============================================================================
def example_parameter_management():
    """参数管理示例"""
    print("\n" + "=" * 60)
    print("示例3: 参数管理")
    print("=" * 60)
    
    # 创建参数管理器
    manager = ParameterManager()
    
    # 方式1: 直接设置参数
    manager.processing.savgol_window = 301
    manager.processing.savgol_order = 2
    manager.processing.trunc_start = 100
    manager.processing.trunc_end = 100
    manager.processing.apodization_t2 = 0.75
    manager.processing.zero_fill_factor = 2.7
    
    print("✓ 设置处理参数:")
    print(f"  - Savgol窗口: {manager.processing.savgol_window}")
    print(f"  - Savgol阶数: {manager.processing.savgol_order}")
    print(f"  - 时域截断: {manager.processing.trunc_start}, {manager.processing.trunc_end}")
    print(f"  - Apodization T2*: {manager.processing.apodization_t2}")
    print(f"  - 零填充因子: {manager.processing.zero_fill_factor}")
    
    # 方式2: 加载预设
    print("\n加载预设...")
    presets = manager.get_preset_names()
    print(f"  可用预设: {presets}")
    
    high_res = manager.load_preset("high_resolution")
    print(f"  'high_resolution' 预设:")
    print(f"    - Savgol窗口: {high_res.savgol_window}")
    print(f"    - 零填充因子: {high_res.zero_fill_factor}")
    
    # 方式3: 保存/加载参数
    try:
        manager.save_all("test_parameters.json")
        print("\n✓ 参数已保存到 test_parameters.json")
        
        # 创建新管理器并加载
        new_manager = ParameterManager()
        new_manager.load_all("test_parameters.json")
        print("✓ 参数已从文件加载")
        
    except Exception as e:
        print(f"✗ 保存/加载失败: {e}")
    
    # 参数验证
    errors = manager.validate_current()
    if errors:
        print(f"\n✗ 参数验证失败:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✓ 参数验证通过")
    
    return manager


# ============================================================================
# 示例4: 完整的数据处理流程
# ============================================================================
def example_processing_pipeline():
    """完整的数据处理流程"""
    print("\n" + "=" * 60)
    print("示例4: 数据处理流程")
    print("=" * 60)
    
    # 1. 创建模拟数据
    data = example_load_from_live()
    
    # 2. 设置参数
    params = ProcessingParameters(
        savgol_window=301,
        savgol_order=2,
        savgol_enabled=True,
        trunc_start=100,
        trunc_end=100,
        apodization_t2=0.5,
        zero_fill_factor=2.0
    )
    
    print("\n处理步骤:")
    
    # TODO: 这些处理函数将在下一步实现
    # 这里展示了预期的调用方式
    
    # 3. Savgol滤波
    # from nmr_processing_lib.processing.filtering import savgol_filter_nmr
    # if params.savgol_enabled:
    #     data.time_data = savgol_filter_nmr(
    #         data.time_data,
    #         params.savgol_window,
    #         params.savgol_order
    #     )
    #     print("  ✓ Savgol滤波完成")
    
    # 4. 时域截断
    # from nmr_processing_lib.processing.preprocessing import truncate_time_domain
    # data.time_data = truncate_time_domain(
    #     data.time_data,
    #     params.trunc_start,
    #     params.trunc_end
    # )
    # print("  ✓ 时域截断完成")
    
    # 5. Apodization
    # from nmr_processing_lib.processing.preprocessing import apply_apodization
    # data.time_data = apply_apodization(
    #     data.time_data,
    #     data.acquisition_time,
    #     params.apodization_t2
    # )
    # print("  ✓ Apodization完成")
    
    # 6. 零填充
    # from nmr_processing_lib.processing.preprocessing import zero_filling
    # data.time_data = zero_filling(
    #     data.time_data,
    #     params.zero_fill_factor
    # )
    # print("  ✓ 零填充完成")
    
    # 7. FFT
    from nmr_processing_lib.core.transforms import apply_fft
    freq_axis, freq_data = apply_fft(data)
    print("  ✓ FFT完成")
    
    # 8. 绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(data.time_axis, data.time_data, 'k', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Time Domain')
    
    plt.subplot(1, 2, 2)
    mask = (freq_axis >= 0) & (freq_axis <= 300)
    plt.plot(freq_axis[mask], np.abs(freq_data[mask]), 'k', linewidth=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain')
    
    plt.tight_layout()
    plt.savefig('nmr_processing_example.png', dpi=150)
    print("\n✓ 结果已保存到 nmr_processing_example.png")
    
    return data


# ============================================================================
# 示例5: UI整合接口演示
# ============================================================================
def example_ui_integration():
    """展示如何在UI中使用这些接口"""
    print("\n" + "=" * 60)
    print("示例5: UI整合接口")
    print("=" * 60)
    
    print("""
    在PySide6 UI中的使用方式:
    
    # 1. 数据加载（文件选择后）
    folder = QFileDialog.getExistingDirectory(self, "Select Folder")
    self.data = DataInterface.from_nmrduino_folder(folder)
    
    # 2. 参数绑定到UI控件
    self.param_manager = ParameterManager()
    
    # SpinBox双向绑定
    self.savgol_spinbox.valueChanged.connect(
        lambda v: setattr(self.param_manager.processing, 'savgol_window', v)
    )
    self.savgol_spinbox.setValue(
        self.param_manager.processing.savgol_window
    )
    
    # 3. 处理按钮回调
    def on_run_processing(self):
        # 获取参数
        params = self.param_manager.processing
        
        # 验证
        errors = params.validate()
        if errors:
            QMessageBox.warning(self, "Invalid", "\\n".join(errors))
            return
        
        # 处理（在worker线程中）
        self.worker = ProcessingWorker(self.data, params)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
    
    # 4. 更新绘图
    def on_processing_finished(self, result_data):
        self.plot_widget.draw(
            result_data.freq_axis,
            np.abs(result_data.freq_data),
            xlabel="Frequency (Hz)",
            title="Processed Spectrum"
        )
    
    # 5. 保存/加载参数
    def save_parameters(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON (*.json)"
        )
        if filepath:
            self.param_manager.save_all(filepath)
    
    # 6. 导出谱图
    def export_spectrum(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Spectrum", "", "NPY (*.npy)"
        )
        if filepath:
            save_spectrum(self.data, filepath)
    """)


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🔬 " * 30)
    print("NMR Processing Library - 使用示例")
    print("🔬 " * 30)
    
    # 运行所有示例
    # example_load_from_file()  # 需要实际数据文件
    example_load_from_live()
    example_parameter_management()
    example_processing_pipeline()
    example_ui_integration()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完毕！")
    print("=" * 60)
    
    # 显示图形（如果有）
    try:
        plt.show()
    except:
        pass
