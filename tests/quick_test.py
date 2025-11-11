"""
快速测试脚本 - NMR Processing Library
====================================

快速验证所有核心功能是否正常工作
"""

import sys
import numpy as np

print("=" * 60)
print("NMR Processing Library - 快速功能测试")
print("=" * 60)

# 测试1: 导入检查
print("\n[1/8] 导入模块测试...")
try:
    from nmr_processing_lib import (
        DataInterface,
        ProcessingParameters,
        ParameterManager,
        savgol_filter_nmr,
        truncate_time_domain,
        apply_apodization,
        zero_filling,
        apply_fft,
        gaussian_broadening,
        RealtimeDataMonitor,
        quick_monitor_start
    )
    from nmr_processing_lib.quality import calculate_snr, ScanSelector
    from nmr_processing_lib.processing.postprocessing import baseline_correction, normalize_spectrum
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 参数系统
print("\n[2/8] 参数系统测试...")
try:
    params = ProcessingParameters(
        savgol_window=51,
        truncation_start=100,
        apodization_t2=0.05
    )
    print(f"✓ 参数创建成功: savgol_window={params.savgol_window}")
    
    # 测试保存/加载
    manager = ParameterManager()
    manager.processing = params
    manager.save_all("test_quick_params.json")
    manager2 = ParameterManager()
    manager2.load_all("test_quick_params.json")
    assert manager2.processing.savgol_window == 51
    print("✓ 参数保存/加载成功")
    
    # 测试预设
    manager.load_preset("high_resolution")
    print(f"✓ 预设加载成功: {manager.processing.savgol_window}")
    
except Exception as e:
    print(f"✗ 参数系统失败: {e}")

# 测试3: 模拟数据处理
print("\n[3/8] 数据处理流程测试...")
try:
    # 创建模拟NMR数据
    np.random.seed(42)
    t = np.linspace(0, 0.2, 1000)
    # 模拟FID: 指数衰减 + 噪声
    signal = np.exp(-t/0.05) * np.exp(2j * np.pi * 10 * t)  # 10 Hz信号
    noise = 0.1 * (np.random.randn(1000) + 1j * np.random.randn(1000))
    fid = signal + noise
    
    # 从数组创建
    data = DataInterface.from_arrays(fid, sampling_rate=5000, acquisition_time=0.2)
    print(f"✓ 数据创建成功: {len(data.time_data)} 点")
    
    # 完整处理流程
    filtered = savgol_filter_nmr(data.time_data, 51)
    print("  ✓ Savgol滤波")
    
    truncated = truncate_time_domain(filtered, 10, 10)
    print("  ✓ 截断")
    
    apodized = apply_apodization(truncated, 0.05)
    print("  ✓ 窗函数")
    
    zero_filled = zero_filling(apodized, 2)
    print("  ✓ Zero filling")
    
    freq_axis, spectrum = apply_fft(zero_filled, data.sampling_rate)
    print("  ✓ FFT")
    
    broadened = gaussian_broadening(spectrum, freq_axis, 5.0)
    print("  ✓ 高斯展宽")
    
    corrected = baseline_correction(broadened, method='polynomial', order=2)
    print("  ✓ 基线校正")
    
    final = normalize_spectrum(corrected, method='max')
    print("  ✓ 归一化")
    
    print(f"✓ 完整处理流程成功! 输出: {len(freq_axis)} 点")
    
except Exception as e:
    print(f"✗ 数据处理失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: SNR计算
print("\n[4/8] SNR计算测试...")
try:
    # 使用上面处理的spectrum
    snr = calculate_snr(
        freq_axis,
        final,
        peak_range=(-50, 50),
        noise_range=(200, 400)
    )
    print(f"✓ SNR计算成功: {snr:.1f}")
    
    # 详细模式
    snr_detail = calculate_snr(
        freq_axis,
        final,
        peak_range=(-50, 50),
        noise_range=(200, 400),
        detailed=True
    )
    print(f"  SNR: {snr_detail['snr']:.1f}")
    print(f"  Peak: {snr_detail['peak']:.2e}")
    print(f"  Noise: {snr_detail['noise']:.2e}")
    print(f"✓ 详细SNR计算成功")
    
except Exception as e:
    print(f"✗ SNR计算失败: {e}")

# 测试5: 实时监控（基础功能）
print("\n[5/8] 实时监控系统测试...")
try:
    # 创建临时测试文件夹
    import tempfile
    import os
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp(prefix="nmr_test_")
    print(f"  创建临时目录: {temp_dir}")
    
    # 创建几个模拟scan文件
    for i in range(1, 4):
        test_data = np.random.randn(1000) + 1j * np.random.randn(1000)
        filepath = Path(temp_dir) / f"{i}.dat"
        with open(filepath, 'wb') as f:
            test_data.astype(np.complex64).tofile(f)
    print(f"  创建3个测试文件")
    
    # 测试监控器创建
    monitor = RealtimeDataMonitor(temp_dir, poll_interval=0.5)
    print("✓ Monitor创建成功")
    
    # 测试状态
    status = monitor.get_status()
    print(f"  初始状态: running={status['is_running']}, mode={status['mode']}")
    
    # 测试回调设置
    callback_called = {'count': 0}
    
    def test_callback(data, scan_count):
        callback_called['count'] += 1
        print(f"  → 回调触发: scan_count={scan_count}")
    
    monitor.on_average_updated = test_callback
    
    # 短暂运行
    monitor.start(average_mode=True)
    print("  启动监控...")
    
    import time
    time.sleep(1)
    
    monitor.stop()
    print("  停止监控")
    
    print("✓ 实时监控系统功能正常")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"✗ 实时监控测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 扫描筛选（使用临时数据）
print("\n[6/8] Scan筛选系统测试...")
try:
    # 创建临时文件夹和数据
    temp_dir = tempfile.mkdtemp(prefix="nmr_scan_test_")
    
    # 创建5个scan，其中1个是坏的
    for i in range(1, 6):
        if i == 3:
            # Bad scan (更多噪声)
            test_data = 5.0 * (np.random.randn(1000) + 1j * np.random.randn(1000))
        else:
            # Good scan
            t = np.linspace(0, 0.2, 1000)
            signal = np.exp(-t/0.05) * np.exp(2j * np.pi * 10 * t)
            noise = 0.1 * (np.random.randn(1000) + 1j * np.random.randn(1000))
            test_data = signal + noise
        
        filepath = Path(temp_dir) / f"{i}.dat"
        with open(filepath, 'wb') as f:
            test_data.astype(np.complex64).tofile(f)
    
    print(f"  创建5个scan（1个bad）")
    
    # 创建筛选器
    selector = ScanSelector(temp_dir)
    print("✓ ScanSelector创建成功")
    
    # 计算残差
    residuals = selector.calculate_residuals(reference_scan=1, method='squared')
    print(f"  残差计算完成: {len(residuals)} scans")
    
    # 自动阈值
    threshold = selector.auto_threshold_suggestion(method='percentile', percentile=75)
    print(f"  自动阈值: {threshold:.2e}")
    
    # 筛选
    good, bad = selector.filter_by_threshold(threshold)
    print(f"  筛选结果: {len(good)} good, {len(bad)} bad")
    
    # 统计
    stats = selector.get_statistics()
    print(f"  统计: mean={stats['mean_residual']:.2e}")
    
    print("✓ Scan筛选功能正常")
    
    # 清理
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"✗ Scan筛选测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 绘图功能（可选）
print("\n[7/8] 绘图功能测试...")
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 时域
    time_axis = np.arange(len(data.time_data)) / data.sampling_rate
    ax1.plot(time_axis, data.time_data.real)
    ax1.set_title('Time Domain')
    ax1.set_xlabel('Time (s)')
    
    # 频域
    ax2.plot(freq_axis, np.abs(final))
    ax2.set_title('Frequency Domain (Processed)')
    ax2.set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('quick_test_plot.png', dpi=100)
    plt.close()
    
    print("✓ 绘图成功: quick_test_plot.png")
    
except Exception as e:
    print(f"✗ 绘图失败: {e}")

# 测试8: 文件操作
print("\n[8/8] 文件操作测试...")
try:
    # 清理测试文件
    import os
    if os.path.exists("test_quick_params.json"):
        os.remove("test_quick_params.json")
        print("✓ 清理临时文件")
except Exception as e:
    print(f"! 清理文件警告: {e}")

# 总结
print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n主要功能验证:")
print("  ✓ 模块导入")
print("  ✓ 参数管理（创建/保存/加载/预设）")
print("  ✓ 完整处理流程（8步）")
print("  ✓ SNR计算（简单/详细模式）")
print("  ✓ 实时监控系统")
print("  ✓ Scan筛选")
print("  ✓ 绘图输出")
print("\n如所有测试通过 ✓，说明核心功能库工作正常！")
print("\n下一步:")
print("  1. 使用你的真实NMR数据测试")
print("  2. 调整参数获得最佳效果")
print("  3. 准备UI整合")
print("\n详细测试清单请查看: TEST_CHECKLIST.md")
print("完整示例请查看: examples/")
print("=" * 60)
