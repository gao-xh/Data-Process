# NMR Data Processing Suite

A professional NMR data processing application with modern UI and complete processing pipeline.

## Quick Start

### Requirements
```bash
pip install PySide6 numpy scipy matplotlib
```

### Running the Application
**Double-click**: `run_enhanced_ui.bat`  
**Or via command line**:
```bash
python ui_nmr_processing_enhanced.py
```

## Main Features

### Processing Pipeline
- **Savgol Filtering**: Baseline correction with adjustable window and polynomial order
- **Time Domain Truncation**: Remove unwanted signal regions
- **Apodization**: T2* exponential weighting for sensitivity/resolution balance
- **Hanning Window**: Reduce spectral artifacts
- **Zero Filling**: Improve digital resolution
- **FFT**: Transform to frequency domain with automatic phase correction

### UI Features
- **Resizable Panels**: Drag splitters to adjust control/plot area sizes
- **Maximizable Plots**: Double-click or use menu to open plots in separate windows
- **Real-time SNR Display**: Automatic signal-to-noise ratio calculation
- **Slider Controls**: Intuitive parameter adjustment with live preview
- **Settings Persistence**: Window state and parameters saved automatically
- **Keyboard Shortcuts**: Fast workflow (see below)

### Keyboard Shortcuts
- `Ctrl+O`: Load data file
- `Ctrl+S`: Save processing parameters
- `F5`: Process data
- `Ctrl+E`: Export results
- `Ctrl+1/2/3`: Maximize time/low freq/high freq plot
- `Ctrl+Q`: Quit application

## Project Structure

```
DUI_10_8/
├── ui_nmr_processing_enhanced.py    # Main application (START HERE)
├── nmrduino_util.py                 # NMRduino data utilities
├── run_enhanced_ui.bat              # Quick launcher
├── README.md                        # This file
│
├── ui_versions/                     # Previous UI versions
├── scripts/                         # Processing scripts
├── notebooks/                       # Jupyter notebooks
├── tests/                           # Test files
├── docs/                            # Documentation
├── examples/                        # Example data
└── nmr_processing_lib/              # Core processing library
```

## Parameter Files

Processing parameters are saved as JSON:
```json
{
  "savgol_window": 51,
  "savgol_order": 3,
  "truncation_start": 100,
  "truncation_end": -100,
  "apodization_t2": 0.05,
  "hanning_enabled": true,
  "zero_fill_factor": 2,
  "freq_display_low": [-5, 5],
  "freq_display_high": [35, 55]
}
```

## Data Format

The application supports:
- **NMRduino compiled files**: `.npy` format with associated `.dat` source files
- **Raw data**: NumPy arrays with time/frequency axes
- **Parameter files**: JSON format for reproducibility

## Tips

1. **Start with Example Data**: Load sample data from `examples/` folder
2. **Adjust Display Range**: Use sliders to focus on region of interest
3. **Save Parameters**: Use `Ctrl+S` to save working parameter sets
4. **Zoom/Pan**: All plots support matplotlib zoom (box select) and pan (arrow drag)
5. **Full Spectrum Access**: Display range only sets initial view - zoom out to see full spectrum

## Documentation

See `docs/` folder for detailed documentation:
- `ARCHITECTURE.md`: System design overview
- `FEATURE_LIST.md`: Complete feature list
- `TEST_CHECKLIST.md`: Testing guidelines

## Legacy Files

- `ui_versions/`: Previous UI implementations
- `scripts/`: Standalone processing scripts
- `notebooks/`: Original Jupyter notebook workflows

## License

This project is for research and educational purposes.

```
文件/实时/数组
    ↓
DataInterface → NMRData
    ↓
Savgol滤波 (baseline reduction)
    ↓
截断 (truncate time domain)
    ↓
窗函数 (apodization/line broadening)
    ↓
Zero Filling
    ↓
FFT → 频域
    ↓
相位校正
    ↓
高斯/洛伦兹展宽
    ↓
基线校正
    ↓
归一化
    ↓
最终谱图
```

## 🎓 最佳实践

### 1. 参数调优顺序

1. **Savgol window**: 从小到大尝试（21, 51, 101...），观察baseline
2. **截断**: 去除首尾噪声点
3. **Apodization T2**: 根据期望线宽调整
4. **Zero filling**: 通常2-4倍
5. **相位**: 手动调整或自动算法
6. **展宽**: 根据分辨率需求
7. **基线**: 最后校正

### 2. 实时监控建议

- **Poll interval**: 根据采集速度调整，通常0.5-2秒
- **Average mode**: 低信噪比实验使用累积平均
- **Single mode**: 检查scan质量或动态过程

### 3. Bad Scan筛选策略

```python
# 保守策略（保留更多scan）
threshold = selector.auto_threshold_suggestion('percentile', percentile=90)

# 激进策略（质量优先）
threshold = selector.auto_threshold_suggestion('sigma', sigma_multiplier=2)

# 中等策略
threshold = selector.auto_threshold_suggestion('percentile', percentile=75)
```

## 🐛 故障排除

### Q: 实时监控检测不到新文件？
**A**: 检查：
1. 文件夹路径是否正确
2. 文件名格式是否为 `{scan}.dat`
3. `poll_interval` 是否太长
4. 文件写入是否完成

### Q: SNR计算不准确？
**A**: 调整：
1. `peak_range` 确保包含主峰
2. `noise_range` 远离信号区域
3. 使用 `detailed=True` 查看peak和noise值

### Q: 参数改变后效果不明显？
**A**: 检查：
1. 参数范围是否合理
2. 是否需要级联其他参数
3. 数据质量是否足够

## 📈 性能优化

- **大数据集**: 使用`zero_fill_factor`而非手动填充
- **实时监控**: 调整`poll_interval`平衡响应速度和CPU使用
- **批处理**: 考虑使用多进程处理多个实验

## 🔮 未来计划

- [ ] Lorentzian拟合
- [ ] SVD滤波
- [ ] Matrix Pencil分析
- [ ] 多核并行处理
- [ ] 完整UI程序
- [ ] 单元测试套件

## 📄 许可证

MIT License

## 📞 支持

如有问题或建议：
- 查看 `ARCHITECTURE.md` 了解设计思路
- 查看 `FEATURE_LIST.md` 了解所有功能
- 查看 `examples/` 目录获取完整示例

---

**版本**: 1.0.0  
**更新**: 2025-01-08  
**状态**: ✅ 核心功能库完成，实时监控已添加
