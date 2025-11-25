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

### Data Comparison
- **Side-by-Side View**: Compare two datasets in separate subplots
- **Overlay (Unified)**: Overlay plots with shared Y-axis to compare relative amplitudes
- **Overlay (Normalized)**: Overlay plots with independent normalization to compare peak shapes
- **Parameter Sync**: Option to apply same processing parameters to both datasets or tune them independently

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

## Processing Workflow

```
File/Live/Array Data
    ↓
DataInterface → NMRData
    ↓
Savgol Filtering (baseline reduction)
    ↓
Truncation (remove edges)
    ↓
Apodization (T2* exponential weighting)
    ↓
Zero Filling
    ↓
FFT → Frequency Domain
    ↓
Phase Correction
    ↓
Gaussian/Lorentzian Broadening
    ↓
Baseline Correction
    ↓
Normalization
    ↓
Final Spectrum
```

## Best Practices

### 1. Parameter Optimization Order

1. **Savgol window**: Try from small to large (21, 51, 101...), observe baseline
2. **Truncation**: Remove noise at edges
3. **Apodization T2**: Adjust based on desired linewidth
4. **Zero filling**: Typically 2-4x
5. **Phase**: Manual adjustment or automatic algorithm
6. **Broadening**: Based on resolution requirements
7. **Baseline**: Final correction

### 2. Real-time Monitoring Tips

- **Poll interval**: Adjust based on acquisition speed, typically 0.5-2 seconds
- **Average mode**: Use cumulative averaging for low SNR experiments
- **Single mode**: Check scan quality or dynamic processes

### 3. Bad Scan Filtering Strategy

```python
# Conservative (keep more scans)
threshold = selector.auto_threshold_suggestion('percentile', percentile=90)

# Aggressive (quality first)
threshold = selector.auto_threshold_suggestion('sigma', sigma_multiplier=2)

# Moderate
threshold = selector.auto_threshold_suggestion('percentile', percentile=75)
```

## Troubleshooting

### Q: Real-time monitor not detecting new files?
**A**: Check:
1. Folder path is correct
2. Filename format is `{scan}.dat`
3. `poll_interval` is not too long
4. File writing is complete

### Q: SNR calculation inaccurate?
**A**: Adjust:
1. `peak_range` to include main peak
2. `noise_range` away from signal region
3. Use `detailed=True` to view peak and noise values

### Q: Parameter changes not obvious?
**A**: Check:
1. Parameter range is reasonable
2. May need to cascade other parameters
3. Data quality is sufficient

## Performance Optimization

- **Large datasets**: Use `zero_fill_factor` instead of manual filling
- **Real-time monitoring**: Adjust `poll_interval` to balance response speed and CPU usage
- **Batch processing**: Consider multiprocessing for multiple experiments

## Future Plans

- [ ] Lorentzian fitting
- [ ] SVD filtering
- [ ] Matrix Pencil analysis
- [ ] Multi-core parallel processing
- [ ] Complete UI program
- [ ] Unit test suite

## License

MIT License - See LICENSE file for details

## Support

For questions or suggestions:
- Check `ARCHITECTURE.md` for design details
- Check `FEATURE_LIST.md` for all features
- See `examples/` directory for complete examples

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Core functionality complete, real-time monitoring added
