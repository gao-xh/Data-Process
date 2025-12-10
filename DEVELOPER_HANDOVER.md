# 📘 NMR Processing Project - 开发者交接指南
代码中不要出现中文和Emoji！！！
### 1. 项目概况 (Project Overview)
这是一个基于 **Python (PySide6)** 和 **Matplotlib** 的桌面应用程序，用于 **NMR (核磁共振) 信号的高级处理与可视化**。
*   **核心目标**: 提供从原始 FID 数据到频谱图的完整处理流，包括降噪、预测、切趾、FFT 和相位校正。
*   **主要文件**:
    *   `ui_nmr_processing_enhanced.py`: **上帝类 (God Class)**。包含 UI 构建、事件处理、多线程控制和绘图逻辑。
    *   `nmr_processing_lib/`: 后端算法库。数学计算（SVD, LP, FFT 等）应封装于此，保持 UI 层轻量。

### 2. 最近完成的关键特性 (Recent Milestones)
*   **SVD 降噪 (SVD Denoising)**: 集成了基于 Hankel 矩阵的 SVD 算法 (`nmr_processing_lib.processing.filtering`)，并在 UI 中添加了 Rank 控制。
*   **后向线性预测 (Backward LP) 增强**:
    *   实现了 **Training Points** 的精细控制（Slider + SpinBox 同步）。
    *   添加了 **Auto (4x Order)** 模式。
    *   **可视化**: 在时域图上通过绿色区域 (`axvspan`) 实时显示预测训练范围。
    *   **范围扩展**: 训练点数上限提升至 **5120**。
*   **UI 现代化**: 全局应用了 **Pastel Theme** (柔和色调)，优化了按钮和分组框的视觉体验。
*   **对比模式 (Comparison Mode)**: 修复了 Overlay 模式下的 **Unified Scale** 问题，确保多图谱叠加时的比例尺一致性。

### 3. 架构与关键逻辑 (Architecture & Key Logic)

#### A. 数据流 (Data Flow)
1.  **参数收集**: `process_data()` 方法从 UI 控件收集所有参数，打包成 `self.params` 字典。
2.  **多线程处理**: `ProcessingWorker` (QThread) 接收参数和原始数据。**注意：所有耗时的数学运算必须在 Worker 中进行，严禁阻塞主线程。**
3.  **结果回传**: Worker 处理完成后发射 `finished` 信号，携带 `results` 字典。
4.  **可视化**: `on_processing_finished` 接收结果，并调用 `plot_results` 更新画布。

#### B. 处理管线顺序 (Processing Pipeline Order)
这是 `ProcessingWorker.process` 中的核心逻辑，顺序至关重要：
1.  **Savgol Filter**: 基线去除。
2.  **SVD Denoising**: 降噪 (耗时操作)。
3.  **Truncation (Start)**: 切除前端无效数据。
4.  **Backward LP**: 信号重建/预测 (依赖于 Truncation 后的数据)。
5.  **Truncation (End)**: 切除后端数据。
6.  **Apodization**: 切趾 (指数衰减)。
7.  **Zero Filling**: 补零。
8.  **FFT**: 傅里叶变换。
9.  **Phase Correction**: 相位校正。

### 4. ⚠️ 避坑指南与注意事项 (Critical Watchlist)

#### 1. 控件同步死循环 (Signal/Slot Loops)
*   **风险**: 项目中大量使用了 Slider 和 SpinBox 互相绑定的模式（如 LP Order, Training Points）。
*   **对策**: 在代码更新值时，务必使用 `blockSignals(True)` 和 `blockSignals(False)`，否则会触发无限递归或不必要的重绘。

#### 2. 索引与可视化对齐 (Indices Alignment)
*   **风险**: `Backward LP` 的训练区域（绿色高亮）是在 **Truncation 之后** 的数据上计算的。
*   **注意**: 在 `plot_results` 中绘制 `axvspan` 时，x 轴是基于最终处理后的时间轴。如果 Truncation 逻辑发生变化，必须确保 `lp_train_len` 对应的物理时间是正确的。

#### 3. 线程数据传递 (Thread Data Passing)
*   **风险**: 如果你在 `ProcessingWorker` 中添加了新算法，但忘记将结果（如中间变量、计算出的指标）放入 `return` 的字典中，UI 层将无法访问这些数据。
*   **案例**: 之前的 "绿色区域消失" bug 就是因为 `lp_train_len` 没传回 UI。

#### 4. Matplotlib 性能 (Plotting Performance)
*   **风险**: 频繁调用 `canvas.draw()` 会导致界面卡顿。
*   **对策**: 尽量使用 `blit=True` (虽然目前主要用全重绘)。在 Comparison Mode 中，如果叠加图谱过多，渲染会变慢。

#### 5. 库的依赖 (Library Dependency)
*   **注意**: `ui_nmr_processing_enhanced.py` 依赖于 `nmr_processing_lib`。修改 UI 调用的算法接口时，必须同步检查 `nmr_processing_lib` 中的函数签名（参数顺序、默认值）。

### 5. 未来维护建议 (Future Roadmap)
*   **代码拆分**: `ui_nmr_processing_enhanced.py` 接近 4000 行，建议将 `ProcessingWorker` 和部分绘图逻辑拆分到单独的文件中。
*   **异常处理**: 目前 SVD 和 Baseline Correction 有基本的 `try-except`，但建议增加更详细的用户提示（如弹窗报错），而不是仅在控制台打印。
*   **配置持久化**: 目前参数在关闭后会重置，建议添加保存/加载处理参数配置的功能 (JSON/YAML)。
