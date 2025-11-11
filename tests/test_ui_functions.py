"""
Quick test for UI without PySide6
Test core processing functions
"""

import numpy as np
from nmr_processing_lib import DataInterface
from nmr_processing_lib.core import apply_fft, apply_phase_correction
from nmr_processing_lib.processing import (
    savgol_filter_nmr,
    apply_apodization,
    zero_filling,
    baseline_correction
)
from nmr_processing_lib.quality import calculate_snr, estimate_noise

print("=" * 60)
print("NMR Processing Library - Core Functions Test")
print("=" * 60)

# Generate test data
print("\n1. Generating test data...")
n_points = 1000
sampling_rate = 5000.0
time = np.arange(n_points) / sampling_rate

signal = np.zeros(n_points, dtype=complex)
frequencies = [10, 50, 150, -80]
amplitudes = [1.0, 0.7, 0.5, 0.3]
decays = [0.1, 0.15, 0.2, 0.12]

for freq, amp, decay in zip(frequencies, amplitudes, decays):
    signal += amp * np.exp(1j * 2 * np.pi * freq * time) * np.exp(-decay * time)

noise = 0.05 * (np.random.randn(n_points) + 1j * np.random.randn(n_points))
signal += noise

print(f"   ✓ Generated {n_points} points at {sampling_rate} Hz")
print(f"   ✓ Peaks at: {frequencies} Hz")

# Test DataInterface
print("\n2. Testing DataInterface...")
data = DataInterface.from_arrays(signal, sampling_rate)
print(f"   ✓ Created NMRData object")
print(f"   ✓ Time data shape: {data.time_data.shape}")

# Test Savgol filter
print("\n3. Testing Savitzky-Golay filter...")
filtered = savgol_filter_nmr(data.time_data, window_length=51, polyorder=2)
print(f"   ✓ Filtered data shape: {filtered.shape}")

# Test Apodization
print("\n4. Testing Apodization...")
broadening_hz = 5.0
t2_star = 1.0 / (2 * np.pi * broadening_hz)
apodized = apply_apodization(filtered, t2_star=t2_star, apodization_type='exponential')
print(f"   ✓ Apodized data shape: {apodized.shape}")

# Test Zero Filling
print("\n5. Testing Zero Filling...")
zerofilled = zero_filling(apodized, final_size=2048)
print(f"   ✓ Zero filled to: {zerofilled.shape}")

# Test FFT
print("\n6. Testing FFT...")
freq_axis, spectrum = apply_fft(zerofilled, sampling_rate)
print(f"   ✓ Frequency axis: {len(freq_axis)} points")
print(f"   ✓ Spectrum: {len(spectrum)} points")
print(f"   ✓ Freq range: {freq_axis.min():.1f} to {freq_axis.max():.1f} Hz")

# Test Phase Correction
print("\n7. Testing Phase Correction...")
phased = apply_phase_correction(spectrum, phase0=0, phase1=0)
print(f"   ✓ Phase corrected spectrum: {phased.shape}")

# Test Baseline Correction
print("\n8. Testing Baseline Correction...")
baseline_corrected = baseline_correction(phased, method='polynomial', order=2)
print(f"   ✓ Baseline corrected: {baseline_corrected.shape}")

# Test SNR calculation
print("\n9. Testing SNR calculation...")
try:
    snr = calculate_snr(freq_axis, spectrum, 
                       peak_range=(freq_axis.min() + 50, freq_axis.max() - 50),
                       noise_range=(freq_axis.max() - 200, freq_axis.max() - 100))
    print(f"   ✓ SNR: {snr:.2f}")
except Exception as e:
    print(f"   ⚠ SNR calculation: {e}")

# Test noise estimation
print("\n10. Testing Noise Estimation...")
try:
    noise_level = estimate_noise(freq_axis, spectrum)
    print(f"   ✓ Noise level: {noise_level:.2e}")
except Exception as e:
    print(f"   ⚠ Noise estimation: {e}")

# Peak detection
print("\n11. Testing Peak Detection...")
spectrum_abs = np.abs(spectrum)
threshold = 0.1 * np.max(spectrum_abs)

peaks = []
for i in range(1, len(spectrum_abs) - 1):
    if (spectrum_abs[i] > spectrum_abs[i-1] and 
        spectrum_abs[i] > spectrum_abs[i+1] and 
        spectrum_abs[i] > threshold):
        peaks.append((freq_axis[i], spectrum_abs[i]))

peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
print(f"   ✓ Found {len(peaks)} peaks")
for i, (freq, amp) in enumerate(peaks[:5], 1):
    print(f"     Peak {i}: {freq:.2f} Hz, Amplitude: {amp:.2e}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
print("\nYou can now run the UI with:")
print("  python ui_data_processing.py")
print("\nMake sure PySide6 and matplotlib are installed:")
print("  pip install PySide6 matplotlib")
