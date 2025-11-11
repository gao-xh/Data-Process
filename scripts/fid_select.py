import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.fft import fft
from nmrduino_util import nmrduino_dat_interp_FAST
import os
import tkinter as tk
from tkinter import filedialog

# Check if running in Jupyter notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter

# Only use ipywidgets if actually in Jupyter
IPYWIDGETS_AVAILABLE = False
if is_notebook():
    try:
        from ipywidgets import interact, FloatSlider, IntSlider
        IPYWIDGETS_AVAILABLE = True
        print("Running in Jupyter - using ipywidgets")
    except ImportError:
        print("ipywidgets not available")
else:
    print("Running in standard Python - using matplotlib slider")


def select_data_folder():
    """
    Select data folder using file dialog
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    folder_path = filedialog.askdirectory(title="Select NMR Data Folder")
    root.destroy()
    
    if not folder_path:
        print("No folder selected, program exit")
        return None
    
    print(f"Selected data folder: {folder_path}")
    return folder_path


def get_available_scans(folder_path):
    """
    Get list of available scan numbers from folder
    """
    dat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.dat')])
    scan_numbers = []
    
    for fname in dat_files:
        try:
            # Extract scan number from filename
            scan_num = int(fname.replace('.dat', '').replace('scan', ''))
            scan_numbers.append(scan_num)
        except:
            continue
    
    return sorted(scan_numbers)


def load_scan_data(folder_path, scan_number):
    """
    Load single scan data using nmrduino_dat_interp_FAST
    
    Args:
        folder_path: Data folder path
        scan_number: Scan number to load
    
    Returns:
        time_data: Time domain data
        sampling_rate: Sampling rate
        acq_time: Acquisition time
    """
    try:
        time_data, sampling_rate, acq_time = nmrduino_dat_interp_FAST(
            folder_path, scan_number, nowarn=True
        )
        return time_data, sampling_rate, acq_time
    except Exception as e:
        print(f"Error loading scan {scan_number}: {e}")
        return None, None, None


def calculate_difference_sums(folder_path, reference_scan, scan_list):
    """
    Calculate sum of squared residuals between each scan and reference scan
    Using the method from bad_scan_filtration_part2.ipynb:
    residual = scan - reference
    metric = Σ(residual²)
    
    Args:
        folder_path: Data folder path
        reference_scan: Reference scan number (goodscan)
        scan_list: List of all scan numbers to compare
    
    Returns:
        residual_sums: Array of sum of squared residuals for each scan
        valid_scans: List of successfully processed scan numbers
    """
    # Load reference scan
    ref_data, ref_sr, ref_acq = load_scan_data(folder_path, reference_scan)
    
    if ref_data is None:
        raise ValueError(f"Unable to load reference scan {reference_scan}")
    
    print(f"Reference scan {reference_scan} loaded: {len(ref_data)} points")
    
    residual_sums = []
    valid_scans = []
    
    for scan_num in scan_list:
        # Load current scan
        scan_data, scan_sr, scan_acq = load_scan_data(folder_path, scan_num)
        
        if scan_data is None:
            print(f"Skip scan {scan_num}: unable to load")
            continue
        
        # Ensure same length
        min_len = min(len(ref_data), len(scan_data))
        
        # Calculate residual (difference)
        residual = scan_data[:min_len] - ref_data[:min_len]
        
        # Calculate sum of squared residuals: Σ(residual²)
        residual_sum = np.sum(residual**2)
        
        residual_sums.append(residual_sum)
        valid_scans.append(scan_num)
        
        # Progress display
        if len(valid_scans) % 50 == 0:
            print(f"Processed {len(valid_scans)} scans...")
    
    return np.array(residual_sums), valid_scans


def plot_difference_analysis(diff_sums, scan_numbers, threshold=None, selected_scans=None):
    """
    Plot sum of squared residuals vs scan number with optional threshold line
    
    Args:
        diff_sums: Array of sum of squared residuals
        scan_numbers: Array of scan numbers
        threshold: Threshold value for filtering (optional)
        selected_scans: List of selected scan indices (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot all scans - scatter plot only (no lines)
    plt.scatter(scan_numbers, diff_sums, c='gray', s=20,
                alpha=0.5, label='All scans')
    
    # Highlight selected scans if provided
    if selected_scans is not None:
        selected_diff = [diff_sums[i] for i in selected_scans]
        selected_nums = [scan_numbers[i] for i in selected_scans]
        plt.scatter(selected_nums, selected_diff, c='green', s=40,
                   label=f'Selected scans ({len(selected_scans)})', zorder=5)
    
    # Draw threshold line
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold = {threshold:.2e}')
    
    plt.xlabel('Scan Number', fontsize=12)
    plt.ylabel('Sum of Squared Residuals: Σ(scan - reference)²', fontsize=12)
    plt.title('Scan Quality Analysis - Squared Residuals from Reference Scan', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def filter_scans_by_threshold(diff_sums, scan_numbers, threshold):
    """
    Filter scans based on difference threshold
    
    Args:
        diff_sums: Array of difference sums
        scan_numbers: Array of scan numbers
        threshold: Threshold value
    
    Returns:
        selected_indices: Indices of selected scans
        selected_scan_nums: Scan numbers of selected scans
    """
    selected_indices = np.where(diff_sums <= threshold)[0]
    selected_scan_nums = [scan_numbers[i] for i in selected_indices]
    
    return selected_indices, selected_scan_nums


def interactive_threshold_selection(diff_sums, scan_numbers):
    """
    Interactive threshold selection using ipywidgets (for Jupyter)
    
    Args:
        diff_sums: Array of difference sums
        scan_numbers: Array of scan numbers
    """
    if not IPYWIDGETS_AVAILABLE:
        print("ipywidgets not available. Use manual threshold selection instead.")
        return
    
    min_diff = np.min(diff_sums)
    max_diff = np.max(diff_sums)
    median_diff = np.median(diff_sums)
    
    # Create a single figure for interactive updates
    fig = plt.figure(figsize=(12, 6))
    
    def update_plot(threshold):
        plt.clf()  # Clear current figure
        selected_idx, selected_nums = filter_scans_by_threshold(
            diff_sums, scan_numbers, threshold
        )
        
        # Plot all scans - scatter plot only (no lines)
        plt.scatter(scan_numbers, diff_sums, c='gray', s=20,
                   alpha=0.5, label='All scans')
        
        # Highlight selected scans
        if len(selected_idx) > 0:
            selected_diff = [diff_sums[i] for i in selected_idx]
            selected_nums_plot = [scan_numbers[i] for i in selected_idx]
            plt.scatter(selected_nums_plot, selected_diff, c='green', s=40,
                       label=f'Selected scans ({len(selected_idx)})', zorder=5)
        
        # Draw threshold line
        plt.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold = {threshold:.2e}')
        
        plt.xlabel('Scan Number', fontsize=12)
        plt.ylabel('Sum of Squared Residuals: Σ(scan - reference)²', fontsize=12)
        plt.title('Scan Quality Analysis - Squared Residuals from Reference Scan', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"Threshold: {threshold:.2e}, Selected: {len(selected_nums)}/{len(scan_numbers)} ({len(selected_nums)/len(scan_numbers)*100:.1f}%)")
    
    # Create slider
    threshold_slider = FloatSlider(
        value=median_diff,
        min=min_diff,
        max=max_diff,
        step=(max_diff - min_diff) / 100,
        description='Threshold:',
        readout_format='.2e'
    )
    
    interact(update_plot, threshold=threshold_slider)


def interactive_slider_selection(diff_sums, scan_numbers, folder_path):
    """
    Interactive threshold selection using matplotlib slider
    
    Args:
        diff_sums: Array of sum of squared residuals
        scan_numbers: Array of scan numbers
        folder_path: Path to save results
    """
    min_diff = np.min(diff_sums)
    max_diff = np.max(diff_sums)
    median_diff = np.median(diff_sums)
    
    # Create figure with extra space for slider
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25)
    
    # Store current selection for saving
    current_selection = {'selected_nums': [], 'threshold': median_diff}
    
    # Plot initial data - using scatter plot (no lines)
    scatter_all = ax.scatter(scan_numbers, diff_sums, c='gray', s=20,
                            alpha=0.5, label='All scans')
    scatter_selected = ax.scatter([], [], c='green', s=40,
                                 label='Selected scans', zorder=5)
    threshold_line = ax.axhline(y=median_diff, color='red', linestyle='--', 
                                linewidth=2, label=f'Threshold = {median_diff:.2e}')
    
    ax.set_xlabel('Scan Number', fontsize=12)
    ax.set_ylabel('Sum of Squared Residuals: Σ(scan - reference)²', fontsize=12)
    ax.set_title('Scan Quality Analysis - Adjust Slider to Filter Scans', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create slider axis
    ax_slider = plt.axes([0.15, 0.12, 0.7, 0.03])
    slider = Slider(
        ax_slider, 
        'Threshold', 
        min_diff, 
        max_diff,
        valinit=median_diff,
        valstep=(max_diff - min_diff) / 1000,
        color='lightblue'
    )
    
    # Create save button
    ax_button = plt.axes([0.15, 0.05, 0.25, 0.04])
    button_save = Button(ax_button, 'Save Selected Scans', color='lightgreen', hovercolor='green')
    
    # Create text for displaying info
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update(val):
        """Update plot when slider changes"""
        threshold = slider.val
        current_selection['threshold'] = threshold
        
        # Filter scans
        selected_idx, selected_nums = filter_scans_by_threshold(
            diff_sums, scan_numbers, threshold
        )
        current_selection['selected_nums'] = selected_nums
        
        # Update threshold line
        threshold_line.set_ydata([threshold, threshold])
        threshold_line.set_label(f'Threshold = {threshold:.2e}')
        
        # Update selected scans scatter plot
        if len(selected_idx) > 0:
            selected_diff = [diff_sums[i] for i in selected_idx]
            selected_nums_plot = [scan_numbers[i] for i in selected_idx]
            scatter_selected.set_offsets(np.c_[selected_nums_plot, selected_diff])
            scatter_selected.set_label(f'Selected scans ({len(selected_idx)})')
        else:
            scatter_selected.set_offsets(np.empty((0, 2)))
            scatter_selected.set_label('Selected scans (0)')
        
        # Update legend and info text
        ax.legend()
        selection_rate = len(selected_nums) / len(scan_numbers) * 100
        info_text.set_text(f'Threshold: {threshold:.2e}\n'
                          f'Selected: {len(selected_nums)}/{len(scan_numbers)} scans\n'
                          f'Selection rate: {selection_rate:.1f}%')
        
        fig.canvas.draw_idle()
    
    def save_results(event):
        """Save selected scans when button is clicked"""
        selected_nums = current_selection['selected_nums']
        threshold = current_selection['threshold']
        
        if len(selected_nums) == 0:
            print("\nNo scans selected! Please adjust threshold.")
            return
        
        output_file = os.path.join(folder_path, 'selected_scans.txt')
        np.savetxt(output_file, selected_nums, fmt='%d')
        print(f"\n{'='*60}")
        print(f"Results saved!")
        print(f"  Threshold: {threshold:.2e}")
        print(f"  Selected scans: {len(selected_nums)}/{len(scan_numbers)} ({len(selected_nums)/len(scan_numbers)*100:.1f}%)")
        print(f"  Saved to: {output_file}")
        
        # Also save residual data
        diff_file = os.path.join(folder_path, 'scan_residuals.npy')
        np.save(diff_file, {
            'scans': scan_numbers, 
            'residual_sums': diff_sums,
            'threshold': threshold,
            'selected_scans': selected_nums
        })
        print(f"  Residual data saved to: {diff_file}")
        print(f"{'='*60}")
    
    # Connect slider and button to update functions
    slider.on_changed(update)
    button_save.on_clicked(save_results)
    
    # Initial update
    update(median_diff)
    
    print("Displaying interactive plot window...")
    print("(If window doesn't appear, check if it's hidden behind other windows)")
    plt.show()
    print("Plot window closed.")


def main():
    """
    Main function for scan selection workflow
    """
    print("=" * 60)
    print("NMR Scan Selection Based on Reference Scan Difference")
    print("=" * 60)
    
    # Step 1: Select data folder
    folder_path = select_data_folder()
    if folder_path is None:
        return
    
    # Step 2: Get available scans
    available_scans = get_available_scans(folder_path)
    
    if len(available_scans) == 0:
        print("No scan files found in selected folder!")
        return
    
    print(f"\nFound {len(available_scans)} scans")
    print(f"Scan range: {min(available_scans)} - {max(available_scans)}")
    
    # Step 3: Select reference scan
    print(f"\nAvailable scans: {available_scans[:10]}..." if len(available_scans) > 10 else f"\nAvailable scans: {available_scans}")
    
    while True:
        try:
            ref_scan = int(input("\nEnter reference scan number (goodscan): "))
            if ref_scan in available_scans:
                break
            else:
                print(f"Scan {ref_scan} not found. Please choose from available scans.")
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"\nUsing scan {ref_scan} as reference...")
    
    # Step 4: Calculate differences
    print("\nCalculating sum of squared residuals for all scans...")
    print("Method: Σ(scan - reference)² for each scan")
    diff_sums, valid_scans = calculate_difference_sums(folder_path, ref_scan, available_scans)
    
    print(f"\nSuccessfully processed {len(valid_scans)} scans")
    print(f"Sum of squared residuals statistics:")
    print(f"  Min:    {np.min(diff_sums):.2e}")
    print(f"  Max:    {np.max(diff_sums):.2e}")
    print(f"  Mean:   {np.mean(diff_sums):.2e}")
    print(f"  Median: {np.median(diff_sums):.2e}")
    print(f"  Std:    {np.std(diff_sums):.2e}")
    
    # Step 5: Plot and filter
    if IPYWIDGETS_AVAILABLE:
        print("\nStarting interactive threshold selection (Jupyter mode)...")
        interactive_threshold_selection(diff_sums, valid_scans)
    else:
        # Use matplotlib slider for interactive threshold selection
        print("\nStarting interactive threshold selection with slider...")
        print("Adjust the slider to change the threshold")
        print("Click 'Save Selected Scans' button to save results")
        print(f"Using matplotlib backend: {matplotlib.get_backend()}")
        interactive_slider_selection(diff_sums, valid_scans, folder_path)
        print("\nWindow closed. Program finished.")


if __name__ == "__main__":
    main()
            
        
