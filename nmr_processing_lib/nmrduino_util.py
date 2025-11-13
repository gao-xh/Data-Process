#%%
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import mpld3
#mpld3.enable_notebook()
from scipy.stats import linregress
from scipy.fft import fft       
from scipy.fft import ifft
import struct
import os
from pathlib import Path
from datetime import datetime
from glob import glob

# try:
# # import metadata helpers
#     from ZULFPy.DataManagement.metadataHelper import save_metadata, create_metadata, load_metadata, is_metadata_up_to_date
# except:
#     from PythonCode.DataManagement.metadataHelper import save_metadata, create_metadata, load_metadata, is_metadata_up_to_date

# Create a function to open the file dialog and set the selected folder to a variable

def select_folder():
    def selector():
        folder_path = filedialog.askdirectory()
        if folder_path:
            selected_folder.set(folder_path)
            root.destroy()  # Close the main window

    # Create the main application window
    root = tk.Tk()
    root.title("Folder Selection")

    # Create a StringVar to store the selected folder path
    selected_folder = tk.StringVar()

    # Create a label to display the selected folder path
    folder_label = tk.Label(root, text="Selected Folder:")
    folder_label.pack()

    # Create an entry widget to display the selected folder path
    folder_entry = tk.Entry(root, textvariable=selected_folder, state="readonly")
    folder_entry.pack()

    # Create a button to open the file dialog
    select_button = tk.Button(root, text="Select Folder", command=selector)
    select_button.pack()

    # Start the main event loop
    root.mainloop()

    # The main window is closed after selecting a folder, and the code continues here
    selected_folder_path = selected_folder.get()
    return selected_folder_path + '/'


################################################################################
#NMRduino data interpretation
################################################################################
#path = specify the path of the NMRduino file and which scans you want to look at. Feel free to use select_folder().
#scans = 0 for all scans
#scans = int>0 for a single scan (1 is first)
#scans = [a,b] for averages from a to b, INCLUDING a and b. ie: 1-20 is first 20 scans.
#Output is a list that gives you a 1d array with the time domain data, sampling rate, and acquisition time.
def nmrduino_dat_interp(path,scans, nowarn = False):
    """
    nmrduino_dat_interp Loads NMRduino data from a folder

    Args:
        path (string): path to the folder containing the experiment.seq file
        scans (int or list[2]): Number of scans or range of scans. 0 for all scans

    Returns:
        data list(halp, sampling_rate, acq_time): Time domain data, sampling rate, acquisition time
    """

    path_to_exp = path
    if os.name == 'nt':
        #Path of the experiment you want to analyze. This will automatically correct based on os.
        if "\\" in path_to_exp:
            path_to_exp = path_to_exp.replace("\\","/")
        #path_to_exp = "/" + path_to_exp[3:] + "/"  # This line only works for abspaths
    else:
        path_to_exp = str(Path(path_to_exp).expanduser().resolve()) + "/"

    halp = None

    # Determining the sampling rate directly from the experiment folder
    SR_filename = '0.ini'
    sr_path= os.path.join(path_to_exp,SR_filename)
    with open(sr_path) as SR_file:
        lines = [line.rstrip() for line in SR_file]
    found_NMRduino = 0
    sampling_rate = 6000
    for line in lines:
        if "[NMRduino]" in line:
            found_NMRduino = 1
        if "SampleRate" in line and found_NMRduino:
            sampling_rate = int(float(line.split("=")[1]))
            break
    x = 0
    # Determining the number of scans in the experiment directly from the experiment folder
    while True:
        filename = str(x)+'.dat'
        if not os.path.exists(os.path.join(path_to_exp,filename)):
            num_scans = x
            break
        x += 1
    assert num_scans > 0
    # binary importing and byte reversing to integer16 values
    def open_sesame(scan_range):
        assert len(scan_range) > 0
        halps =[]
        for b in scan_range:
            #filename = path_to_exp+ str(b) + '.dat'
            filename = os.path.join(path_to_exp, str(b) + '.dat')
            with open(filename, 'rb') as file:
                moon_runes = file.read()
            runes_moon = bytearray(moon_runes)
            runes_moon.reverse()
            len_byte_array = len(runes_moon)//2
            integer16 = struct.unpack('<{}h'.format(len_byte_array), runes_moon)
            teger16 = integer16[20:]
            teg = teger16[:-2]
            if b == scan_range[0]:
                halp = np.array(teg)
            else:
                try:
                    halp += np.array(teg)
                    halps.append(halp)
                except:
                    halps.append(np.zeros(len(teg)))
                    print("Error in scan number " + str(b))
        halp = halp/len(scan_range)
        return halp
    #Same as above but for single scan
    def open_sesame_single_scan(b):
        filename = str(b) + '.dat'
        with open(os.path.join(path , filename), 'rb') as file:
            moon_runes = file.read()
        runes_moon = bytearray(moon_runes)
        runes_moon.reverse()
        len_byte_array = len(runes_moon)//2
        integer16 = struct.unpack('<{}h'.format(len_byte_array), runes_moon)
        teger16 = integer16[20:]
        teg = teger16[:-2]
        halp = np.array(teg)
        return halp
    #Reading NMRduino data based on input parameters
    if isinstance(scans,int):
        if scans == 0:
            num_scans = np.arange(0,num_scans,1)
            halp = np.flip(open_sesame(num_scans))
        if isinstance(scans,int) and scans >0 and not scans-1>x:
            if not nowarn:
                print("Scan number "+str(scans) + " only")
            halp = np.flip(open_sesame_single_scan(scans-1))
        if scans-1>x:
            raise ValueError("Scan Doesn't exist")
    if isinstance(scans,list):
        if len(scans) != 2:
            raise ValueError("Please specify a list with 2 elements")
        if scans[0] > scans[1] or scans[0] == 0 or scans[1] ==0 or scans[1]-1>x:
            raise ValueError("Invalid scan range")
        else:
            corrected_scans = np.arange(scans[0]-1,scans[1],1)
            halp = np.flip(open_sesame(corrected_scans))
    return(halp[1:],sampling_rate, len(halp[1:])/sampling_rate)

################################################################################
#NMRduino data interp FAST
################################################################################
# Virtually the same as above, but optimized by the church of chatgpt.
################################################################################

def nmrduino_dat_interp_FAST(path,scans, nowarn = False):
    """
    nmrduino_dat_interp Loads NMRduino data from a folder

    Args:
        path (string): path to the folder containing the experiment.seq file
        scans (int or list[2]): Number of scans or range of scans. 0 for all scans

    Returns:
        data list(halp, sampling_rate, acq_time): Time domain data, sampling rate, acquisition time
    """

    path_to_exp = path
    if os.name == 'nt':
        #Path of the experiment you want to analyze. This will automatically correct based on os.
        if "\\" in path_to_exp:
            path_to_exp = path_to_exp.replace("\\","/")
        #path_to_exp = "/" + path_to_exp[3:] + "/"  # This line only works for abspaths
    else:
        path_to_exp = str(Path(path_to_exp).expanduser().resolve()) + "/"

    halp = None

    # Determining the sampling rate directly from the experiment folder
    SR_filename = '0.ini'
    sr_path= os.path.join(path_to_exp,SR_filename)
    with open(sr_path) as SR_file:
        lines = [line.rstrip() for line in SR_file]
    found_NMRduino = 0
    sampling_rate = 6000
    for line in lines:
        if "[NMRduino]" in line:
            found_NMRduino = 1
        if "SampleRate" in line and found_NMRduino:
            sampling_rate = int(float(line.split("=")[1]))
            break
    # # Determining the number of scans in the experiment directly from the experiment folder
    # while True:
    #     filename = str(x)+'.dat'
    #     if not os.path.exists(os.path.join(path_to_exp,filename)):
    #         num_scans = x
    #         break
    #     x += 1
    # assert num_scans > 0
    # binary importing and byte reversing to integer16 values
    dat_files = sorted(glob(os.path.join(path_to_exp, '*.dat')))
    num_scans = len(dat_files)
    x = num_scans
    def open_sesame(scan_range):
        assert len(scan_range) > 0
        halps =[]
        for b in scan_range:
            #filename = path_to_exp+ str(b) + '.dat'
            filename = os.path.join(path_to_exp, str(b) + '.dat')
            with open(filename, 'rb') as file:
                moon_runes = file.read()
            runes_moon = bytearray(moon_runes)
            runes_moon.reverse()
            len_byte_array = len(runes_moon)//2
            integer16 = struct.unpack('<{}h'.format(len_byte_array), runes_moon)
            teger16 = integer16[20:]
            teg = teger16[:-2]
            if b == scan_range[0]:
                halp = np.array(teg)
            else:
                try:
                    halp += np.array(teg)
                    halps.append(halp)
                except:
                    halps.append(np.zeros(len(teg)))
                    print("Error in scan number " + str(b))
        halp = halp/len(scan_range)
        return halp
    #Same as above but for single scan
    def open_sesame_single_scan(b):
        filename = str(b) + '.dat'
        with open(os.path.join(path , filename), 'rb') as file:
            moon_runes = file.read()
        runes_moon = bytearray(moon_runes)
        runes_moon.reverse()
        len_byte_array = len(runes_moon)//2
        integer16 = struct.unpack('<{}h'.format(len_byte_array), runes_moon)
        teger16 = integer16[20:]
        teg = teger16[:-2]
        halp = np.array(teg)
        return halp
    #Reading NMRduino data based on input parameters
    if isinstance(scans,int):
        if scans == 0:
            # find number of experiments
            num_scans = np.arange(0,num_scans,1)
            halp = np.flip(open_sesame(num_scans))
        if isinstance(scans,int) and scans >0 and not scans-1>x:
            if not nowarn:
                print("Scan number "+str(scans) + " only")
            halp = np.flip(open_sesame_single_scan(scans-1))
        if scans-1>x:
            raise ValueError("Scan Doesn't exist")
    if isinstance(scans,list):
        if len(scans) != 2:
            raise ValueError("Please specify a list with 2 elements")
        if scans[0] > scans[1] or scans[0] == 0 or scans[1] ==0 or scans[1]-1>x:
            raise ValueError("Invalid scan range")
        else:
            corrected_scans = np.arange(scans[0]-1,scans[1],1)
            halp = np.flip(open_sesame(corrected_scans))
    return(halp[1:],sampling_rate, len(halp[1:])/sampling_rate)

################################################################################
#NMRduino data interp FASTER with Chatgpt!
################################################################################
# Virtually the same as above, but optimized by the church of chatgpt and implemented by him/her as well!
################################################################################

def nmrduino_dat_interp_chatgpt(path, scans, nowarn=False):
    path_to_exp = Path(path).resolve()

    # Get sampling rate from 0.ini
    sampling_rate = 6000  # Default fallback
    with open(path_to_exp / '0.ini', 'r') as f:
        in_nmr_block = False
        for line in f:
            line = line.strip()
            if line == '[NMRduino]':
                in_nmr_block = True
            elif in_nmr_block and line.startswith('SampleRate'):
                sampling_rate = int(float(line.split('=')[1]))
                break

    # Get sorted list of .dat files
    dat_files = sorted(path_to_exp.glob("*.dat"), key=lambda f: int(f.stem))
    num_scans_total = len(dat_files)

    def read_scan(scan_index):
        """Reads and processes a single .dat scan."""
        filename = path_to_exp / f"{scan_index}.dat"
        with open(filename, 'rb') as file:
            byte_data = bytearray(file.read())[::-1]
        int16_data = struct.unpack('<{}h'.format(len(byte_data)//2), byte_data)
        np_data = np.array(int16_data[20:-2], dtype=np.int16)
        return np_data

    # Determine which scans to read
    if isinstance(scans, int):
        if scans == 0:
            scan_indices = range(num_scans_total)
        elif scans > 0 and scans - 1 < num_scans_total:
            if not nowarn:
                print(f"Scan number {scans} only")
            scan = read_scan(scans - 1)
            return np.flip(scan)[1:], sampling_rate, len(scan[1:]) / sampling_rate
        else:
            raise ValueError("Invalid scan number")
    elif isinstance(scans, list) and len(scans) == 2:
        start, end = scans
        if start <= 0 or end <= 0 or start > end or end > num_scans_total:
            raise ValueError("Invalid scan range")
        scan_indices = range(start - 1, end)
    else:
        raise ValueError("Invalid scans input")

    # Process multiple scans
    summed_data = None
    for idx in scan_indices:
        try:
            scan = read_scan(idx)
            if summed_data is None:
                summed_data = scan
            else:
                summed_data += scan
        except Exception as e:
            if not nowarn:
                print(f"Error in scan {idx}: {e}")
            if summed_data is None:
                summed_data = np.zeros_like(scan)
            else:
                summed_data += np.zeros_like(scan)

    averaged = summed_data #/ len(scan_indices)
    return np.flip(averaged)[1:], sampling_rate, len(averaged[1:]) / sampling_rate


# def nmrduino_dat_interp_stack(path, cache = False):
#     #TODO: remove cache and move it up a level
#     """
#     nmrduino_dat_interp_stdev Alternative version to nmrduino_dat_interp that returns the full stacked data-set without projection

#     Args:
#         path (string): path to the folder containing the experiment.seq file
#         cache (bool): if True, will cache the data-set for faster loading next time

#     Returns:
#         np.array: Stacked data-set (shape: (num_scans, len(halp)))
#     """
#     path_to_exp = path
#     if os.name == 'nt':
#         #Path of the experiment you want to analyze. This will automatically correct based on os.
#         if "\\" in path_to_exp:
#             path_to_exp = path_to_exp.replace("\\","/")
#         path_to_exp = "/" + path_to_exp[3:] + "/"
#     else:
#         path_to_exp = str(Path(path_to_exp).expanduser().resolve()) + "/"

#     # check if cached
#     if cache:
#         meta_data = load_metadata(path_to_exp)
#         if is_metadata_up_to_date(meta_data, path_to_exp):
#             with open(os.path.join(path_to_exp, 'cache.npy'), 'rb') as f:
#                 return np.load(f)
    
#     # Determining the sampling rate directly from the experiment folder
#     SR_filename = '0.ini'
#     sr_path= os.path.join(path_to_exp,SR_filename)
#     with open(sr_path) as SR_file:
#         lines = [line.rstrip() for line in SR_file]
#     found_NMRduino = 0
#     sampling_rate = 6000
#     for line in lines:
#         if "[NMRduino]" in line:
#             found_NMRduino = 1
#         if "SampleRate" in line and found_NMRduino:
#             sampling_rate = int(float(line.split("=")[1]))
#             break
#     if not found_NMRduino:
#         print("No NMRduino data found, using default sampling rate of 6000")
    
#     # find number of experiments
#     files = os.listdir(path_to_exp)
#     files = [f for f in files if f.endswith('.dat')]
#     num_scans = len(files)

#     # binary importing and byte reversing to integer16 values
#     def parse_dat_files_in_exp(scan_range):
#         """
#         parse_dat_files_in_exp Reads all dat files in the scan_range and collects them into a halps array

#         Args:
#             scan_range (list): list of all scans to include

#         Returns:
#             np.array: array with shape (len(scan_range), len(halp)) containing all scans in scan_range
#         """
#         halps = []
#         for b in scan_range:
#             filename = path_to_exp+ str(b) + '.dat'
#             with open(filename, 'rb') as file:
#                 moon_runes = file.read()
#             runes_moon = bytearray(moon_runes)
#             runes_moon.reverse()
#             len_byte_array = len(runes_moon)//2
#             integer16 = struct.unpack('<{}h'.format(len_byte_array), runes_moon)
#             teger16 = integer16[20:]
#             teg = teger16[:-2]

#             halp = np.array(teg)
#             halps.append(halp)
#         return np.array(halps).swapaxes(0,1)
    
#     #Reading NMRduino data based on input parameters
#     scan_range = np.arange(0,num_scans,1)

#     halps = np.flip(parse_dat_files_in_exp(scan_range), axis = 0)

#     # create cache
#     if cache:
#         with open(os.path.join(path_to_exp, 'cache.npy'), 'wb') as f:
#             # create metadata
#             meta_data = create_metadata("NMRduino", "stacked_data", ["cache"])
#             save_metadata(meta_data, f)
#             np.save(f, halps)

#     return halps, sampling_rate, np.shape(halps)[0]/sampling_rate


################################################################################
#Scan number extraction.
################################################################################
#This function determines the number of scans/averages directly from the experiment folder.
#This takes the input of a path and returns an int indicating the number of scans.
def scan_number_extraction(path):
    path_to_exp = path
    if os.name == 'nt':
        #Path of the experiment you want to analyze. This will automatically correct the path based on the os.
        if "\\" in path_to_exp:
            path_to_exp = path_to_exp.replace("\\","/")
        path_to_exp = "/" + path_to_exp[3:] + "/"
    else:
        path_to_exp = str(Path(path_to_exp).expanduser().resolve()) + "/"
    x = 0
    # Determining the number of scans in the experiment directly from the experiment folder
    while True:
        filename = str(x)+'.dat'
        if not os.path.exists(path_to_exp+filename):
            num_scans = x
            break
        x += 1
    return(x)


################################################################################
#Single-experiment standard deviation of peak height
################################################################################
#Finds the standard deviation of the peak height to give error bars on calibration curves.
#As of right now, this only has capability of doing this with every scan, not a slice.
#This also doesn't incorporate any phasing, only the absolute value spectrum.
def single_stddev_peak_height(path,freq_range):
    scans = scan_number_extraction(path)
    assert scans > 0 #assert that there are scans in the folder
    peak_height_list = []
    for scan in range(scans):
        data = nmrduino_dat_interp(path,scan+1, nowarn=True)
        yf = np.abs(fft(data[0])) # type: ignore
        xf = np.arange(0, data[1], data[1]/np.size(yf))
        index_peak_i = np.where(xf >= freq_range[0])[0][0]
        index_peak_f = np.where(xf >= freq_range[1])[0][0]
        peak_height_list.append(max(yf[index_peak_i:index_peak_f]))
    assert len(peak_height_list) == scans


    return(np.std(peak_height_list))
    
################################################################################
#Task scheduler standard deviation (error bars) generator
################################################################################
#This function takes a task scheduler file and generates standard deviations for the peak of interest (specified by the frequency range (freq_range))
#This is used as error bars for the calibration curve plots.
def task_scheduler_stddev(path,freq_range):
    path_to_exp = path
    stddev_list = []
    if os.name == 'nt':
        #Path of the experiment you want to analyze. This will automatically correct the path based on the os.
        if "\\" in path_to_exp:
            path_to_exp = path_to_exp.replace("\\","/")
        path_to_exp = "/" + path_to_exp[3:]
    else:
        path_to_exp = str(Path(path_to_exp).expanduser().resolve()) + "/"
    x = 0
    # Determining the number of scans in the experiment directly from the experiment folder
    while True:
        filename = str(x)
        if not os.path.exists(path_to_exp+filename):
            break
        single_stddev = single_stddev_peak_height(path_to_exp+filename+"/",freq_range)
        stddev_list.append(single_stddev)
        x += 1
        print("path: " +path+filename)
    print("x = "+str(x))
    print("stddev list length: "+ str(len(stddev_list)))
    return(print(stddev_list))
################################################################################
#Bandpass filter
################################################################################
#This function takes a path from an NMRduino experiment and a frequency range and generates time domain data filtered only from frequency range given.
#It returns the filtered time domain data and the original acquisition time for plotting purposes.
def bandpass(path,freq_range):
    index_peak_i = freq_range[0]
    index_peak_f = freq_range[1]
    data = nmrduino_dat_interp(path,0)
    halp,sampling_rate,acq_time = data
    yf = fft(halp)
    xf = np.arange(0, sampling_rate, sampling_rate/np.size(yf))
    index_peak_i = np.where(xf >= freq_range[0])[0][0]
    index_peak_f = np.where(xf >= freq_range[1])[0][0]
    yf[:index_peak_i] = yf[index_peak_i]
    yf[index_peak_f:] = yf[index_peak_f]
    halp_trunc=ifft(yf)
    return(halp_trunc,acq_time)

################################################################################
#Bandpass filter (from data)
################################################################################
#This function takes a path from an NMRduino experiment and a frequency range and generates time domain data filtered only from frequency range given.
#It returns the filtered time domain data and the original acquisition time for plotting purposes.
def bandpass_data(halp,sampling_rate,acq_time,freq_range):
    index_peak_i = freq_range[0]
    index_peak_f = freq_range[1]
    yf = fft(halp)
    xf = np.arange(0, sampling_rate, sampling_rate/np.size(yf))
    index_peak_i = np.where(xf >= freq_range[0])[0][0]
    index_peak_f = np.where(xf >= freq_range[1])[0][0]
    yf[:index_peak_i] = yf[index_peak_i]
    yf[index_peak_f:] = yf[index_peak_f]
    halp_trunc=ifft(yf)
    return(halp_trunc,acq_time)

################################################################################
#Signal to noise ratio (SNR) calculation.
################################################################################
#This function takes an already processed spectrum and calculates the SNR using a frequency range for the peak and a frequency
#range for the noise. The reason it takes an already processed spectrum is that this can be done in post, after corrections (phasing,
#etc. have already been applied. It returns the SNR value as well as prints the peak height and SNR as well.
def snr_calc(xf,yf,peak_freq_range,noise_freq_range,nowarn=False):
    # Acquiring noise in specified frequenc range:
    index_noise_i = np.where(xf >= noise_freq_range[0])[0][0]
    index_noise_f = np.where(xf >= noise_freq_range[1])[0][0]
    yf_noise = yf[index_noise_i:index_noise_f]
    xf_noise = xf[index_noise_i:index_noise_f]
    # Finding first order baseline for correction:
    slope, intercept, r_value, p_value, std_err = linregress(xf_noise, yf_noise)
    yf_baseline = slope*xf_noise+intercept
    yf_noise_corrected = yf_noise - yf_baseline
    # Finding the value for the maximum peak in specified frequency range:
    index_peak_i = np.where(xf >= peak_freq_range[0])[0][0]
    index_peak_f = np.where(xf >= peak_freq_range[1])[0][0]
    peak_range = yf[index_peak_i:index_peak_f]
    peak_offset = np.mean(yf_noise)
    max_val = max(peak_range) - peak_offset
    
    # Find the index of a specific value in yf
    avg_noise_power =np.mean(np.sqrt(np.square(yf_noise_corrected)))
    signal_to_noise = max_val/avg_noise_power
    if not nowarn:
        print("SNR = "+ str(signal_to_noise))
        print('Peak height = ' + str(max_val))
    return(signal_to_noise)


################################################################################
#NMRduino plotter
################################################################################
#This function conveniently plots data taken from NMRduino directly from the path.
#It's inputs are the same as nmrduino_dat_interp, and plots the fft spectrum accordingly.
#Currently doesn't provide phasing and plots the absolute value spectrum.
def process(path, scans, filter_funcs=None, plot=False):
    """
    Analyze data using filter functions

    Parameters:
        path (string): folder path containing scans. obtained from select_folder()
        scans (int or list[2]): number of scans or range of scans. 0 for all scans
        filter_funcs (list[]): list containing a sequence of filter functions to be run on the data
        plot (boolean): if True, will generate plots for appropriate filter functions, used for debugging

    Current allowed filter functions: time_domain, frequency_domain(freq_i=0, freq_f=400), phase(val), zerofill(factor)
    """
    data = nmrduino_dat_interp(path,scans)
    halp = data[0]
    sr = data[1] #sampling rate
    acq = data[2] #acquisition time
    if not filter_funcs:
        raise Exception("Filter Functions Required")

    num = 1
    #initial fourier transform to obtain xf and yf, the initial frequency domain data
    yf = fft(halp-np.mean(halp[:1000])) #don't abs xf and yf before phasing to keep the raw data
    xf = np.arange(0, sr, sr/np.size(yf))


    for f in filter_funcs:
        halp, xf, yf, sr, acq, num = f(halp, xf, yf, sr, acq, num, plot)

    plt.show()
        


def time_domain(halp, xf, yf, sr, acq, num, plot):
    """
    Filter Function for 'process' used for plotting time domain data
    No data is changed
    """
    if plot:
        plt.figure(num)
        plt.title("Time Domain Data")
        plt.xlabel("Seconds")
        plt.ylabel("Voltage")
        plt.plot(np.arange(0,acq,float(acq)/len(halp)), halp, "k", linewidth=.5)
    return halp, xf, yf, sr, acq, num+1



def frequency_domain(freq_i=0, freq_f=400):
    """
    Filter Function for 'process' that plots frequency_domain data
    No data is changed

    Parameters:
        freq_i (float): the starting frequency for plotting
        freq_f (float): the ending frequency for plotting
    """
    def frequency_domain_helper(halp, xf, yf, sr, acq, num, plot): 
        if plot:
            # Use absolute value of the complex data to plot
            plotting_xf = np.abs(xf[:int(np.rint(len(xf)))])
            plotting_yf = np.abs(yf[:int(np.rint(len(yf)))])
            # Create a boolean mask for the frequency range
            mask = (plotting_xf >= freq_i) & (plotting_xf <= freq_f)
            # Apply the mask to select the values within the specified range
            plot_xf = plotting_xf[mask]
            plot_yf = plotting_yf[mask]
            #plot
            plt.figure(num)
            plt.title("Frequency Domain Data")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Signal [a.u.]")
            plt.plot(plot_xf, plot_yf, "k", linewidth=.5)
            plt.xlim(plot_xf[0], plot_xf[len(plot_xf)-1])
            plt.ylim(min(plot_yf), max(plot_yf[int(len(plot_yf)/50):]))
        return halp, xf, yf, sr, acq, num+1

    return frequency_domain_helper

def apply_phase_shift(yf, val=0):
    """
    Applies a phase shift to the yf data

    Parameters:
        yf (np array): the complex Fourier-transformed data
        val (float): the phase shift applied in radians
    """
    return yf * np.exp(1j * val)

def phase(val, freq_i=0, freq_f=400):
    """
    Filter Function for 'process' that plots phased frequency domain data
    Changes the data in yf

    Parameters:
        val (float): the desired phase shift for the data (default is no shift)
        freq_i (float): the starting frequency
        freq_f (float): the ending frequency
    """
    def phase_helper(halp, xf, yf, sr, acq, num, plot):
        yf = apply_phase_shift(yf, val)
        halp = ifft(yf) 
        if plot:
            # Use absolute value of the complex data to plot
            plotting_xf = np.abs(xf[:int(np.rint(len(xf)))])
            plotting_yf = np.abs(yf[:int(np.rint(len(yf)))])
            # Create a boolean mask for the frequency range
            mask = (plotting_xf >= freq_i) & (plotting_xf <= freq_f)
            # Apply the mask to select the values within the specified range
            plot_xf = plotting_xf[mask]
            plot_yf = plotting_yf[mask]
            plt.figure(num)
            plt.title("Newly Phased Frequency Domain Data")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Signal [a.u.]")
            plt.plot(plot_xf, plot_yf, "k", linewidth=.5)
            plt.xlim(plot_xf[0], plot_xf[len(plot_xf)-1])
            plt.ylim(min(plot_yf), max(plot_yf[int(len(plot_yf)/50):]))
        return halp, xf, yf, sr, acq, num+1

    return phase_helper

def zerofill(factor=2):
    """
    Filter Function for 'process' that zero-fills the data, increasing the time by factor to increase resolution
    Changes the data in halp

    Parameters:
    factor (float): the zero fill factor; by what factor should the time scale increase
    """
    def zero_helper(halp, xf, yf, sr, acq, num, plot):
        new_halp = np.zeros(len(halp) * factor)
        finalval = halp[-1]
        new_halp[:len(halp)] = halp
        new_halp[len(halp):] = finalval

        yf = fft(new_halp-np.mean(new_halp[:1000]))
        xf = np.arange(0, sr, sr/np.size(yf))
        xf = np.abs(xf[:int(np.rint(len(xf)))])
        yf = np.abs(yf[:int(np.rint(len(yf)))])
        return new_halp, xf, yf, sr, acq, num
    return zero_helper


if __name__ == "__main__": #only run if file is executed, if imported it is ignored
    today = datetime.now()
    pi = np.pi
    path = select_folder()
    #process(path, 0, [time_domain, frequency_domain(200, 210), frequency_domain(220, 225), zerofill(2), time_domain, 
                      #frequency_domain(200, 210), frequency_domain(220, 225), phase(3, 210, 230)], True)

    process(path, 0, [frequency_domain(210, 230), phase(0.75, 210, 230), phase(0.75, 210, 230), phase(0.75, 210, 230), phase(0.75, 210, 230), phase(-3, 210, 230)], True)
