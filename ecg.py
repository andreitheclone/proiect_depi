import wfdb
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def estimate_blood_pressure(ecg_signal, fs):
    # Find R peaks
    r_peaks, _ = find_peaks(ecg_signal, distance=int(0.5*fs))
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    
    # Calculate heart rate
    heart_rate = 60 / np.mean(rr_intervals)
    
    # Rough estimation of blood pressure based on heart rate
    # Note: This is a very simplified estimation
    # Typical relationship: Higher HR often correlates with higher BP
    systolic = 90 + (0.33 * heart_rate)  # Rough estimate
    diastolic = 60 + (0.25 * heart_rate)  # Rough estimate
    
    return systolic, diastolic, heart_rate

# Specify the path to your .dat and .hea files (without the file extension)
# Example: if you have 'ecg_data.dat' and 'ecg_data.hea', just use 'ecg_data'
record_path = 'C:/Users/gutaa/Videos/proiect_depi/dataset/0001'

try:
    # Read the record (this will read both .dat and .hea files)
    record = wfdb.rdrecord(record_path)
    
    # Get signal information
    print(f"Number of signals: {record.n_sig}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Signal names: {record.sig_name}")
    
    # Get the ECG signal (assuming it's the first channel)
    ecg_signal = record.p_signal[:, 0]
    
    # Estimate blood pressure
    systolic, diastolic, heart_rate = estimate_blood_pressure(ecg_signal, record.fs)
    
    print(f"\nEstimated values:")
    print(f"Heart Rate: {heart_rate:.1f} BPM")
    print(f"Estimated Systolic BP: {systolic:.1f} mmHg")
    print(f"Estimated Diastolic BP: {diastolic:.1f} mmHg")
    
    # Plot the signal
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(ecg_signal)
    plt.title('ECG Signal')
    plt.ylabel('Amplitude')
    
    plt.subplot(212)
    plt.hist(ecg_signal, bins=50)
    plt.title('ECG Signal Distribution')
    plt.xlabel('Amplitude')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Could not find the files {record_path}.dat and/or {record_path}.hea")
except Exception as e:
    print(f"An error occurred: {str(e)}")