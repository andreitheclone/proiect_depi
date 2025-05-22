import wfdb
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from glob import glob

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

def process_database(database_path):
    # Get all .dat files in the directory
    dat_files = glob(os.path.join(database_path, "*.dat"))
    results = []

    for dat_file in dat_files:
        # Remove .dat extension to get record path
        record_path = dat_file[:-4]
        try:
            # Read the record
            record = wfdb.rdrecord(record_path)
            
            # Get the ECG signal (first channel)
            ecg_signal = record.p_signal[:, 0]
            
            # Estimate blood pressure
            systolic, diastolic, heart_rate = estimate_blood_pressure(ecg_signal, record.fs)
            
            # Store results
            results.append({
                'record': os.path.basename(record_path),
                'heart_rate': heart_rate,
                'systolic': systolic,
                'diastolic': diastolic
            })
            
            print(f"\nProcessed {os.path.basename(record_path)}:")
            print(f"Heart Rate: {heart_rate:.1f} BPM")
            print(f"Systolic BP: {systolic:.1f} mmHg")
            print(f"Diastolic BP: {diastolic:.1f} mmHg")
            
        except Exception as e:
            print(f"Error processing {record_path}: {str(e)}")
    
    return results

# Specify the database directory
database_path = 'C:/Users/gutaa/Videos/proiect_depi/dataset'

# Process all records
results = process_database(database_path)

# Calculate average values
if results:
    avg_hr = np.mean([r['heart_rate'] for r in results])
    avg_systolic = np.mean([r['systolic'] for r in results])
    avg_diastolic = np.mean([r['diastolic'] for r in results])
    
    print("\nDatabase Summary:")
    print(f"Average Heart Rate: {avg_hr:.1f} BPM")
    print(f"Average Systolic BP: {avg_systolic:.1f} mmHg")
    print(f"Average Diastolic BP: {avg_diastolic:.1f} mmHg")
    
    # Plot distribution of results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist([r['heart_rate'] for r in results], bins=20)
    plt.title('Heart Rate Distribution')
    plt.xlabel('BPM')
    
    plt.subplot(132)
    plt.hist([r['systolic'] for r in results], bins=20)
    plt.title('Systolic BP Distribution')
    plt.xlabel('mmHg')
    
    plt.subplot(133)
    plt.hist([r['diastolic'] for r in results], bins=20)
    plt.title('Diastolic BP Distribution')
    plt.xlabel('mmHg')
    
    plt.tight_layout()
    plt.show()
else:
    print("No records were processed successfully.")