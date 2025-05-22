import os
import wfdb
import numpy as np
from scipy.signal import find_peaks
import csv  # <-- Add this import

# ===== User Configuration =====
# Set the directory where your .dat and .hea files reside
DATA_DIR = 'C:/Users/gutaa/Videos/proiect_depi/dataset'
CSV_OUTPUT = 'results.csv'  # Output CSV file

# Attempt to import GQRS detector (wfdb >=3.x) or fall back
try:
    from wfdb.processing import gqrs_detect
except ImportError:
    try:
        from wfdb import gqrs
        gqrs_detect = gqrs.gqrs_detect
    except ImportError:
        raise ImportError("GQRS detector not found in wfdb. Please upgrade wfdb: pip install wfdb>=3.4.0")

def process_record(record_path, record_name):
    record = wfdb.rdrecord(record_path)
    fs = record.fs  # sampling frequency

    # Identify the ECG channel (common names: 'ECG', 'II', 'MLII')
    ecg_candidates = [name for name in record.sig_name if 'ECG' in name.upper() or name.upper() in ('II', 'MLII')]
    if not ecg_candidates:
        print(f'[{record_name}] No ECG channel found in record signals: {record.sig_name}')
        return
    ecg_idx = record.sig_name.index(ecg_candidates[0])
    ecg_signal = record.p_signal[:, ecg_idx]

    # Detect R-peaks using the GQRS detector
    r_peaks = gqrs_detect(sig=ecg_signal, fs=fs)
    if len(r_peaks) < 2:
        print(f'[{record_name}] Insufficient R-peaks detected for heart rate calculation')
        return

    # Compute heart rate (bpm)
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    heart_rate = 60.0 / np.mean(rr_intervals)

    # Identify the blood pressure channel (common names: 'NIBP', 'ABP', 'ART')
    bp_candidates = [name for name in record.sig_name if any(tag in name.upper() for tag in ('NIBP', 'ABP', 'ART'))]
    if not bp_candidates:
        print(f'[{record_name}] No blood pressure channel found in record signals: {record.sig_name}')
        return
    bp_idx = record.sig_name.index(bp_candidates[0])
    bp_signal = record.p_signal[:, bp_idx]

    # Detect systolic peaks (maxima)
    min_distance = int(0.5 * fs)  # at least 0.5 seconds between peaks
    systolic_peaks, _ = find_peaks(bp_signal, distance=min_distance)
    if len(systolic_peaks) == 0:
        print(f'[{record_name}] No systolic peaks detected')
        return None  # <-- Return None if failed
    systolic_values = bp_signal[systolic_peaks]
    systolic_pressure = np.mean(systolic_values)

    # Detect diastolic troughs (minima)
    diastolic_troughs, _ = find_peaks(-bp_signal, distance=min_distance)
    if len(diastolic_troughs) == 0:
        print(f'[{record_name}] No diastolic troughs detected')
        return None  # <-- Return None if failed
    diastolic_values = bp_signal[diastolic_troughs]
    diastolic_pressure = np.mean(diastolic_values)

    print(f'[{record_name}] Heart rate: {heart_rate:.1f} bpm | '
          f'Systolic BP: {systolic_pressure:.1f} mmHg | '
          f'Diastolic BP: {diastolic_pressure:.1f} mmHg')

    # Return the results as a tuple
    return (record_name, heart_rate, systolic_pressure, diastolic_pressure)

def main():
    # Find all .hea files in the data directory
    records = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.hea')]
    if not records:
        print('No .hea files found in the data directory.')
        return

    # Ask user for the number of files to process
    try:
        n_files = int(input(f'How many files to process? (1-{len(records)}): '))
        if n_files < 1 or n_files > len(records):
            print('Invalid number, processing all files.')
            n_files = len(records)
    except Exception:
        print('Invalid input, processing all files.')
        n_files = len(records)

    results = []  # <-- Collect results here

    for record_name in records[:n_files]:
        record_path = os.path.join(DATA_DIR, record_name)
        try:
            result = process_record(record_path, record_name)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f'[{record_name}] Error: {e}')

    # Write results to CSV
    with open(CSV_OUTPUT, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'heart_rate', 'systolic_pressure', 'diastolic_pressure'])
        for row in results:
            writer.writerow([row[0], f'{row[1]:.1f}', f'{row[2]:.1f}', f'{row[3]:.1f}'])

    print(f'Results saved to {CSV_OUTPUT}')

if __name__ == '__main__':
    main()
