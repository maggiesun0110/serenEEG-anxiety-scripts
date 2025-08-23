import os
import re
import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt, resample, iirnotch, windows
from scipy.integrate import simpson

# === Feature functions ===
def bandpower(data, sf, bands):
    # Apply Welch PSD with Hann window to reduce spectral leakage
    freqs, psd = welch(data, sf, window='hann', nperseg=sf * 2)
    return [
        simpson(psd[(freqs >= low) & (freqs <= high)],
                freqs[(freqs >= low) & (freqs <= high)])
        for low, high in bands
    ]

def hjorth_parameters(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / (var_zero + 1e-10))
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-10)) / (mobility + 1e-10)
    return [var_zero, mobility, complexity]

def notch_filter(data, fs, freq=60.0, Q=30.0):
    # Notch filter to remove powerline noise at 60 Hz
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='bandpass', fs=sf)
    return filtfilt(b, a, data)

def preprocess(data, orig_sf, target_sf=200):
    # Apply notch filter first to remove powerline noise
    data = notch_filter(data, orig_sf, freq=60.0)
    
    # Bandpass filter the data between low and high freq
    data = bandpass_filter(data, orig_sf)
    
    # Resample if original sampling freq is different from target
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))
    return data

def common_average_reference(data_array):
    # data_array shape: channels x samples
    mean_signal = np.mean(data_array, axis=0)
    return data_array - mean_signal

# === Labeling function ===
def binary_label_neutral_happy_vs_sad_fear(label):
    return 0 if label in [0, 3] else 1

# === Config ===
BASE_PATH = os.path.join("..", "..", "data", "SEED_IV", "eeg_raw_data")
RESULTS_PATH = os.path.join("..", "results")
CHANNEL_INDICES = [14, 23, 32] 
EPOCH_LEN = 20  # seconds
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5  # 50%
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "seediv_anxiety_8fts_20s_overlap.npz")

session_labels = {
    "1": [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    "2": [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    "3": [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

X, y, subject_ids = [], [], []

sessions = ["1", "2", "3"]

for session_label in sessions:
    session_path = os.path.join(BASE_PATH, session_label)
    if not os.path.exists(session_path):
        print(f"Session folder not found: {session_path}")
        continue

    files = [f for f in os.listdir(session_path) if f.endswith(".mat")]

    print(f"Processing session {session_label}, {len(files)} files found")

    for file in files:
        filepath = os.path.join(session_path, file)

        print(f"Loading file: {filepath}")
        try:
            mat_data = sio.loadmat(filepath)

            all_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            pattern = re.compile(r'.*_eeg\d+$')
            trial_keys = [k for k in all_keys if pattern.match(k)]
            trial_keys = sorted(trial_keys, key=lambda x: int(re.findall(r'\d+$', x)[0]))

            print(f"  Trials found in {file}: {trial_keys}")

            subject_num = int(file.split('_')[0])
            subject_id = f"sub{subject_num}"
            orig_label = session_labels[session_label][subject_num - 1]
            bin_label = binary_label_neutral_happy_vs_sad_fear(orig_label)

            if not trial_keys:
                print(f"  No trials found in {file}")
                continue

            for trial_key in trial_keys:
                trial_data = mat_data[trial_key]  # shape (channels, samples)

                processed_channels = []
                for ch_idx in CHANNEL_INDICES:
                    raw = trial_data[ch_idx, :]
                    filtered = preprocess(raw, orig_sf=200, target_sf=SF_TARGET)
                    processed_channels.append(filtered)

                trial_array = np.array(processed_channels)

                # Apply Common Average Reference (CAR)
                trial_array = common_average_reference(trial_array)

                step = int(EPOCH_SAMPLES * (1 - OVERLAP))
                n_windows = (trial_array.shape[1] - EPOCH_SAMPLES) // step + 1
                if n_windows <= 0:
                    print(f"  Trial too short for sliding windows: {trial_key} in {file}")
                    continue

                for w in range(n_windows):
                    feats = []
                    start = w * step
                    end = start + EPOCH_SAMPLES
                    for ch_data in trial_array:
                        epoch = ch_data[start:end]

                        bp = bandpower(epoch, SF_TARGET, BANDS)
                        hjorth = hjorth_parameters(epoch)

                        feats.extend(bp)
                        feats.extend(hjorth)

                    X.append(feats)
                    y.append(bin_label)
                    subject_ids.append(subject_id)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

os.makedirs(RESULTS_PATH, exist_ok=True)
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

np.savez(OUTFILE, features=X, labels=y, subject_ids=subject_ids)

print(f"Saved features to: {OUTFILE}")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique subjects/files:", len(np.unique(subject_ids)))