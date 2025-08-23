import os
import re
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

# === Feature functions ===
def bandpower(data, sf, bands):
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

def common_average_reference(data_array):
    mean_signal = np.mean(data_array, axis=0)
    return data_array - mean_signal

# === Config ===
CLEANED_PATH = os.path.join("..", "results", "seediv_ica_cleaned")
RESULTS_PATH = os.path.join("..", "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

SESSIONS = ["1", "2", "3"]
CHANNEL_INDICES = [14, 23, 32]
SF_TARGET = 200
EPOCH_LEN = 20  # seconds
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5  # 50%
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "seediv_ica_features_20s_overlap.npz")

# Original session labels for subjects
session_labels = {
    "1": [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    "2": [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    "3": [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

X, y, subject_ids = [], [], []

for sess in SESSIONS:
    sess_folder = os.path.join(CLEANED_PATH)
    for file in os.listdir(sess_folder):
        if not file.endswith(".npz") or f"sess{sess}" not in file:
            continue

        filepath = os.path.join(sess_folder, file)
        print(f"Loading cleaned data file: {file}")

        data = np.load(filepath, allow_pickle=True)
        trial_keys = data.files  # e.g., ['trial_0', 'trial_1', ..., 'trial_23']

        # Extract subject number from filename like "sub1_sess3_cleaned_trials.npz"
        subject_num = int(re.findall(r"\d+", file)[0])
        subject_id = f"sub{subject_num}"
        orig_label = session_labels[sess][subject_num - 1]
        bin_label = 0 if orig_label in [0, 3] else 1

        for key in trial_keys:
            trial_array = data[key]  # shape (channels x samples)

            # Apply CAR referencing
            trial_array = common_average_reference(trial_array)

            step = int(EPOCH_SAMPLES * (1 - OVERLAP))
            n_windows = (trial_array.shape[1] - EPOCH_SAMPLES) // step + 1
            if n_windows <= 0:
                print(f"  Trial too short for sliding windows: {key} in {file}")
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

X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

np.savez(OUTFILE, features=X, labels=y, subject_ids=subject_ids)

print(f"Saved extracted features to: {OUTFILE}")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique subjects/files:", len(np.unique(subject_ids)))