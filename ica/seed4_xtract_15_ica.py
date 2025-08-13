import os
import re
import numpy as np

from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import skew

# === Feature functions ===
def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
    return [
        simpson(psd[(freqs >= low) & (freqs <= high)],
                freqs[(freqs >= low) & (freqs <= high)])
        for low, high in bands
    ]

def band_ratios(powers):
    delta, theta, alpha, beta, gamma = powers
    eps = 1e-10
    return [
        delta / (theta + eps),
        delta / (alpha + eps),
        alpha / (beta + eps),
        theta / (alpha + eps),
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

def statistical_features(data):
    return [np.mean(data), np.std(data), skew(data)]

# === Labeling function ===
def binary_label_neutral_happy_vs_sad_fear(label):
    return 0 if label in [0, 3] else 1

# === Config ===
CLEANED_PATH = os.path.join("..", "results", "seediv_ica_cleaned")
RESULTS_PATH = os.path.join("..", "results")
CHANNEL_COUNT = 3
SF_TARGET = 200
EPOCH_LEN = 20  # seconds
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5  # 50%
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "seediv_ica_cleaned_15fts_20s_overlap.npz")

session_labels = {
    "1": [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    "2": [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    "3": [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

X, y, subject_ids = [], [], []

sessions = ["1", "2", "3"]

for sess in sessions:
    sess_folder = os.path.join(CLEANED_PATH)
    if not os.path.exists(sess_folder):
        print(f"Cleaned data folder not found: {sess_folder}")
        continue

    # Get all cleaned npz files for this session
    # Filenames example: sub1_sess1_cleaned_trials.npz
    all_files = [f for f in os.listdir(sess_folder) if f.endswith(".npz") and f"_sess{sess}_" in f]

    print(f"Processing cleaned session {sess}, {len(all_files)} files found")

    for f in all_files:
        filepath = os.path.join(sess_folder, f)
        print(f"Loading cleaned data file: {filepath}")

        try:
            data = np.load(filepath)
            # Keys are trial_0, trial_1, ... each shape: (channels x samples)
            trial_keys = [k for k in data.keys() if k.startswith("trial_")]
            trial_keys = sorted(trial_keys, key=lambda x: int(x.split("_")[1]))

            # Extract subject num from filename: sub{num}_sess{sess}_cleaned_trials.npz
            subject_num = int(re.findall(r"sub(\d+)", f)[0])
            subject_id = f"sub{subject_num}"
            orig_label = session_labels[sess][subject_num - 1]
            bin_label = binary_label_neutral_happy_vs_sad_fear(orig_label)

            for trial_key in trial_keys:
                trial_data = data[trial_key]  # shape (channels, samples)

                step = int(EPOCH_SAMPLES * (1 - OVERLAP))
                n_windows = (trial_data.shape[1] - EPOCH_SAMPLES) // step + 1
                if n_windows <= 0:
                    print(f"  Trial too short for sliding windows: {trial_key} in {f}")
                    continue

                for w in range(n_windows):
                    feats = []
                    start = w * step
                    end = start + EPOCH_SAMPLES
                    for ch_idx in range(CHANNEL_COUNT):
                        epoch = trial_data[ch_idx, start:end]

                        bp = bandpower(epoch, SF_TARGET, BANDS)
                        feats.extend(bp)

                        feats.extend(band_ratios(bp))

                        feats.extend(hjorth_parameters(epoch))

                        feats.extend(statistical_features(epoch))

                    X.append(feats)
                    y.append(bin_label)
                    subject_ids.append(subject_id)

        except Exception as e:
            print(f"Error loading or processing {f}: {e}")
            continue

os.makedirs(RESULTS_PATH, exist_ok=True)
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

np.savez(OUTFILE, features=X, labels=y, subject_ids=subject_ids)

print(f"Saved cleaned features to: {OUTFILE}")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique subjects/files:", len(np.unique(subject_ids)))