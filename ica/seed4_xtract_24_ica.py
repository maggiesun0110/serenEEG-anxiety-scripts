import os
import re
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.integrate import simpson

# === Feature functions ===

def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
    powers = [
        simpson(psd[(freqs >= low) & (freqs <= high)],
                freqs[(freqs >= low) & (freqs <= high)])
        for low, high in bands
    ]
    return powers

def differential_entropy(powers):
    # DE = log(PSD power)
    return [np.log(p + 1e-10) for p in powers]

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
    activity = np.var(data)
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    return [activity, mobility, complexity]

def statistical_features(data):
    return [np.mean(data), np.std(data), skew(data), kurtosis(data)]

def spectral_entropy(data, sf):
    freqs, psd = welch(data, sf, nperseg=sf*2)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm + 1e-12  # avoid log(0)
    se = -np.sum(psd_norm * np.log(psd_norm))
    return se

def zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum() / len(data)

def line_length(data):
    return np.sum(np.abs(np.diff(data))) / len(data)

# === Config ===

BASE_PATH = os.path.join("..", "results", "seediv_ica_cleaned") 
RESULTS_PATH = os.path.join("..", "results")
OUTFILE = os.path.join(RESULTS_PATH, "seediv_anxiety_24_ica_cleaned.npz")

CHANNELS = 3  # Number of channels cleaned and saved per trial
EPOCH_LEN = 20  # seconds
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5  # 50% overlap

# Frequency bands for PSD and DE
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]

X, y, subject_ids = [], [], []

# Your session labels for binary classification (adjust if needed)
session_labels = {
    "1": [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    "2": [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    "3": [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

def binary_label_neutral_happy_vs_sad_fear(label):
    return 0 if label in [0, 3] else 1

sessions = ["1", "2", "3"]

for session_label in sessions:
    session_path = os.path.join(BASE_PATH)
    files = [f for f in os.listdir(session_path) if f.startswith(f"sub") and f"sess{session_label}" in f and f.endswith(".npz")]

    print(f"Processing session {session_label}, {len(files)} cleaned files found")

    for file in files:
        filepath = os.path.join(session_path, file)
        print(f"Loading cleaned data file: {filepath}")

        try:
            data = np.load(filepath)
            cleaned_trials = [data[f"trial_{i}"] for i in range(len(data.files))]  # list of np arrays (channels x samples)

            subject_num = int(re.findall(r'\d+', file)[0])
            subject_id = f"sub{subject_num}"
            orig_label = session_labels[session_label][subject_num - 1]
            bin_label = binary_label_neutral_happy_vs_sad_fear(orig_label)

            for trial_data in cleaned_trials:
                step = int(EPOCH_SAMPLES * (1 - OVERLAP))
                n_windows = (trial_data.shape[1] - EPOCH_SAMPLES) // step + 1
                if n_windows <= 0:
                    print(f"  Trial too short for sliding windows in {file}")
                    continue

                for w in range(n_windows):
                    feats = []
                    start = w * step
                    end = start + EPOCH_SAMPLES

                    for ch_idx in range(CHANNELS):
                        epoch = trial_data[ch_idx, start:end]

                        # PSD band powers
                        bp = bandpower(epoch, SF_TARGET, BANDS)
                        feats.extend(bp)

                        # Differential entropy (log band power)
                        feats.extend(differential_entropy(bp))

                        # Band ratios
                        feats.extend(band_ratios(bp))

                        # Hjorth parameters
                        feats.extend(hjorth_parameters(epoch))

                        # Statistical features
                        feats.extend(statistical_features(epoch))

                        # Spectral entropy
                        feats.append(spectral_entropy(epoch, SF_TARGET))

                        # Zero crossing rate
                        feats.append(zero_crossing_rate(epoch))

                        # Line length
                        feats.append(line_length(epoch))

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

print(f"Saved rich features to: {OUTFILE}")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique subjects/files:", len(np.unique(subject_ids)))