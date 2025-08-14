import os
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, resample
from scipy.integrate import simpson
from scipy.stats import skew, kurtosis

# === Feature functions ===
def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
    return [simpson(psd[(freqs >= low) & (freqs <= high)],
                    freqs[(freqs >= low) & (freqs <= high)])
            for low, high in bands]

def differential_entropy(powers):
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
    return -np.sum(psd_norm * np.log(psd_norm))

def zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum() / len(data)

def line_length(data):
    return np.sum(np.abs(np.diff(data))) / len(data)

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='bandpass', fs=sf)
    return filtfilt(b, a, data)

def preprocess(data, orig_sf, target_sf=200):
    data = bandpass_filter(data, orig_sf)
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))
    return data

def extract_subject_id(filename):
    # Example: "Arithmetic_sub_25.csv" -> "sub_25"
    name = filename.replace(".csv", "")
    parts = name.split('_')
    return "_".join(parts[1:])  # Join from second part onward

# === Config ===
BASE_PATH = os.path.join("..", "..", "data", "sam40")
RESULTS_PATH = os.path.join("..", "results")
CHANNELS = ["FT7", "T7", "TP7"]
EPOCH_LEN = 20
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "sam40_anxiety_24fts_20s_overlap.npz")

label_map = {
    "ArithmeticFolder": 1,
    "StroopFolder": 1,
    "RelaxFolder": 0
}

X, y, subject_ids = [], [], []

# === Main loop ===
for folder_name, label in label_map.items():
    folder_path = os.path.join(BASE_PATH, folder_name, folder_name.replace("Folder", ""))
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    print(f"Processing folder: {folder_name} with label {label}")
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for file in files:
        filepath = os.path.join(folder_path, file)
        print(f"  Processing file: {file}")
        try:
            df = pd.read_csv(filepath)

            if df.shape[0] < df.shape[1]:
                df = df.T
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])

            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')

            if not set(CHANNELS).issubset(df.columns):
                print(f"    Missing required channels in {file}")
                continue

            data_ch = df[CHANNELS].T.values
            orig_sf = 200  # adjust if needed

            processed_ch = [preprocess(data_ch[i], orig_sf, SF_TARGET) for i in range(len(CHANNELS))]

            step = int(EPOCH_SAMPLES * (1 - OVERLAP))
            n_windows = (len(processed_ch[0]) - EPOCH_SAMPLES) // step + 1
            if n_windows <= 0:
                print(f"    File too short for sliding windows: {file}")
                continue

            for w in range(n_windows):
                feats = []
                start = w * step
                end = start + EPOCH_SAMPLES

                for ch_data in processed_ch:
                    epoch = ch_data[start:end]
                    if np.any(np.isnan(epoch)) or np.any(np.isinf(epoch)):
                        epoch = np.nan_to_num(epoch)

                    # PSD band powers
                    bp = bandpower(epoch, SF_TARGET, BANDS)
                    feats.extend(bp)

                    # Differential entropy
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
                y.append(label)
                subject_ids.append(extract_subject_id(file))

        except Exception as e:
            print(f"    Error processing {file}: {e}")
            continue

# === Save ===
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE, features=X, labels=y, subject_ids=subject_ids)

print(f"Saved features to: {OUTFILE}")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique subjects/files:", len(np.unique(subject_ids)))