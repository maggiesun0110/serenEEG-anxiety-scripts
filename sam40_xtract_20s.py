import os
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, resample
from scipy.integrate import simpson

# === Feature Functions ===
def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
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
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-6) if var_d1 > 0 else 0
    return [var_zero, mobility, complexity]

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='band', fs=sf)
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
CHANNELS = ["T7", "TP7", "FT7"]
EPOCH_LEN = 20  # seconds
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OVERLAP = 0.5  # 50% overlap
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "sam40_anxiety_8fts_20s_overlap.npz")

# Label mapping
label_map = {
    "ArithmeticFolder": 1,
    "StroopFolder": 1,
    "RelaxFolder": 0
}

X, y, subject_ids = [], [], []

# === Main Loop ===
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

            # If rows < columns, transpose and fix headers
            if df.shape[0] < df.shape[1]:
                df = df.T
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(axis=0, how='any')

            if not set(CHANNELS).issubset(df.columns):
                print(f"    Missing required channels in {file}")
                continue

            data_ch = df[CHANNELS].T.values
            orig_sf = 200  # fixed as per dataset info

            # Preprocess channels (filter + resample)
            processed_ch = [preprocess(data_ch[i], orig_sf, SF_TARGET) for i in range(len(CHANNELS))]

            # Sliding windows with 50% overlap
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
                    feats.extend(bandpower(epoch, SF_TARGET, BANDS))
                    feats.extend(hjorth_parameters(epoch))

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