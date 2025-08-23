import os
import re
import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt, resample, iirnotch
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

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='bandpass', fs=sf)
    return filtfilt(b, a, data)

def notch_filter(data, sf, freq=50.0, Q=30.0):
    b, a = iirnotch(w0=freq/(sf/2), Q=Q)
    return filtfilt(b, a, data)

def detrend(data):
    return data - np.polyval(np.polyfit(np.arange(len(data)), data, 1), np.arange(len(data)))

def remove_artifacts(data, threshold=150e-6):
    clean = np.copy(data)
    clean[np.abs(clean) > threshold] = 0
    return clean

def preprocess(data, orig_sf, target_sf=200):
    # Detrend to remove slow drifts
    data = detrend(data)

    # Notch filter for powerline noise (adjust freq 50 or 60 Hz as needed)
    data = notch_filter(data, orig_sf, freq=50.0, Q=30)

    # Bandpass filter EEG frequency band
    data = bandpass_filter(data, orig_sf, low=0.5, high=40)

    # Remove artifacts by amplitude thresholding
    data = remove_artifacts(data, threshold=150e-6)

    # Resample if original sampling freq differs from target
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))

    # Remove NaNs or infs
    data = np.nan_to_num(data)

    return data

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
OVERLAP = 0.5  # 50% overlap
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "seediv_anxiety_15fts_20s_overlap.npz")

session_labels = {
    "1": [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    "2": [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    "3": [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
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

                        # 5 band powers
                        bp = bandpower(epoch, SF_TARGET, BANDS)
                        feats.extend(bp)

                        # 4 band ratios
                        feats.extend(band_ratios(bp))

                        # 3 Hjorth parameters
                        feats.extend(hjorth_parameters(epoch))

                        # 3 statistical features
                        feats.extend(statistical_features(epoch))

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