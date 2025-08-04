import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from antropy import hjorth_params
from tqdm import tqdm

# === Feature Extraction Functions ===
def bandpass_filter(signal, fs=128, lowcut=1, highcut=40, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(data, fs=128):
    features = []

    # Apply bandpass filter channel-wise
    data = np.array([bandpass_filter(chan, fs) for chan in data])

    for chan in data:
        # PSD features in standard bands
        freqs, psd = welch(chan, fs=fs, nperseg=fs*2)
        psd_bands = [
            np.mean(psd[(freqs >= 1) & (freqs < 4)]),   # delta
            np.mean(psd[(freqs >= 4) & (freqs < 8)]),   # theta
            np.mean(psd[(freqs >= 8) & (freqs < 13)]),  # alpha
            np.mean(psd[(freqs >= 13) & (freqs < 30)]), # beta
            np.mean(psd[(freqs >= 30) & (freqs <= 40)]) # gamma
        ]
        # Hjorth parameters: activity, mobility, complexity
        hj = hjorth_params(chan)

        features.extend(psd_bands + list(hj))

    return np.array(features)

# === Main Extraction Loop ===
def process_task(task_name, label, root_dir="../../data/sam40"):
    task_folder = os.path.join(root_dir, f"{task_name}Folder", task_name)
    print(f"Processing task '{task_name}' in folder: {task_folder}")
    assert os.path.exists(task_folder), f"❌ Path not found: {task_folder}"

    X = []
    y = []

    files = [f for f in os.listdir(task_folder) if f.endswith(".csv")]
    print(f"Found {len(files)} CSV files.")

    for file in tqdm(files, desc=f"Processing {task_name}"):
        filepath = os.path.join(task_folder, file)
        try:
            df = pd.read_csv(filepath)

            # Check shape and transpose if channels are in columns (common)
            if df.shape[0] < df.shape[1]:
                data = df.values.T  # channels x time
            else:
                data = df.values    # assume already channels x time
            
            features = extract_features(data)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)
    save_path = f"features_{task_name.lower()}.npz"
    np.savez(save_path, X=X, y=y)
    print(f"✅ Saved features to {save_path}")

# === Run for Each Task ===
if __name__ == "__main__":
    tasks = [("Relax", 0), ("Arithmetic", 1), ("Stroop", 1)] 
    for task, label in tasks:
        process_task(task, label)