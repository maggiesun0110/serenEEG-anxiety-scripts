import os
import re
import numpy as np
import scipy.io as sio
import mne

BASE_PATH = os.path.join("..", "..", "data", "SEED_IV", "eeg_raw_data")
OUT_PATH = os.path.join("..", "results", "seediv_ica_cleaned")
os.makedirs(OUT_PATH, exist_ok=True)

SESSIONS = ["1", "2", "3"]
CHANNEL_INDICES = [14, 23, 32]
CHANNEL_NAMES = [f"Ch{ch}" for ch in CHANNEL_INDICES]
SFREQ = 200

for sess in SESSIONS:
    sess_path = os.path.join(BASE_PATH, sess)
    if not os.path.exists(sess_path):
        print(f"Session folder {sess_path} not found, skipping.")
        continue

    mat_files = [f for f in os.listdir(sess_path) if f.endswith(".mat")]
    mat_files = sorted(mat_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for mat_file in mat_files:
        subject_num = int(re.findall(r'\d+', mat_file)[0])
        print(f"Processing Subject {subject_num}, Session {sess}")

        mat_path = os.path.join(sess_path, mat_file)
        mat_data = sio.loadmat(mat_path)

        all_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        trial_keys = sorted([k for k in all_keys if re.match(r'.*_eeg\d+$', k)],
                            key=lambda x: int(re.findall(r'\d+$', x)[0]))

        trials_selected = []
        trial_lengths = []

        for key in trial_keys:
            trial = mat_data[key]  # shape (channels, samples)
            selected = trial[CHANNEL_INDICES, :]
            trial_lengths.append(selected.shape[1])
            trials_selected.append(selected)

        concat_data = np.concatenate(trials_selected, axis=1)

        info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=SFREQ, ch_types=['eeg']*len(CHANNEL_NAMES))
        raw = mne.io.RawArray(concat_data, info)

        # Filtering
        raw.notch_filter(freqs=60)
        raw.filter(l_freq=1, h_freq=75)

        # ICA
        ica = mne.preprocessing.ICA(n_components=3, random_state=97, max_iter='auto')
        ica.fit(raw)

        # Only detect muscle artifacts (skip EOG)
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw, threshold=4.0)
        ica.exclude = muscle_indices
        print(f"Automatically excluding muscle ICA components: {ica.exclude}")

        raw_clean = ica.apply(raw.copy())

        cleaned_trials = []
        idx = 0
        for length in trial_lengths:
            trial_data = raw_clean.get_data()[:, idx:idx+length]
            cleaned_trials.append(trial_data)
            idx += length

        # Save cleaned trials with separate keys to avoid inhomogeneous array error
        save_path = os.path.join(OUT_PATH, f"sub{subject_num}_sess{sess}_cleaned_trials.npz")
        np.savez(save_path, **{f"trial_{i}": cleaned_trials[i] for i in range(len(cleaned_trials))})
        print(f"Saved cleaned trials to {save_path}")