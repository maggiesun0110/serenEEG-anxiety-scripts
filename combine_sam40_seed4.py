import numpy as np
import os

# === Paths to your aligned feature files ===
seediv_file = "../results/seediv_anxiety_24_ica_cleaned.npz"
sam40_file = "../results/sam40_anxiety_24fts_20s_overlap.npz"
combined_file = "../results/seediv_sam40_combined.npz"

# === Load SEED-IV ===
seediv_data = np.load(seediv_file)
X_seediv = seediv_data["features"]
y_seediv = seediv_data["labels"]
subj_seediv = seediv_data["subject_ids"]

print("SEED-IV:", X_seediv.shape, y_seediv.shape, len(np.unique(subj_seediv)))

# === Load SAM40 ===
sam40_data = np.load(sam40_file)
X_sam40 = sam40_data["features"]
y_sam40 = sam40_data["labels"]
subj_sam40 = sam40_data["subject_ids"]

print("SAM40:", X_sam40.shape, y_sam40.shape, len(np.unique(subj_sam40)))

# === Sanity check: features must match ===
if X_seediv.shape[1] != X_sam40.shape[1]:
    raise ValueError("Feature mismatch! Both datasets must have same number of features per sample.")

# === Combine datasets ===
X_combined = np.vstack([X_seediv, X_sam40])
y_combined = np.hstack([y_seediv, y_sam40])
subject_ids_combined = np.hstack([subj_seediv, subj_sam40])

print("Combined dataset:", X_combined.shape, y_combined.shape, len(np.unique(subject_ids_combined)))

# === Save combined dataset ===
os.makedirs(os.path.dirname(combined_file), exist_ok=True)
np.savez(combined_file, features=X_combined, labels=y_combined, subject_ids=subject_ids_combined)

print(f"Saved combined dataset to: {combined_file}")

combined_file = "../results/seediv_sam40_combined.npz"
data = np.load(combined_file)
X = data["features"]

channels = ["FT7", "T7", "TP7"]
features_per_channel = [
    "PSD_delta", "PSD_theta", "PSD_alpha", "PSD_beta", "PSD_gamma",
    "DE_delta", "DE_theta", "DE_alpha", "DE_beta", "DE_gamma",
    "Ratio_delta_theta", "Ratio_delta_alpha", "Ratio_alpha_beta", "Ratio_theta_alpha",
    "Hjorth_activity", "Hjorth_mobility", "Hjorth_complexity",
    "Stat_mean", "Stat_std", "Stat_skew", "Stat_kurtosis",
    "Spectral_entropy", "Zero_crossing_rate", "Line_length"
]

# Print column ranges per channel
col_idx = 0
for ch in channels:
    print(f"\nChannel: {ch}")
    for feat in features_per_channel:
        print(f"Column {col_idx}: {feat}")
        col_idx += 1

print("\nTotal features:", X.shape[1])