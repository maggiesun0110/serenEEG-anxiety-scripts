import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")

# === Paths ===
seed_path = os.path.join("..", "results", "seediv_anxiety_8fts_20s_overlap.npz")
sam_path = os.path.join("..", "results", "sam40_anxiety_8fts_20s_overlap.npz")
combined_path = os.path.join("..", "results", "combined_sam40_seed4_base8.npz")  

# === Load data ===
seed_data = np.load(seed_path)
sam_data = np.load(sam_path)

X_seed, y_seed, ids_seed = seed_data["features"], seed_data["labels"], seed_data["subject_ids"]
X_sam, y_sam, ids_sam = sam_data["features"], sam_data["labels"], sam_data["subject_ids"]

print(f"Seed data shape: {X_seed.shape}, Labels: {y_seed.shape}, IDs: {ids_seed.shape}")
print(f"SAM data shape: {X_sam.shape}, Labels: {y_sam.shape}, IDs: {ids_sam.shape}")

# === Combine datasets ===
X_combined = np.vstack([X_seed, X_sam])
y_combined = np.concatenate([y_seed, y_sam])
ids_combined = np.concatenate([ids_seed, ids_sam])

print("\n=== Combined Data Info ===")
print("Features shape:", X_combined.shape)
print("Labels shape:", y_combined.shape)
print("Unique subjects:", len(np.unique(ids_combined)))
print("Feature dims equal?", X_seed.shape[1] == X_sam.shape[1])

# === Label distribution check ===
class_counts = Counter(y_combined)
print("\nClass distribution:", class_counts)

# === Balance adjustment ===
max_class = max(class_counts.values())
min_class = min(class_counts.values())
imbalance_ratio = max_class / min_class

if imbalance_ratio > 1.5:  # Arbitrary threshold for imbalance
    print(f"Detected imbalance ratio: {imbalance_ratio:.2f}")
    # Choose oversampling or undersampling here
    oversample = True  # Change to False for undersampling
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_combined, y_combined = ros.fit_resample(X_combined, y_combined)
        print("Applied oversampling. New distribution:", Counter(y_combined))
        # Resample subject IDs as well
        ids_combined = np.array(ros.fit_resample(ids_combined.reshape(-1, 1), y_combined)[0]).flatten()
    else:
        rus = RandomUnderSampler(random_state=42)
        X_combined, y_combined = rus.fit_resample(X_combined, y_combined)
        print("Applied undersampling. New distribution:", Counter(y_combined))
        ids_combined = np.array(rus.fit_resample(ids_combined.reshape(-1, 1), y_combined)[0]).flatten()
else:
    print("No significant imbalance detected. No resampling applied.")

# === Scale features ===
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
print("Features scaled (StandardScaler).")

# === Save combined dataset ===
np.savez(combined_path, features=X_combined, labels=y_combined, subject_ids=ids_combined)
print(f"Saved combined, scaled, balanced data to: {combined_path}")