import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")

# Paths to your pre-extracted feature files
seed_file = "../results/seediv_anxiety_15fts_20s_overlap.npz"
sam_file = "../results/sam40_anxiety_15fts_20s_overlap.npz"
combined_outfile = "../results/combined_seed_sam_15fts_20s_overlap.npz"

# Load Seed-IV features
seed_data = np.load(seed_file)
X_seed = seed_data["features"]
y_seed = seed_data["labels"]
subj_seed = seed_data["subject_ids"]

# Load SAM40 features
sam_data = np.load(sam_file)
X_sam = sam_data["features"]
y_sam = sam_data["labels"]
subj_sam = sam_data["subject_ids"]

# Confirm feature dims match
assert X_seed.shape[1] == X_sam.shape[1], "Feature dimension mismatch!"

# Combine
X_combined = np.vstack([X_seed, X_sam])
y_combined = np.hstack([y_seed, y_sam])
subj_combined = np.hstack([subj_seed, subj_sam])

# Basic sanity printouts
print(f"Seed features: {X_seed.shape}, labels: {y_seed.shape}, unique subjects: {len(np.unique(subj_seed))}")
print(f"SAM features: {X_sam.shape}, labels: {y_sam.shape}, unique subjects: {len(np.unique(subj_sam))}")
print(f"Combined features: {X_combined.shape}, labels: {y_combined.shape}, unique subjects: {len(np.unique(subj_combined))}")
print("Combined label distribution:", Counter(y_combined))

# === Label distribution check and balance adjustment ===
class_counts = Counter(y_combined)
max_class = max(class_counts.values())
min_class = min(class_counts.values())
imbalance_ratio = max_class / min_class

if imbalance_ratio > 1.5:
    print(f"Detected imbalance ratio: {imbalance_ratio:.2f}")
    oversample = True  # Change to False for undersampling
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_combined, y_combined = ros.fit_resample(X_combined, y_combined)
        print("Applied oversampling. New distribution:", Counter(y_combined))
        subj_combined = np.array(ros.fit_resample(subj_combined.reshape(-1,1), y_combined)[0]).flatten()
    else:
        rus = RandomUnderSampler(random_state=42)
        X_combined, y_combined = rus.fit_resample(X_combined, y_combined)
        print("Applied undersampling. New distribution:", Counter(y_combined))
        subj_combined = np.array(rus.fit_resample(subj_combined.reshape(-1,1), y_combined)[0]).flatten()
else:
    print("No significant imbalance detected. No resampling applied.")

# === Scale features ===
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
print("Features scaled (StandardScaler).")

# Save combined data
np.savez(combined_outfile, features=X_combined, labels=y_combined, subject_ids=subj_combined)
print(f"Saved combined data to {combined_outfile}")