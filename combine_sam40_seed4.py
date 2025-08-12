import os
import numpy as np

seed_path = os.path.join("..", "results", "seediv_anxiety_8fts_20s_overlap.npz")
sam_path = os.path.join("..", "results", "sam40_anxiety_8fts_20s_overlap.npz")
combined_path = os.path.join("..", "results", "combined_sam40_seed4_base8.npz")

# Load the data
seed_data = np.load(seed_path)
sam_data = np.load(sam_path)

X_seed, y_seed, ids_seed = seed_data["features"], seed_data["labels"], seed_data["subject_ids"]
X_sam, y_sam, ids_sam = sam_data["features"], sam_data["labels"], sam_data["subject_ids"]

print(f"Seed data shape: {X_seed.shape}, Labels: {y_seed.shape}, IDs: {ids_seed.shape}")
print(f"SAM data shape: {X_sam.shape}, Labels: {y_sam.shape}, IDs: {ids_sam.shape}")

# Concatenate features vertically (axis=0)
X_combined = np.vstack([X_seed, X_sam])

# Concatenate labels and subject ids vertically (axis=0)
y_combined = np.concatenate([y_seed, y_sam])
ids_combined = np.concatenate([ids_seed, ids_sam])

print("Combined features shape:", X_combined.shape)
print("Combined labels shape:", y_combined.shape)
print("Combined unique subjects:", len(np.unique(ids_combined)))

print("Seed feature shape:", X_seed.shape)
print("SAM feature shape:", X_sam.shape)
print("Are feature dims equal?", X_seed.shape[1] == X_sam.shape[1])
print("Seed labels unique and counts:", np.unique(y_seed, return_counts=True))
print("SAM labels unique and counts:", np.unique(y_sam, return_counts=True))

# Save combined data
np.savez(combined_path, features=X_combined, labels=y_combined, subject_ids=ids_combined)
print(f"Combined data saved to: {combined_path}")