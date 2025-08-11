#15 --> FT7
#24 --> R7
#33 --> T7
#Label:
# The labels of the three sessions for the same subjects are as follows,
# session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
# session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
# session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];

# The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.
#each participant has 3 sessions, each 24 trials (5s, 2min, 45s)
#-------------------------------

import scipy.io as sio
import os

# Path to the .mat file
mat_path = os.path.join("..", "..", "data", "SEED_IV", "eeg_raw_data", "1", "2_20150915.mat")

# Load the .mat file
mat_data = sio.loadmat(mat_path)

# Print all keys (variables) inside the .mat file
print("Keys in MAT file:", mat_data.keys())

# Loop through variables (skip metadata like __header__, __version__, __globals__)
for key in mat_data:
    if not key.startswith("__"):
        print(f"\nVariable: {key}")
        print("Type:", type(mat_data[key]))
        print("Shape:", getattr(mat_data[key], "shape", "No shape"))
        print(mat_data[key])  # This will print the data (may be large)