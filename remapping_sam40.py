import os
import pandas as pd

# Mapping from CSV column indices to EEG channel names
channel_map = {
    '0': 'AF3', '1': 'F7', '2': 'F3', '3': 'FC5', '4': 'T7', '5': 'P7', '6': 'O1',
    '7': 'O2', '8': 'P8', '9': 'T8', '10': 'FC6', '11': 'F4', '12': 'F8', '13': 'AF4',
    '14': 'Fz', '15': 'Cz', '16': 'Pz', '17': 'Oz', '18': 'Fp1', '19': 'Fp2', '20': 'CP1',
    '21': 'CP2', '22': 'CP5', '23': 'CP6', '24': 'TP7', '25': 'TP8', '26': 'PO3', '27': 'PO4',
    '28': 'PO7', '29': 'PO8', '30': 'FT7', '31': 'FT8'
}

# Left ear channels only (for your project)
left_ear_channels = ['T7', 'FT7', 'TP7']

# Task folders (update names if needed)
tasks = ['StroopFolder', 'ArithmeticFolder', 'RelaxFolder']

base_dir = '../../data/sam40'

for task in tasks:
    task_folder = os.path.join(base_dir, task)
    if not os.path.exists(task_folder):
        print(f"Warning: Task folder not found: {task_folder}")
        continue

    # List subfolders inside task folder (e.g., Stroop, Arithmetic, Relax)
    subfolders = [f for f in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, f))]
    if not subfolders:
        # No inner subfolders, process CSVs directly in task folder
        subfolders = ['']

    for subfolder in subfolders:
        folder_path = os.path.join(task_folder, subfolder) if subfolder else task_folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"\nProcessing {len(files)} CSV files in {folder_path}...")

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            # Drop 'Unnamed: 0' if present
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])

            # Rename columns from indices to channel names (if possible)
            rename_dict = {col: channel_map[col] for col in df.columns if col in channel_map}
            df.rename(columns=rename_dict, inplace=True)

            # Filter only left ear channels (drop others)
            available_channels = [ch for ch in left_ear_channels if ch in df.columns]
            if not available_channels:
                print(f"  Warning: None of the left ear channels found in {filename}. Skipping file.")
                continue

            df = df[available_channels]

            # Sanity print
            print(f"  Columns kept (left ear only): {df.columns.tolist()}")
            print(df.head(3))

            # Overwrite CSV with cleaned and filtered data
            df.to_csv(file_path, index=False)

print("\nDone processing all files.")