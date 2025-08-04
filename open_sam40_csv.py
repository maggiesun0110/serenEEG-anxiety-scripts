import pandas as pd
import os

file_path = '../../data/sam40/RelaxFolder/Relax/Relax_sub_1.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

print("loaded: ", file_path)
print("shape: ", df.shape)
print("columns: ", df.columns.tolist())
