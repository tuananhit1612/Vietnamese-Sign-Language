import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

input_folder = Path("C:/Users/ADMIN/Desktop/Dataset/train_landmark_files/augmented_combinations")
label_file = "C:/Users/ADMIN/Desktop/Dataset/train.xlsx"
output_pkl = "C:/Users/ADMIN/Desktop/Dataset/packed_dataset.pkl"

df_labels = pd.read_excel(label_file)
unique_glosses = sorted(df_labels["gloss"].astype(str).unique())
gloss2id = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

fileid_to_labelid = {}
for _, row in df_labels.iterrows():
    fileid = str(row["file_id"])
    gloss = str(row["gloss"])

    fileid_to_labelid[fileid] = gloss2id[gloss]

X = []
y = []

for parquet_path in input_folder.glob("*.parquet"):
    filename = parquet_path.stem

    file_id = filename.split("_")[0]

    if file_id not in fileid_to_labelid:
        print(f"Skip {filename}: label not found")
        continue

    df = pd.read_parquet(parquet_path)

    if df.shape[1] < 126:
        print(f"Skip {filename}: less than 126 features")
    X.append(df.iloc[:, :126].values.astype(np.float32))
    y.append(fileid_to_labelid[file_id])

print(f"Loaded {len(X)} samples")

with open(output_pkl, "wb") as f:
    pickle.dump({"X": X, "y": y, "label_map": gloss2id}, f)

print(f"Packed dataset saved to: {output_pkl}")
