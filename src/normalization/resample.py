import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import interp1d

excel_path = "D:/train_test/train.xlsx"
landmark_folder = "D:/train_test/train_landmark_files/augmented_combinations"
output_pkl = "D:/train_test/resample_data.pkl"

df_label = pd.read_excel(excel_path)
unique_glosses = sorted(df_label["gloss"].unique())
label_map = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
id2label = dict(zip(df_label['file_id'], df_label['gloss']))

def resample_sequence(df, target_frames=50):
    old_len = len(df)
    new_idx = np.linspace(0, old_len - 1, target_frames)

    resampled_cols = []
    for col in df.columns:
        f = interp1d(np.arange(old_len), df[col], kind='linear', fill_value="extrapolate")
        resampled_cols.append(pd.Series(f(new_idx), name=col))

    df_resampled = pd.concat(resampled_cols, axis=1)

    if len(df_resampled) < target_frames:
        padding_len = target_frames - len(df_resampled)
        padding = pd.DataFrame(np.nan, index=range(padding_len), columns=df_resampled.columns)
        df_resampled = pd.concat([df_resampled, padding], ignore_index=True)

    return df_resampled

X = []
y = []

for idx, fname in enumerate(os.listdir(landmark_folder)):
    if not fname.endswith('.parquet'):
        continue

    full_path = os.path.join(landmark_folder, fname)
    file_id = fname.rsplit('_', 5)[0].replace('.parquet', '')

    if file_id not in id2label:
        print(f"Không tìm thấy nhãn cho: {file_id}")
        continue

    try:
        df = pd.read_parquet(full_path)
        df_fixed = resample_sequence(df, target_frames=50)

        if 'frame' in df_fixed.columns:
            df_fixed = df_fixed.drop(columns=['frame'])

        X.append(df_fixed.values.astype(np.float32))
        y.append(label_map[id2label[file_id]])  # Convert gloss → int label

        print(f"{idx} - {fname}:")

    except Exception as e:
        print(f"{fname}:lỗi - {e}")

with open(output_pkl, "wb") as f:
    pickle.dump({'X': X, 'y': y, 'label_map': label_map}, f)

