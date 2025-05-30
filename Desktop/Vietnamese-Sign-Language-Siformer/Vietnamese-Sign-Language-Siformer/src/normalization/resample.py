import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import interp1d

excel_path = "C:/Users/ADMIN/Desktop/Dataset/train.xlsx"
landmark_folder = "C:/Users/ADMIN/Desktop/Dataset/train_landmark_files/augmented_combinations"
output_pkl = "C:/Users/ADMIN/Desktop/Dataset/resample_data.pkl"

df_label = pd.read_excel(excel_path)

df_label["file_id"] = df_label["file_id"].astype(str).str.strip()
df_label["gloss"] = df_label["gloss"].astype(str).str.strip()

unique_glosses = sorted(df_label["gloss"].unique())
id2label = dict(zip(df_label["file_id"], df_label["gloss"]))
gloss2id = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

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
    if not fname.endswith(".parquet"):
        continue

    full_path = os.path.join(landmark_folder, fname)
    file_id = fname.split("_")[0].strip()

    if file_id not in id2label:
        print(f" Không tìm thấy nhãn cho: {fname} → {file_id}")
        continue

    try:
        df = pd.read_parquet(full_path)

        df = df.drop(columns=['frame'], errors='ignore')

        df_fixed = resample_sequence(df, target_frames=50)

        if df_fixed.isnull().values.any():
            print(f" {fname}: có NaN sau resample → bỏ qua")
            continue

        X.append(df_fixed.values.astype(np.float32))
        gloss = id2label[file_id]
        y.append(gloss2id[gloss])

        print(f"{idx} - {fname} ")

    except Exception as e:
        print(f" {fname}: lỗi - {e}")

with open(output_pkl, "wb") as f:
    pickle.dump({"X": X, "y": y, "label_map": gloss2id}, f)

print(f"\n Hoàn tất! Tổng sample: {len(X)}, Tổng nhãn: {len(y)}, Số lớp: {len(gloss2id)}")
print(f" Dữ liệu đã lưu tại: {output_pkl}")
