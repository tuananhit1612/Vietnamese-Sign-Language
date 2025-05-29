import os
import pandas as pd
import numpy as np
input_dir = "D:/train_test/train_landmark_files"
output_dir = "D:/train_test/train_landmark_files"
os.makedirs(output_dir, exist_ok=True)

landmark_cols = [str(i) for i in range(126)]


def mirror_landmarks(df):
    data = df[landmark_cols].values.copy()
    num_frames = data.shape[0]
    reshaped = data.reshape(num_frames, 42, 3)

    mask = ~np.all(reshaped == 0, axis=2)
    reshaped[mask, 0] = 1.0 - reshaped[mask, 0]

    mirrored = reshaped.copy()
    mirrored[:, :21, :] = reshaped[:, 21:, :]
    mirrored[:, 21:, :] = reshaped[:, :21, :]

    flat = mirrored.reshape(num_frames, 126)
    new_df = pd.DataFrame(flat, columns=landmark_cols)

    left_flag = df["right"].iloc[0]
    right_flag = df["left"].iloc[0]

    new_df["left"] = left_flag
    new_df["right"] = right_flag

    return new_df




for fname in os.listdir(input_dir):
    if not fname.endswith(".parquet"):
        continue

    fpath = os.path.join(input_dir, fname)
    df = pd.read_parquet(fpath)

    has_left = df["left"].iloc[0]
    has_right = df["right"].iloc[0]

    if has_left and not has_right:
        mirrored_df = mirror_landmarks(df)
        new_fname = fname.replace(".parquet", "_mirrored_right.parquet")
        mirrored_df["left"] = 0
        mirrored_df["right"] = 1
        mirrored_df.to_parquet(os.path.join(output_dir, new_fname), index=False)
        print(f"Mirrored RIGHT: {new_fname}")

    elif has_right and not has_left:
        mirrored_df = mirror_landmarks(df)
        new_fname = fname.replace(".parquet", "_mirrored_left.parquet")
        mirrored_df["left"] = 1
        mirrored_df["right"] = 0
        mirrored_df.to_parquet(os.path.join(output_dir, new_fname), index=False)
        print(f"Mirrored LEFT: {new_fname}")

    elif has_left and has_right:
        mirrored_df = mirror_landmarks(df)
        new_fname = fname.replace(".parquet", "_mirrored_both.parquet")
        mirrored_df.to_parquet(os.path.join(output_dir, new_fname), index=False)
        print(f"Mirrored BOTH: {new_fname}")

    else:
        print(f"Skip (no hand): {fname}")

