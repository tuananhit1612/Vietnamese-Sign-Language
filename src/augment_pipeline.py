import pandas as pd
import numpy as np
import random

# tinh tien
def add_offset(df, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(-0.05, 0.05)):
    df_offset = df.copy()
    dx, dy, dz = np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)
    for hand in ['left', 'right']:
        for i in range(21):
            df_offset[f'x_{hand}_hand_{i}'] = (df_offset[f'x_{hand}_hand_{i}'] + dx).clip(0, 1)
            df_offset[f'y_{hand}_hand_{i}'] = (df_offset[f'y_{hand}_hand_{i}'] + dy).clip(0, 1)
            df_offset[f'z_{hand}_hand_{i}'] = (df_offset[f'z_{hand}_hand_{i}'] + dz).clip(-1, 1)
    return df_offset

# them nhieu~
def add_noise_to_landmarks(df, std=0.01, clip=True):
    df_noisy = df.copy()
    for hand in ['left', 'right']:
        for i in range(21):
            for axis in ['x', 'y', 'z']:
                col = f'{axis}_{hand}_hand_{i}'
                noise = np.random.normal(0, std, size=df_noisy.shape[0])
                df_noisy[col] += noise
                if clip:
                    if axis in ['x', 'y']:
                        df_noisy[col] = df_noisy[col].clip(0, 1)
                    else:
                        df_noisy[col] = df_noisy[col].clip(-1, 1)
    return df_noisy

# phong to/nho
def scale_landmarks(df, scale_factor=1.1, center_mode='mean'):
    df_scaled = df.copy()
    for hand in ['left', 'right']:
        landmarks = np.array([[df[f'x_{hand}_hand_{i}'],
                               df[f'y_{hand}_hand_{i}'],
                               df[f'z_{hand}_hand_{i}']] for i in range(21)])
        center = landmarks.mean(axis=0) if center_mode == 'mean' else landmarks[0]
        scaled = (landmarks - center) * scale_factor + center
        for i in range(21):
            df_scaled[f'x_{hand}_hand_{i}'] = np.clip(scaled[i, 0], 0, 1)
            df_scaled[f'y_{hand}_hand_{i}'] = np.clip(scaled[i, 1], 0, 1)
            df_scaled[f'z_{hand}_hand_{i}'] = np.clip(scaled[i, 2], -1, 1)
    return df_scaled

# lat tay
def move_and_flip_hand(df):
    updated_rows = []
    for _, row in df.iterrows():
        new_row = row.copy()
        left_present = any(not np.isclose(row[f'x_left_hand_{i}'], 0.0) for i in range(21))
        right_present = any(not np.isclose(row[f'x_right_hand_{i}'], 0.0) for i in range(21))
        if left_present and not right_present:
            for i in range(21):
                new_row[f'x_right_hand_{i}'] = 1.0 - row[f'x_left_hand_{i}']
                new_row[f'y_right_hand_{i}'] = row[f'y_left_hand_{i}']
                new_row[f'z_right_hand_{i}'] = row[f'z_left_hand_{i}']
                new_row[f'x_left_hand_{i}'] = new_row[f'y_left_hand_{i}'] = new_row[f'z_left_hand_{i}'] = 0.0
        elif right_present and not left_present:
            for i in range(21):
                new_row[f'x_left_hand_{i}'] = 1.0 - row[f'x_right_hand_{i}']
                new_row[f'y_left_hand_{i}'] = row[f'y_right_hand_{i}']
                new_row[f'z_left_hand_{i}'] = row[f'z_right_hand_{i}']
                new_row[f'x_right_hand_{i}'] = new_row[f'y_right_hand_{i}'] = new_row[f'z_right_hand_{i}'] = 0.0
        updated_rows.append(new_row)
    return pd.DataFrame(updated_rows)

# them nhieu 2
def temporal_jitter(df, drop_prob=0.1, duplicate_prob=0.1, shuffle_window=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    frames = df.copy()
    if drop_prob > 0:
        mask = np.random.rand(len(frames)) > drop_prob
        frames = frames[mask].reset_index(drop=True)
    if duplicate_prob > 0:
        dup = frames[np.random.rand(len(frames)) < duplicate_prob]
        frames = pd.concat([frames, dup], ignore_index=True).sort_values(by='frame').reset_index(drop=True)
    if shuffle_window > 1:
        chunks = []
        for i in range(0, len(frames), shuffle_window):
            chunk = frames.iloc[i:i + shuffle_window].sample(frac=1).reset_index(drop=True)
            chunks.append(chunk)
        frames = pd.concat(chunks, ignore_index=True)
    frames['frame'] = range(len(frames))
    return frames

def augment_pipeline(df, apply):
    if apply.get('mirror'):
        df = move_and_flip_hand(df)
    if apply.get('offset'):
        df = add_offset(df, x_range=(-0.05, 0.05), y_range=(-0.03, 0.03), z_range=(-0.02, 0.02))
    if apply.get('noise'):
        std = np.random.uniform(0.0001, 0.005)
        df = add_noise_to_landmarks(df, std=std)
    if apply.get('scale'):
        factor = np.random.uniform(0.7, 2.5)
        df = scale_landmarks(df, scale_factor=factor)
    if apply.get('jitter'):
        df = temporal_jitter(df, drop_prob=0.1, duplicate_prob=0.1, shuffle_window=3)
    return df
