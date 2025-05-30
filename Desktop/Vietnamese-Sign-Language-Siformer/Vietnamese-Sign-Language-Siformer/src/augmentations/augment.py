import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from pathlib import Path
import copy


def add_offset(data, dx, dy, dz):
    mask = ~np.all(data == 0, axis=2)

    data[mask] += np.array([dx, dy, dz])
    return data

def rotate_x(data, angle_deg=None):
    if angle_deg is None:
        angle_deg = np.random.uniform(-20, 20)
    angle = np.radians(angle_deg)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    R = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

    mask = ~np.all(data == 0, axis=2)  # [T, 42]
    rotated = data.copy()
    rotated[mask] = np.dot(data[mask], R.T)
    return rotated
def rotate_y(data, angle_deg=None):
    if angle_deg is None:
        angle_deg = np.random.uniform(-20, 20)
    angle = np.radians(angle_deg)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    mask = ~np.all(data == 0, axis=2)
    rotated = data.copy()
    rotated[mask] = np.dot(data[mask], R.T)
    return rotated
def rotate_z(data, angle_deg=None):
    if angle_deg is None:
        angle_deg = np.random.uniform(-15, 15)
    angle = np.radians(angle_deg)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    mask = ~np.all(data == 0, axis=2)
    rotated = data.copy()
    rotated[mask] = np.dot(data[mask], R.T)
    return rotated

def add_noise(data, std=0.001):
    mask = ~np.all(data == 0, axis=2)  # [T, 42]
    noise = np.random.normal(0, std, size=data.shape)
    data[mask] += noise[mask]
    return data

def scale(data, factor):
    mask = ~np.all(data == 0, axis=2)
    data[mask] *= factor
    return data


def temporal_jitter(data, drop_prob=0.1, duplicate_prob=0.1):
    frames = list(data)
    new_frames = []
    for f in frames:
        if np.random.rand() < drop_prob:
            continue
        new_frames.append(f)
        if np.random.rand() < duplicate_prob:
            new_frames.append(f.copy())
    if len(new_frames) < 2:
        return np.array(frames)
    return np.array(new_frames)

INPUT_DIR = Path("C:/Users/ADMIN/Desktop/Dataset/train_landmark_files")
OUTPUT_DIR = Path("C:/Users/ADMIN/Desktop/Dataset/train_landmark_files/augmented_combinations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

techniques = ['offset', 'noise', 'scale', 'jitter','rotate_x']
combinations = list(product([False, True], repeat=len(techniques)))[1:]
valid_combinations = [dict(zip(techniques, combo)) for combo in combinations]


def apply_augmentations(data, combo):
    aug = data.copy()

    if combo.get('offset'):
        dx, dy, dz = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.03, 0.03), np.random.uniform(-0.02, 0.02)
        aug = add_offset(aug, dx, dy, dz)

    if combo.get('noise'):
        std = np.random.uniform(0.0001, 0.002)
        aug = add_noise(aug, std)

    if combo.get('scale'):
        factor = np.random.uniform(0.9, 1.5)
        aug = scale(aug, factor)

    if combo.get('jitter'):
        aug = temporal_jitter(aug)
    # if combo.get('rotate_x'):
    #     aug = rotate_x(aug)
    # if combo.get('rotate_y'):
    #     aug = rotate_y(aug)
    if combo.get('rotate_z'):
        aug = rotate_z(aug)
    return aug

for file in tqdm(list(INPUT_DIR.glob("*.parquet"))):
    file_id = file.stem
    df = pd.read_parquet(file)

    if df.shape[1] < 126:
        continue

    data = df.iloc[:, :126].values.reshape(len(df), 42, 3)  # [T, 42, 3]

    df.iloc[:, :126].to_parquet(OUTPUT_DIR / f"{file_id}_aug_00_original.parquet", index=False)

    for i, combo in enumerate(valid_combinations, start=1):
        aug_data = apply_augmentations(data, combo)

        if aug_data.shape[0] < 2:
            print(f"{file_id} augment {i} quá ít frame, skip.")
            continue

        aug_df = pd.DataFrame(aug_data.reshape(aug_data.shape[0], -1))  # [T, 126]

        if aug_df.isna().all().all() or aug_df.empty:
            print(f"{file_id} augment {i} rỗng hoặc toàn NaN.")
            continue

        aug_df.to_parquet(OUTPUT_DIR / f"{file_id}_aug_{i:02d}.parquet", index=False)

print("Tăng cường hoàn tất.")
