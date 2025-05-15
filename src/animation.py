import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def load_landmark_sequence_from_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)

    renamed_cols = {}
    for i in range(21):
        renamed_cols[f'x_left_{i}'] = f'x_left_hand_{i}'
        renamed_cols[f'y_left_{i}'] = f'y_left_hand_{i}'
        renamed_cols[f'z_left_{i}'] = f'z_left_hand_{i}'
        renamed_cols[f'x_right_{i}'] = f'x_right_hand_{i}'
        renamed_cols[f'y_right_{i}'] = f'y_right_hand_{i}'
        renamed_cols[f'z_right_{i}'] = f'z_right_hand_{i}'

    df = df.rename(columns=renamed_cols)
    return df

def create_animation_from_df(df, hand='right'):
    images = []

    for i in range(len(df)):
        x_cols = df.filter(regex=f'x_{hand}_hand_').iloc[i].values
        y_cols = df.filter(regex=f'y_{hand}_hand_').iloc[i].values
        z_cols = df.filter(regex=f'z_{hand}_hand_').iloc[i].values

        image = np.zeros((600, 600, 3), dtype=np.uint8)
        landmarks = landmark_pb2.NormalizedLandmarkList()

        for x, y, z in zip(x_cols, y_cols, z_cols):
            landmarks.landmark.add(x=float(x), y=float(y), z=float(z))

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

        images.append(image)

    fig = plt.figure()
    im = plt.imshow(images[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=images, interval=100, blit=True)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parquet_file = r"D:\tmp\test\train_landmark_files\augmented_combinations\D0489_aug_18_mirror_scale.parquet"

    df = load_landmark_sequence_from_parquet(parquet_file)

    create_animation_from_df(df, hand='right')
