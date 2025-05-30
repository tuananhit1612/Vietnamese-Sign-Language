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

def extract_hand_landmarks(df, hand="right"):
    hand_idx = 0 if hand == "left" else 21
    coords = df.iloc[:, hand_idx*3 : (hand_idx+21)*3].values.reshape(-1, 21, 3)
    return coords

def create_animation_from_array(hand_array):
    images = []

    for frame in hand_array:
        image = np.zeros((600, 600, 3), dtype=np.uint8)
        landmarks = landmark_pb2.NormalizedLandmarkList()

        for x, y, z in frame:
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
    parquet_path = r"D:\train_test\train_landmark_files\D0489_rotated_z45.parquet"

    df = pd.read_parquet(parquet_path)

    hand_to_view = "right"
    if df[hand_to_view].iloc[0] != 1:
        print(f"Tay {hand_to_view} không xuất hiện trong dữ liệu.")
    else:
        hand_array = extract_hand_landmarks(df, hand=hand_to_view)
        create_animation_from_array(hand_array)
