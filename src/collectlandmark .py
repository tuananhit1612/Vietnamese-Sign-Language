import os
import time

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5
)

df_csv = pd.read_excel("D:/tmp/test/train.xlsx")

def extract_landmarks_and_save(video_path, file_id):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    rows = []

    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(img_rgb)
        if not res.left_hand_landmarks and not res.right_hand_landmarks:
            continue
        cv2.imshow('video',frame)
        if cv2.waitKey(1) == 27:
            break
        row = {'frame': frame_idx}

        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                row[f'x_left_hand_{i}'] = lm.x
                row[f'y_left_hand_{i}'] = lm.y
                row[f'z_left_hand_{i}'] = lm.z
        else:
            for i in range(21):
                row[f'x_left_hand_{i}'] = 0.0
                row[f'y_left_hand_{i}'] = 0.0
                row[f'z_left_hand_{i}'] = 0.0

        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                row[f'x_right_hand_{i}'] = lm.x
                row[f'y_right_hand_{i}'] = lm.y
                row[f'z_right_hand_{i}'] = lm.z
        else:
            for i in range(21):
                row[f'x_right_hand_{i}'] = 0.0
                row[f'y_right_hand_{i}'] = 0.0
                row[f'z_right_hand_{i}'] = 0.0

        rows.append(row)
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    print(f"{file_id} - {len(df)} frames saved")
    return df

for _, row in df_csv.iterrows():
    video_path = f"D:/tmp/test/{row['file_id']}.mp4"
    file_id = row['file_id']

    df_landmarks = extract_landmarks_and_save(video_path, file_id)
    if df_landmarks is not None:
        output_path = f"D:/tmp/test/train_landmark_files/{file_id}.parquet"
        df_landmarks.to_parquet(output_path, index=False)
