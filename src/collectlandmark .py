import os
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

df_csv = pd.read_excel("D:/train_test/train.xlsx")

def extract_landmarks_and_save(video_path, file_id):
    cap = cv2.VideoCapture(video_path)
    left_hand_seq = []
    right_hand_seq = []
    left_flags = []
    right_flags = []

    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(img_rgb)

        if result.left_hand_landmarks:
            left = [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark]
            left_flags.append(1)
        else:
            left = [[0.0, 0.0, 0.0]] * 21
            left_flags.append(0)

        if result.right_hand_landmarks:
            right = [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark]
            right_flags.append(1)
        else:
            right = [[0.0, 0.0, 0.0]] * 21
            right_flags.append(0)

        left_hand_seq.append(left)
        right_hand_seq.append(right)

    cap.release()

    has_hand = [l or r for l, r in zip(left_flags, right_flags)]
    try:
        start = has_hand.index(1)
        end = len(has_hand) - 1 - has_hand[::-1].index(1)
    except ValueError:
        return None

    left_hand_seq = left_hand_seq[start:end+1]
    right_hand_seq = right_hand_seq[start:end+1]
    left_flags_crop = left_flags[start:end+1]
    right_flags_crop = right_flags[start:end+1]

    left_flag = int(any(left_flags_crop))
    right_flag = int(any(right_flags_crop))

    left_array = np.array(left_hand_seq).reshape(len(left_hand_seq), -1)
    right_array = np.array(right_hand_seq).reshape(len(right_hand_seq), -1)
    combined = np.concatenate([left_array, right_array], axis=1)

    df = pd.DataFrame(combined)
    df["left"] = left_flag
    df["right"] = right_flag

    return df

output_dir = "D:/train_test/train_landmark_files"
os.makedirs(output_dir, exist_ok=True)

for _, row in df_csv.iterrows():
    video_path = f"D:/Video_Full/{row['file_id']}.mp4"
    file_id = row['file_id']

    df = extract_landmarks_and_save(video_path, file_id)
    if df is not None:
        output_path = os.path.join(output_dir, f"{file_id}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")
    else:
        print(f"Skip: {file_id}")
