import cv2
import mediapipe as mp
import pandas as pd

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                enable_segmentation=False,
                                refine_face_landmarks=True,
                                min_detection_confidence=0.5)

df_csv = pd.read_excel("D:/tmp/test/train.xlsx")

cols = ['sequence_id', 'frame'] + \
       [f'x_left_{i}' for i in range(21)] + [f'y_left_{i}' for i in range(21)] + [f'z_left_{i}' for i in range(21)] + \
       [f'x_right_{i}' for i in range(21)] + [f'y_right_{i}' for i in range(21)] + [f'z_right_{i}' for i in range(21)]

def extract_landmarks(video_path, sequence_id):
    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_idx = 0
    start_collect = False
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return pd.DataFrame()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(img_rgb)

        landmarks = []
        if res.left_hand_landmarks or res.right_hand_landmarks:
            start_collect = True
        else:
            start_collect = False
        if start_collect:
            if res.left_hand_landmarks:
                landmark_list = list(res.left_hand_landmarks.landmark)
                if len(landmark_list) < 21:
                    landmark_list.extend([mp.framework.formats.landmark_pb2.Landmark(x=0.0, y=0.0, z=0.0)] * (21 - len(landmark_list)))
                for lm in landmark_list:
                    landmarks += [lm.x, lm.y, lm.z]
            else:
                landmarks += [0.0] * 63

            if res.right_hand_landmarks:
                landmark_list = list(res.right_hand_landmarks.landmark)

                if len(landmark_list) < 21:
                    landmark_list.extend([mp.framework.formats.landmark_pb2.Landmark(x=0.0, y=0.0, z=0.0)] * (21 - len(landmark_list)))

                for lm in landmark_list:
                    landmarks += [lm.x, lm.y, lm.z]
            else:
                landmarks += [0.0] * 63

        if landmarks:
            rows.append([sequence_id, frame_idx] + landmarks)
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows, columns=cols)
    return df


for index, row in df_csv.iterrows():
    video_path = f"D:/tmp/test/{row['file_id']}.mp4"
    file_id = row['file_id']
    sequence_id = row['sequence_id']

    df_landmarks = extract_landmarks(video_path, sequence_id)
    if not df_landmarks.empty:
        df_landmarks.to_parquet(f"D:/tmp/test/train_landmark_files/{file_id}.parquet", index=False)
        print(f"Done {file_id}")
