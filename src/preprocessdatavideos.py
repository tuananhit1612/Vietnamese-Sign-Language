import cv2
import numpy as np
import os
import modules.HandTrackingModule as htm
import time

INPUT_VIDEO_DIR = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/videos'
OUTPUT_NPY_DIR = 'D:/Dev//DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/npy'
os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    detector = htm.handDetector()
    sequence = []
    video_name = os.path.basename(video_path)
    pTime = 0
    cTime = 0
    if not cap.isOpened():
        print(f"Can't open {video_name}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.findHands(frame)
        keypoints = detector.extractAllPosition(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == 23: #esc
            break
        if keypoints is not None:
            sequence.append(keypoints)

    cap.release()
    cv2.destroyAllWindows()

    if sequence:
        sequence = np.array(sequence)
        np.save(output_path, sequence)
    else:
        print(f"Video {video_name} not found.")

def preprocess_dataset():
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("Not found videos 'data/videos/'.")
        return

    for video_file in video_files:
        video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        output_filename = os.path.splitext(video_file)[0] + '.npy'
        output_path = os.path.join(OUTPUT_NPY_DIR, output_filename)

        process_video(video_path, output_path)

    print("Complete process dataset!")

if __name__ == '__main__':
    preprocess_dataset()
