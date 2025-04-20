import cv2
import numpy as np
import os
import time
import modules.HandTrackingModule as htm
import modules.FaceTrackingModule as ftm
import modules.PoseTrackingModule as ptm

INPUT_VIDEO_DIR = 'D:/output/test'
OUTPUT_NPY_DIR = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/npy'
os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    # Initialize detectors
    hand_detector = htm.handDetector()
    #face_detector = ftm.faceDetector()
    pose_detector = ptm.poseDetector()

    sequence = []
    gesture_count = 1
    last_detected_time = time.time()
    max_idle_time = 1.0

    video_name = os.path.basename(video_path).split('.')[0]
    last_frame_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = hand_detector.findHands(frame)
        #frame = face_detector.findFace(frame)
        frame = pose_detector.findPose(frame)

        hand_keypoints = hand_detector.extractAllPosition(frame)
        #face_keypoints = face_detector.extractAllPosition(frame)
        pose_keypoints = pose_detector.extractAllPosition(frame)

        if hand_keypoints is not None and pose_keypoints is not None:
            combined_keypoints = hand_keypoints + pose_keypoints
            sequence.append(combined_keypoints)
            last_detected_time = time.time()

        if not hand_detector.results.multi_hand_landmarks:
            current_time = time.time()
            if current_time - last_detected_time > max_idle_time:
                if sequence:
                    print(f"Gesture {gesture_count} completed. Saving data.")
                    output_path = os.path.join(output_dir, f"{video_name}_{gesture_count}.npy")
                    np.save(output_path, np.array(sequence))
                    gesture_count += 1


                sequence = []

        cTime = time.time()
        fps = 1 / (cTime - last_frame_time) if last_frame_time else 0
        last_frame_time = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    #os.remove(video_path)

def preprocess_dataset():
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("No videos found in the data folder.")
        return

    for i, video_file in enumerate(video_files):
        print(f"Processing video {i + 1}/{len(video_files)}")
        video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        process_video(video_path, OUTPUT_NPY_DIR)

    print("Dataset processing complete!")

if __name__ == '__main__':
    preprocess_dataset()
