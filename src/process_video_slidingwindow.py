import cv2
import numpy as np
import os
import modules.HandTrackingModule as htm
#import modules.FaceTrackingModule as ftm
#import modules.PoseTrackingModule as ptm

INPUT_VIDEO_DIR = 'D:/tmp/test'
OUTPUT_NPY_DIR = 'D:/tmp/test/train_landmark_files'
os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    hand_detector = htm.handDetector()
    #face_detector = ftm.faceDetector()
    #pose_detector = ptm.poseDetector()

    sequence = []
    gesture_count = 1

    video_name = os.path.basename(video_path).split('.')[0]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_frame = hand_detector.findHands(frame)
        #face_frame = face_detector.findFace(frame)
        #pose_frame = pose_detector.findPose(frame)

        hand_keypoints = hand_detector.extract_landmarks(hand_frame)
        #f#ace_keypoints = face_detector.extractAllPosition(face_frame)
        #pose_keypoints = pose_detector.extractAllPosition(pose_frame)

        if hand_keypoints is not None:
            #combined_keypoints = hand_keypoints + face_keypoints + pose_keypoints
            sequence.append(hand_keypoints)
            frame_count += 1

        if frame_count == 40:
            print(f"Saving gesture {gesture_count} with 30 frames.")
            output_path = os.path.join(output_dir, f"{video_name}_{gesture_count}.npy")
            np.save(output_path, np.array(sequence))
            gesture_count += 1
            frame_count = 0
            sequence = []

    if len(sequence) > 0:
        print(f"Video {video_name} has less than 30 frames, padding to 30 frames.")
        while len(sequence) < 40:
            sequence.append(np.zeros_like(sequence[0]))
        output_path = os.path.join(output_dir, f"{video_name}_{gesture_count}.npy")
        np.save(output_path, np.array(sequence))
        gesture_count += 1

    cap.release()
    cv2.destroyAllWindows()

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
