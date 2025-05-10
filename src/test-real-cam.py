import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import modules.HandTrackingModule as htm
import modules.PoseTrackingModule as ptm
import modules.FaceTrackingModule as ftm


model = load_model('D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/models/sign_language_model.h5')

cap = cv2.VideoCapture("D:/tmp/test/TTA003.mp4")
hand_detector = htm.handDetector()
#face_detector = ftm.faceDetector()
#pose_detector = ptm.poseDetector()

excel_file_path = 'D:/tmp/test/video_to_text_data_basic.xlsx'
df = pd.read_excel(excel_file_path)

frame_count = 0
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hand_frame = hand_detector.findHands(frame)
    #pose_frame = pose_detector.findPose(frame)
    #face_frame = face_detector.findFace(frame)

    hand_keypoints = hand_detector.extractAllPosition(frame,hand_frame)
    #pose_keypoints = pose_detector.extractAllPosition(pose_frame)
    #face_keypoints = face_detector.extractAllPosition(face_frame)

    if hand_keypoints is not None:
        sequence.append(hand_keypoints)
        frame_count += 1

        if frame_count == 30:
            sequence_data = np.array(sequence).reshape(1, 30, 210)
            res = model.predict(np.array(sequence_data))
            predicted_class = np.argmax(res, axis=1)
            predicted_label = df.loc[predicted_class[0], 'text']
            print(predicted_label)
            #cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            sequence = []
            frame_count = 0

    cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
