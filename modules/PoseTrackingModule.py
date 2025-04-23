import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, poseNo=0, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def normalize_pose_keypoints(self, landmarks):
        keypoints = []
        landmark_list = list(landmarks)

        # Padding nếu số lượng landmark ít hơn 33
        if len(landmark_list) < 33:
            LandmarkType = type(landmark_list[0])
            for _ in range(33 - len(landmark_list)):
                landmark_list.append(LandmarkType(x=0.0, y=0.0, z=0.0))

        base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z

        for lm in landmark_list[:33]:
            dx = lm.x - base_x
            dy = lm.y - base_y
            dz = lm.z - base_z
            keypoints.extend([dx, dy, dz])

        return keypoints

    def extractAllPosition(self, img, draw=True):
        all_pose_keypoints = []

        if self.results.pose_landmarks:
            landmark_list = list(self.results.pose_landmarks.landmark)

            if len(landmark_list) < 33:
                LandmarkType = type(landmark_list[0])
                for _ in range(33 - len(landmark_list)):
                    landmark_list.append(LandmarkType(x=0.0, y=0.0, z=0.0))

            pose_kp = self.normalize_pose_keypoints(landmark_list)
            all_pose_keypoints.extend(pose_kp)

        else:
            all_pose_keypoints.extend([0.0] * 33 * 3)

        return all_pose_keypoints
