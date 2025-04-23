import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, mode=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpFace = mp.solutions.face_mesh
        self.faceMesh = self.mpFace.FaceMesh(
            static_image_mode=self.mode,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms,self.mpFace.FACEMESH_CONTOURS)
        return img

    def findPosition(self, img, faceNo=0, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            myFace = self.results.multi_face_landmarks[faceNo]
            for id, lm in enumerate(myFace.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        return lmList

    def normalize_face_keypoints(self, landmarks):
        keypoints = []

        landmark_list = list(landmarks)

        base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z

        for lm in landmark_list[:468]:
            dx = lm.x - base_x
            dy = lm.y - base_y
            dz = lm.z - base_z
            keypoints.extend([dx, dy, dz])

        return keypoints

    def extractAllPosition(self, img, draw=True):
        all_face_keypoints = []

        if self.results.multi_face_landmarks:
            face_landmarks_list = list(self.results.multi_face_landmarks)

            first_face_landmarks = face_landmarks_list[0]
            landmark_list = list(first_face_landmarks.landmark)

            if len(landmark_list) < 468:
                LandmarkType = type(landmark_list[0])
                for _ in range(468 - len(landmark_list)):
                    landmark_list.append(LandmarkType(x=0.0, y=0.0, z=0.0))

            face_kp = self.normalize_face_keypoints(landmark_list)
            all_face_keypoints.extend(face_kp)

        else:
            return None

        return all_face_keypoints

