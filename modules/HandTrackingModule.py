import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    def normalize_hand_keypoints(self, landmarks):
        keypoints = []

        landmark_list = list(landmarks)

        if len(landmark_list) < 21:
            LandmarkType = type(landmark_list[0])
            for _ in range(21 - len(landmark_list)):
                landmark_list.append(LandmarkType(x=0.0, y=0.0, z=0.0))

        base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z

        for lm in landmark_list[:21]:
            dx = lm.x - base_x
            dy = lm.y - base_y
            dz = lm.z - base_z
            keypoints.extend([dx, dy, dz])

        return keypoints

    def extractAllPosition(self, img, draw=True):
        all_hand_keypoints = []

        if self.results.multi_hand_landmarks:
            hand_landmarks_list = list(self.results.multi_hand_landmarks)

            for handLms in hand_landmarks_list:
                landmark_list = list(handLms.landmark)

                if len(landmark_list) < 21:
                    LandmarkType = type(landmark_list[0])
                    for _ in range(21 - len(landmark_list)):
                        landmark_list.append(LandmarkType(x=0.0, y=0.0, z=0.0))

                hand_kp = self.normalize_hand_keypoints(landmark_list)
                all_hand_keypoints.extend(hand_kp)

            if len(hand_landmarks_list) == 1:
                all_hand_keypoints.extend([0.0] * 63)

            if len(hand_landmarks_list) > 2:
                all_hand_keypoints = all_hand_keypoints[:126]

        else:
            return None

        return all_hand_keypoints


def main():
    """
    test
    """
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture("D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/videos/D0001T.mp4")
    detector = handDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        all_hand_keypoints = detector.extractAllPosition(img)
        if all_hand_keypoints:
            print(all_hand_keypoints)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 13:
            break
if __name__ == "__main__":
    main()