import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import mediapipe as mp
from SiFormer.SiFormer import SiFormer
import time

MODEL_PATH = "D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/models/best_model.pth"
NUM_CLASSES = 7
SEQ_LEN = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiFormer(num_classes=NUM_CLASSES, num_hid=126, seq_len=SEQ_LEN, device=DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

mp_hands = mp.solutions.holistic
holistic = mp_hands.Holistic(min_detection_confidence=0.5, model_complexity=1)

cap = cv2.VideoCapture("D:\Video_Full/D0495.mp4")
left_hand_seq = []
right_hand_seq = []

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(img_rgb)

        if result.left_hand_landmarks:
            left = [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark]

        else:
            left = [[0.0, 0.0, 0.0]] * 21

        if result.right_hand_landmarks:
            right = [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark]
        else:
            right = [[0.0, 0.0, 0.0]] * 21
        left_hand_seq.append(left)
        right_hand_seq.append(right)

        left_hand_seq = left_hand_seq[-SEQ_LEN:]
        right_hand_seq = right_hand_seq[-SEQ_LEN:]

        if len(left_hand_seq) == SEQ_LEN:
            l_tensor = torch.tensor(left_hand_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            r_tensor = torch.tensor(right_hand_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            logits = model(l_tensor, r_tensor)
            pred = torch.argmax(logits, dim=-1).item()
            print("Predicted class ID:", pred)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
