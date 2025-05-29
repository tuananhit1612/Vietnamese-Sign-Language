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
SEQ_LEN = 50
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = torch.load("D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data\models/normalization_stats.pth")
mean = stats["mean"]
std = stats["std"]

mean_l = mean[:63]
mean_r = mean[63:]
std_l = std[:63]
std_r = std[63:]

model = SiFormer(
    num_classes=7,
    num_hid=126,
    seq_len=50,
    num_enc_layers=4,
    num_dec_layers=3,
    device=DEVICE
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

mp_hands = mp.solutions.holistic
holistic = mp_hands.Holistic(min_detection_confidence=0.5, model_complexity=1)

cap = cv2.VideoCapture("D:/train_test/D0489.mp4")
left_hand_seq = []
right_hand_seq = []

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(img_rgb)
        if result.left_hand_landmarks or result.right_hand_landmarks:
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

                l_tensor = l_tensor.view(1, SEQ_LEN, -1)
                r_tensor = r_tensor.view(1, SEQ_LEN, -1)

                mean_l = mean_l.view(1, 1, -1).to(DEVICE)
                std_l = std_l.view(1, 1, -1).to(DEVICE)
                mean_r = mean_r.view(1, 1, -1).to(DEVICE)
                std_r = std_r.view(1, 1, -1).to(DEVICE)

                l_tensor = (l_tensor - mean_l) / (std_l + 1e-6)
                r_tensor = (r_tensor - mean_r) / (std_r + 1e-6)

                logits = model(l_tensor, r_tensor)
                probs = torch.softmax(logits, dim=-1)
                max_prob, pred = torch.max(probs, dim=-1)

                pred_id = pred.item()
                confidence = max_prob.item()

                if confidence >= THRESHOLD:
                    print(f"predicted: {pred_id} (confidence: {confidence:.2f})")
                else:
                    print(f"skip ({confidence:.2f}).")
        else:
            left_hand_seq = []
            right_hand_seq = []
        cv2.imshow("Test",frame)
        if cv2.waitKey(1) ==  27:
            break

cap.release()
cv2.destroyAllWindows()
