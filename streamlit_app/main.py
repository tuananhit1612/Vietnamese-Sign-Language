import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import streamlit as st
import cv2
import modules.HandTrackingModule as htm
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from collections import Counter


model = load_model('D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/models/sign_language_model.h5')
excel_file_path = 'D:/tmp/test/video_to_text_data_basic.xlsx'
df = pd.read_excel(excel_file_path)

cap = cv2.VideoCapture("D:/tmp/D0490B_rotate.mp4")
hand = htm.handDetector()

st.set_page_config(layout="wide")
st.title("Test real-cam")
sentence = []
sequence = []
predictions = []
threshold = 0.5
window_frame = 40

st.markdown("""
        <style>
            .big-font {
                color: #e76f51 !important;
                font-size: 60px !important;
                border: 0.5rem solid #fcbf49 !important;
                border-radius: 2rem;
                text-align: center;
            }
        </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([6, 2,2])

with col1:
    video_placeholder = st.empty()

with col2:
    prediction_placeholder = st.empty()
with col3:
    list_prediction_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    hand_frame = hand.findHands(frame)
    hand_keypoints = hand.extract_landmarks(hand_frame)

    if hand_keypoints is not None:
        sequence.append(hand_keypoints)
        sequence = sequence[-window_frame:]

        if len(sequence) == window_frame:
            print(len(sentence))
            sequence_data = np.array(sequence).reshape(1, window_frame, 210)
            res = model.predict(sequence_data)
            predicted_class = np.argmax(res)
            confidence = res[0][predicted_class]
            predicted_label = df.loc[predicted_class, 'text']

            predictions.append(predicted_label)
            print(predicted_label)

            if np.unique(predictions[-10:])[0] == predicted_label:
                #if confidence > threshold:
                    if len(sentence) > 0:
                        if predicted_label != sentence[-1]:
                            sentence.append(predicted_label)
                    else:
                        sentence.append(predicted_label)
            print(sentence)
            prediction_placeholder.markdown(
                f'''<h2 class="big-font">{' '.join(sentence)}</h2>''',
                unsafe_allow_html=True
            )

    cTime = time.time()
    fps = 1 / (cTime - pTime) if 'pTime' in globals() else 0
    pTime = cTime

    cv2.putText(hand_frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    rgb_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)

    video_placeholder.image(rgb_frame, use_container_width=True)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
