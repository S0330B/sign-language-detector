import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time


model_dict = pickle.load(open('models/model.pickle', 'rb'))
data_dict = pickle.load(open('models/data.pickle', 'rb'))

model = model_dict['model']
labels = data_dict['labels']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🖐️ Real-Time Sign Language Detection</h1>", unsafe_allow_html=True)
st.write("Detects and labels both your **left** and **right** hand signs using your trained model.")


if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.cap = None

def toggle_camera():
    if st.session_state.camera_running:
        st.session_state.camera_running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            cv2.destroyAllWindows()
    else:
        st.session_state.camera_running = True

st.button("🎥 Start / Stop Camera", on_click=toggle_camera)
FRAME_WINDOW = st.image([])


def process_frame(frame, hands, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predictions = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            data_aux = []

            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x, lm.y])

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_sign = prediction[0]
                predictions.append((hand_label, predicted_sign))

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    for idx, (hand_label, sign) in enumerate(predictions):
        cv2.putText(frame, f"{hand_label} Hand: {sign}",
                    (10, 40 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


if st.session_state.camera_running:
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    st.info("Press the button again to stop the camera.")

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to access the camera.")
            break

        frame = cv2.flip(frame, 1)
        frame = process_frame(frame, hands, model)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)  

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    st.session_state.camera_running = False
else:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        cv2.destroyAllWindows()
    st.info("Click **Start / Stop Camera** to begin.")
