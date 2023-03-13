import streamlit as st
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
#Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from functions import train_val_test_split, parquets_to_Xy, load_relevant_data_subset



st.markdown("Welcome to our Isolated Sign Language Interpreter!")
st.markdown("Ready? Click below and sign away!")
button = st.button("Start sign capture")
if button:
    st.markdown("Your sign is being analyzed, one moment please..")
    frame_count = 0
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        landmark_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Right hand
            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    landmark_data.append([frame_count, 'right_hand', idx, landmark.x, landmark.y, landmark.z])
            else:
                for idx, dum_landmark in enumerate(list(range(21))):
                    landmark_data.append([frame_count, 'right_hand', idx, (np.nan), (np.nan), (np.nan)])
            # Left Hand
            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    landmark_data.append([frame_count, 'left_hand', idx, landmark.x, landmark.y, landmark.z])
            else:
                for idx, dum_landmark in enumerate(list(range(21))):
                    landmark_data.append([frame_count, 'left_hand', idx, (np.nan), (np.nan), (np.nan)])
            # Draw face landmarks
            if results.face_landmarks:
                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    landmark_data.append([frame_count, 'face', idx, landmark.x, landmark.y, landmark.z])
            else:
                for idx, dum_landmark in enumerate(list(range(468))):
                    landmark_data.append([frame_count, 'face', idx, (np.nan), (np.nan), (np.nan)])
            # Pose Detections
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_data.append([frame_count, 'pose', idx, landmark.x, landmark.y, landmark.z])
            else:
                for idx, dum_landmark in enumerate(list(range(33))):
                    landmark_data.append([frame_count, 'pose', idx, (np.nan), (np.nan), (np.nan)])
            # Render detections
            mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            frame_count += 1
    # Convert list to dataframe
    landmark_df = pd.DataFrame(landmark_data, columns=['frame', 'type', 'landmark_index', 'x', 'y', 'z'])
    # Write dataframe to parquet file
    landmark_df.to_parquet('landmarks.parquet')
    cap.release()
    cv2.destroyAllWindows()

    ## Reading parquet
    path = r'~/code/Mohammad-Fadel/isolated_sign_language/streamlit_app/landmarks.parquet'
    ## Reading parquet and processing it into X_pred format
    lips_landmarks = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]
    ROWS_PER_FRAME = 75 + len(lips_landmarks)
    input_read = load_relevant_data_subset(path, max_frames=100)
    X_pred = np.array(input_read)
    model = tf.keras.models.load_model(r'~/code/Mohammad-Fadel/isolated_sign_language/streamlit_app/my_model')
    y_pred = model.predict(X_pred)
    # take index of highest probability value and match it to the list of words the model was trained on
    y_pred_index = np.argmax(y_pred)
    signs = ["drink","water","after","another","child","dad","every","thankyou","bye","airplane"]
    prediction = signs[y_pred_index]
    st.markdown("Your sign is:")
    st.markdown(prediction)
