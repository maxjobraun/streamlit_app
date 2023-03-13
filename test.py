import pandas as pd
import cv2
import mediapipe as mp
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
