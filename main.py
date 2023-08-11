import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define a custom video transformer class
class HolisticTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to RGB format
        img = frame.to_ndarray(format="rgb24")
        # Process the image with MediaPipe holistic
        results = holistic.process(img)
        # Draw the pose, face, and hand landmarks on the image
        img = mp.solutions.drawing_utils.draw_landmarks(
            img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        img = mp.solutions.drawing_utils.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        img = mp.solutions.drawing_utils.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        img = mp.solutions.drawing_utils.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # Return the transformed image
        return img

# Create a title and a sidebar
st.title("Streamlit Webrtc MediaPipe Holistic Demo")
st.sidebar.markdown("## Settings")
# Get the key and the mode from the sidebar
key = st.sidebar.text_input("Key", "example")
mode = st.sidebar.selectbox("Mode", ["off", "sendonly", "recvonly", "sendrecv"])

# Start the webrtc streamer with the custom video transformer
webrtc_streamer(
    key=key,
    video_transformer_factory=HolisticTransformer,
    async_transform=True)
