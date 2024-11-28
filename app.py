import streamlit as st
from modules.video_loader import load_video, display_video
from modules.optical_flow import track_optical_flow
from modules.object_tracking import track_objects
import os

# App Title
st.title("Interactive Video Tracking Tool")
st.write("Upload a video and explore different object tracking algorithms.")

# Sidebar Configuration
algorithm = st.sidebar.selectbox("Select Tracking Algorithm", ["Optical Flow", "Object Tracking"])
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if video_file:
    # Save uploaded video temporarily
    input_path = f"./assets/{video_file.name}"
    with open(input_path, "wb") as f:
        f.write(video_file.read())

    # Load Video
    st.write("### Original Video")
    display_video(input_path)

    # Apply Tracking Algorithm
    if algorithm == "Optical Flow":
        st.write("### Optical Flow Tracking")
        processed_video_path = track_optical_flow(input_path)
    elif algorithm == "Object Tracking":
        st.write("### Object Detection + Tracking")
        processed_video_path = track_objects(input_path)

    # Display Processed Video
    st.write("### Processed Video")
    display_video(processed_video_path)

    # Download Processed Video
    with open(processed_video_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
