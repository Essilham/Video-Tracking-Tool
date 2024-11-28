import streamlit as st
from modules.video_loader import load_predefined_video, display_video
from modules.optical_flow import track_optical_flow
from modules.object_tracking import track_objects
from modules.background_subtraction import apply_background_subtraction
import os

# Ensure the outputs folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# App Title
st.title("Interactive Video Tracking Tool")
st.write("Explore object tracking using Optical Flow, Object Tracking, and Background Subtraction!")

# Sidebar Configuration
st.sidebar.header("Tracking Configuration")
functionality = st.sidebar.selectbox(
    "Choose Tracking Functionality",
    ["Optical Flow", "Object Tracking", "Background Subtraction"]
)
video_option = st.sidebar.selectbox(
    "Choose Video",
    ["Synthetic Cells", "Synthetic Vehicles"]
)

# Load Predefined Videos
video_path = load_predefined_video(video_option)

# Display Original Video
st.write("### Original Video")
display_video(video_path)

# Apply Selected Functionality
if st.sidebar.button("Start Processing"):
    st.write(f"### {functionality} Results")

    # Process Video Based on User Selection
    if functionality == "Optical Flow":
        processed_video_path = track_optical_flow(video_path)
    elif functionality == "Object Tracking":
        processed_video_path = track_objects(video_path)
    elif functionality == "Background Subtraction":
        processed_video_path = apply_background_subtraction(video_path)

    # Display Processed Video
    st.write("### Processed Video")
    display_video(processed_video_path)

    # Allow Download of Processed Video
    with open(processed_video_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
