import streamlit as st
from modules.video_loader import load_predefined_video, display_video
from modules.optical_flow import track_optical_flow
from modules.object_tracking import track_objects
from modules.background_subtraction import apply_background_subtraction

# App Title
st.title("Interactive Video Tracking Tool")
st.write("Explore different object tracking features using synthetic videos!")

# Sidebar Configuration
st.sidebar.header("Tracking Configuration")
functionality = st.sidebar.selectbox("Choose Tracking Functionality", 
                                      ["Optical Flow", "Object Tracking", "Background Subtraction"])
video_option = st.sidebar.selectbox("Choose Video", ["Synthetic Cells", "Synthetic Vehicles"])

# Load Predefined Videos
video_path = load_predefined_video(video_option)

# Display Original Video
st.write("### Original Video")
display_video(video_path)

# Apply Selected Functionality
if st.sidebar.button("Start Processing"):
    if functionality == "Optical Flow":
        st.write("### Optical Flow Tracking Results")
        processed_video_path = track_optical_flow(video_path)
    elif functionality == "Object Tracking":
        st.write("### Object Tracking Results")
        processed_video_path = track_objects(video_path)
    elif functionality == "Background Subtraction":
        st.write("### Background Subtraction Results")
        processed_video_path = apply_background_subtraction(video_path)

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
