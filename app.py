import streamlit as st
from modules.yolo import yolo_object_detection
import os

# Ensure the outputs folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# App Title
st.title("YOLO Object Detection Tool")
st.write("Detect objects in videos using YOLO.")

# Sidebar Configuration
st.sidebar.header("Configuration")
video_option = st.sidebar.selectbox(
    "Choose Video",
    ["Synthetic Cells", "Synthetic Vehicles"]
)

model_option = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n", "yolov8s", "yolov8m"]  # Options for YOLO model sizes
)

# Load Predefined Videos
def load_video_path(video_option):
    if video_option == "Synthetic Cells":
        return "./assets/synthetic_cells.mp4"
    elif video_option == "Synthetic Vehicles":
        return "./assets/synthetic_vehicles.mp4"

video_path = load_video_path(video_option)

# Display Original Video
st.write("### Original Video")
st.video(video_path)

# Process Video with YOLO
if st.sidebar.button("Run YOLO Detection"):
    st.write(f"### YOLO Detection Results ({model_option})")

    # Apply YOLO detection
    processed_video_path = yolo_object_detection(video_path, model_name=model_option)

    # Display Processed Video
    st.write("### Processed Video")
    st.video(processed_video_path)

    # Allow Download of Processed Video
    with open(processed_video_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
