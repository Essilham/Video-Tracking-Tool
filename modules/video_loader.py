import os
import streamlit as st

def load_predefined_video(video_option):
    if video_option == "Synthetic Cells":
        return os.path.abspath("./assets/synthetic_cells.mp4")
    elif video_option == "Synthetic Vehicles":
        return os.path.abspath("./assets/synthetic_vehicles.mp4")

def display_video(video_path):
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return
    try:
        st.video(video_path)
    except Exception as e:
        st.error(f"Error displaying video: {e}")
