import cv2
import tempfile
import streamlit as st

def display_video(video_path):
    # Temporarily store video for Streamlit video player
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(open(video_path, "rb").read())
        st.video(tmp.name)
