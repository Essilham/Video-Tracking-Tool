import os

def load_predefined_video(video_option):
    # Load videos based on user selection
    if video_option == "Synthetic Cells":
        return "./assets/synthetic_cells.mp4"
    elif video_option == "Synthetic Vehicles":
        return "./assets/synthetic_vehicles.mp4"

def display_video(video_path):
    st.video(video_path)
