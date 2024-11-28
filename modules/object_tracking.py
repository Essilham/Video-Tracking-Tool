import cv2
import tempfile
import streamlit as st

def track_objects(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to load the video.")

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # User-defined bounding box via Streamlit sliders
    st.sidebar.subheader("Set Object Bounding Box")
    x = st.sidebar.slider("X Coordinate", 0, frame_width, 50)
    y = st.sidebar.slider("Y Coordinate", 0, frame_height, 50)
    w = st.sidebar.slider("Width", 10, frame_width - x, 100)
    h = st.sidebar.slider("Height", 10, frame_height - y, 100)
    bbox = (x, y, w, h)

    # Initialize the tracker
    tracker = cv2.TrackerKCF_create()  # Alternatively, use cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracker
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

            out.write(frame)

        cap.release()
        out.release()

    return out_path
