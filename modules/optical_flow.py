import cv2
import numpy as np
import tempfile

def dense_optical_flow(video_path):
    """
    Perform Dense Optical Flow tracking on a video.

    :param video_path: Path to the input video.
    :return: Path to the processed video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Read the first frame
        ret, old_frame = cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame from the video.")
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Initialize HSV image for flow visualization
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255  # Saturation is fixed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                                                poly_n=5, poly_sigma=1.2, flags=0)

            # Convert flow to visual representation
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

            # Convert HSV to BGR for visualization
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Write frame to output video
            out.write(rgb_flow)

            # Update for the next frame
            old_gray = frame_gray.copy()

        cap.release()
        out.release()

    return out_path
