import cv2
import numpy as np
import tempfile

def track_optical_flow(video_path):
    """
    Perform Optical Flow tracking on a video.

    :param video_path: Path to the input video.
    :return: Path to the processed video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Parameters for Optical Flow
        feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Read the first frame
        ret, old_frame = cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame from the video.")
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Detect initial feature points
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        if p0 is None or len(p0) == 0:
            raise ValueError("No features detected in the first frame.")

        # Create a mask image for drawing the flow
        mask = np.zeros_like(old_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is None or st is None:
                break

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                frame = cv2.circle(frame, (a, b), 5, color=(0, 255, 0), thickness=-1)

            # Overlay the mask on the original frame
            img = cv2.add(frame, mask)

            # Write the frame to the output video
            out.write(img)

            # Update for the next frame
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        out.release()

    return out_path
