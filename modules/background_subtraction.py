import cv2
import tempfile

def apply_background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fgmask = fgbg.apply(frame)

            # Convert mask to BGR for visualization
            fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            out.write(fgmask_bgr)

        cap.release()
        out.release()

    return out_path
