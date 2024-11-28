import cv2
import tempfile

def apply_background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Create a temporary file for the processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fgmask = fgbg.apply(frame)

            # Highlight the moving objects
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter out small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

    return out_path
