import cv2
import tempfile

def track_objects(video_path):
    tracker = cv2.TrackerCSRT_create()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create a temporary file for the processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Initialize tracker
        ret, frame = cap.read()
        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, bbox)

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
