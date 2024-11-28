import cv2

def track_objects(video_path):
    tracker = cv2.TrackerCSRT_create()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "./outputs/object_tracking_output.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return out_path
