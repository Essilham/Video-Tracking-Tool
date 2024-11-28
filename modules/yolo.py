from ultralytics import YOLO
import cv2
import tempfile

def yolo_object_detection(video_path, model_name="yolov8n"):
    """
    Perform object detection on the video using YOLO.

    :param video_path: Path to the input video.
    :param model_name: YOLO model name (default: yolov8n).
    :return: Path to the processed video.
    """
    # Load YOLO model
    model = YOLO(model_name)  # Load pre-trained YOLOv8 model
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        out_path = temp_file.name
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model.predict(source=frame, conf=0.4, save=False)
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                confidence = result.conf[0]  # Confidence score
                class_id = int(result.cls[0])  # Class ID
                label = f"{model.names[class_id]}: {confidence:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

    return out_path

