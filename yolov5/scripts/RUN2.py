import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
import time

import torch
import cv2

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Set up the video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Human class ID in COCO dataset is 0
human_class_id = 0
LABEL = "Human"

# Time tracking for logging every 3 seconds
last_log_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)
# results = model(frame.to('cpu'))  # Force inference on CPU

    human_detected = False

    # Process results
    for *xyxy, conf, cls in results.xyxy[0]:
        # Filter only humans
        if int(cls) == human_class_id:

            human_detected = True

            # Get the coordinates and confidence
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = conf.item()

            # Draw a rectangle around the human
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{LABEL} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    # Log every 3 seconds
    current_time = time.time()
    
    if current_time - last_log_time >= 3:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if human_detected:
            print(f"{timestamp}, Human detected")
        else:
            print(f"{timestamp}, ")

        last_log_time = current_time

    # Display the frame with detections
    cv2.imshow("YOLOv5 Human Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
