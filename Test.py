import cv2
from ultralytics import YOLO
from datetime import datetime

# Load the pre-trained YOLOv8 model
model = YOLO('D:/Python Projects/pythonProject/.venv/Include/runs/detect/train6/weights/best.pt')  # Ensure this path points to your trained model file

# Start video capture (0 for webcam, or provide a video path)
cap = cv2.VideoCapture(0)

# Tracking and logging setup
behavior_counts = {
    "menyontek-berdiri-jongkok": 0,
    "menyontek-deteksi-hape-contekan": 0,
    "menyontek-lihat-atas-bawah": 0,
    "menyontek-lirik-kiri-kanan": 0,
    "menyontek-tengok-depan-belakang": 0,
    "menyontek-tengok-kiri-kanan": 0,
    "normal": 0
}

# Alert thresholds for specific behaviors
alert_thresholds = {
    "menyontek-berdiri-jongkok": 5,
    "menyontek-deteksi-hape-contekan": 5,
    "menyontek-lihat-atas-bawah": 5,
    "menyontek-lirik-kiri-kanan": 5,
    "menyontek-tengok-depan-belakang": 5,
    "menyontek-tengok-kiri-kanan": 5,
}

# Translation mapping from Indonesian to English
translation_map = {
    "menyontek-berdiri-jongkok": "Cheating - Crouching",
    "menyontek-deteksi-hape-contekan": "Cheating - Using Phone",
    "menyontek-lihat-atas-bawah": "Cheating - Looking Up and Down",
    "menyontek-lirik-kiri-kanan": "Cheating - Looking Left and Right",
    "menyontek-tengok-depan-belakang": "Cheating - Looking Forward and Backward",
    "menyontek-tengok-kiri-kanan": "Cheating - Looking Side to Side",
    "normal": "Normal Behavior"
}

log_file = "suspicious_activity_log.txt"

# Placeholder for student name identification (replace with actual logic)
def get_student_name():
    # This function should return the name of the detected student.
    # For now, we'll use a placeholder. You should replace this with actual identification logic.
    return "Student_Name"

# Function to log suspicious events
def log_event(event_type, student_name):
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - Suspicious Behavior Detected: {event_type} by {student_name}\n")
    print(f"Alert: {event_type} detected by {student_name} and logged.")

# Function to alert the user (you can modify this to send an email, SMS, etc.)
def alert_user(event_type, student_name):
    print(f"ALERT! {event_type} detected by {student_name}!")

# Real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the current frame
    results = model.predict(frame, stream=True)  # Use `predict` for inference only

    # Process detections for the current frame
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract detection data
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            label = model.names[class_id]  # Class name from model labels

            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                english_label = translation_map.get(label, label)  # Translate to English
                cv2.putText(frame, f'{english_label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Get the student name (you need to replace this with actual identification logic)
                student_name = get_student_name()

                # Track and count occurrences of specific behaviors
                if label in behavior_counts:
                    behavior_counts[label] += 1  # Increment count
                    # Trigger an alert if the count exceeds the threshold
                    if behavior_counts[label] == alert_thresholds[label]:
                        log_event(english_label, student_name)  # Log with student name
                        alert_user(english_label, student_name)   # Alert the user
                        behavior_counts[label] = 0  # Reset count after logging

    # Display the processed frame
    cv2.imshow('Exam Invigilation', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
