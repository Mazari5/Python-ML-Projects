import cv2
from fer import FER

# Initialize the emotion detector
detector = FER(mtcnn=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Analyze emotions in the frame
    emotions = detector.detect_emotions(frame)

    # Display the resulting frame with emotions
    for emotion in emotions:
        box = emotion["box"]
        emotions_detected = emotion["emotions"]
        max_emotion = max(emotions_detected, key=emotions_detected.get)
        text = f"{max_emotion}: {emotions_detected[max_emotion]:.2f}"

        # Draw the bounding box around the face
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
