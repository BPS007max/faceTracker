# import cv2
# video_cap= cv2.VideoCapture(0)
# while True:
#     ret, video_data = video_cap.read()
#     if not ret:
#        print("failed to grab frame: ")
#        break
#     cv2.imshow("video_live",video_data)
#     if cv2.waitKey(10) & 0xFF== ord("a"):
#       break
# video_cap.release()
# cv2.destroyAllWindows()

import cv2

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
