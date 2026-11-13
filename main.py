
import cv2

# Open laptop webcam
camera = cv2.VideoCapture(0)

# Check whether camera opened successfully
if not camera.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Webcam started successfully!")
print("Press Q to close the camera.")

while True:

    # Capture video frame
    success, frame = camera.read()

    if not success:
        print("Error: Cannot read camera frame")
        break

    # Display webcam video
    cv2.imshow("Drowsiness Detection System", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Webcam closed successfully!")
