
import cv2
import time

from src.camera.webcam import Webcam
from src.detection.face_detector import FaceDetector
from src.detection.eye_detector import EyeDetector


webcam = Webcam()
face_detector = FaceDetector()
eye_detector = EyeDetector()

webcam.start()

start_time = time.perf_counter()

print(webcam.camera.isOpened())


while True:

    frame = webcam.read_frame()

    if frame is None:
        break

    timestamp = int(
        (time.perf_counter() - start_time) * 1000
    )

    landmarks = face_detector.detect(frame, timestamp)

    if landmarks:

        face = landmarks[0]

        # Extract right eye landmarks
        right_eye = eye_detector.extract_eye_points(
            face,
            eye_detector.right_eye_indices
        )

        # Extract left eye landmarks
        left_eye = eye_detector.extract_eye_points(
            face,
            eye_detector.left_eye_indices
        )

        # Calculate EAR
        right_ear = eye_detector.calculate_ear(right_eye)
        left_ear = eye_detector.calculate_ear(left_eye)

        # Calculate average EAR
        ear = (right_ear + left_ear) / 2

        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw facial landmarks
        for landmark in face:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


webcam.stop()
face_detector.detector.close()
cv2.destroyAllWindows()
