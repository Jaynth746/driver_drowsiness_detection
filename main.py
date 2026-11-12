
import cv2
import time

from src.camera.webcam import Webcam
from src.detection.face_detector import FaceDetector
from src.detection.eye_detector import EyeDetector
from src.detection.mouth_detector import MouthDetector


# Thresholds
EAR_THRESHOLD = 0.430
DROWSINESS_THRESHOLD = 2.0

# MAR threshold (initial provisional value)
MAR_THRESHOLD = 0.60

# Yawning duration threshold
YAWN_DURATION_THRESHOLD = 2.0


# Initialize objects
webcam = Webcam()
face_detector = FaceDetector()
eye_detector = EyeDetector()
mouth_detector = MouthDetector()


# Eye closure timer
eye_closed_start = None
eye_closed_duration = 0.0

# Yawning timer
yawn_start = None
yawn_duration = 0.0

# Drowsiness status
drowsiness_status = "NORMAL"


# Start webcam
webcam.start()

start_time = time.perf_counter()

print(webcam.camera.isOpened())


while True:

    frame = webcam.read_frame()

    if frame is None:
        break

    # Calculate timestamp
    timestamp = int(
        (time.perf_counter() - start_time) * 1000
    )

    # Detect face landmarks
    landmarks = face_detector.detect(frame, timestamp)

    if landmarks:

        face = landmarks[0]

        # ---------------- EAR CALCULATION ----------------

        right_eye = eye_detector.extract_eye_points(
            face,
            eye_detector.right_eye_indices
        )

        left_eye = eye_detector.extract_eye_points(
            face,
            eye_detector.left_eye_indices
        )

        right_ear = eye_detector.calculate_ear(right_eye)
        left_ear = eye_detector.calculate_ear(left_eye)

        ear = (right_ear + left_ear) / 2

        # ---------------- EYE CLOSURE DETECTION ----------------

        if ear < EAR_THRESHOLD:

            if eye_closed_start is None:

                eye_closed_start = time.perf_counter()

            eye_closed_duration = (
                time.perf_counter() - eye_closed_start
            )

            if eye_closed_duration >= DROWSINESS_THRESHOLD:

                drowsiness_status = "DROWSY"

            else:

                drowsiness_status = "NORMAL"

        else:

            eye_closed_start = None
            eye_closed_duration = 0.0

            drowsiness_status = "NORMAL"

        # ---------------- MAR CALCULATION ----------------

        mouth_points = mouth_detector.extract_mouth_points(face)

        mar = mouth_detector.calculate_mar(mouth_points)

        # ---------------- YAWNING DETECTION ----------------

        if mar >= MAR_THRESHOLD:

            if yawn_start is None:

                yawn_start = time.perf_counter()

            yawn_duration = time.perf_counter() - yawn_start

        else:

            yawn_start = None
            yawn_duration = 0.0

        if yawn_duration >= YAWN_DURATION_THRESHOLD:

            yawn_status = "YAWNING"

        else:

            yawn_status = "NORMAL"

        # ---------------- DISPLAY EAR ----------------

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ---------------- DISPLAY EYE STATUS ----------------

        cv2.putText(
            frame,
            f"Eyes: {'CLOSED' if ear < EAR_THRESHOLD else 'OPEN'}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0),
            2
        )

        # ---------------- DISPLAY EYE CLOSURE DURATION ----------------

        cv2.putText(
            frame,
            f"Closed: {eye_closed_duration:.2f}s",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ---------------- DISPLAY DROWSINESS STATUS ----------------

        cv2.putText(
            frame,
            f"Status: {drowsiness_status}",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if drowsiness_status == "DROWSY" else (0, 255, 0),
            2
        )

        # ---------------- DISPLAY MAR ----------------

        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        # ---------------- DISPLAY YAWN DURATION ----------------

        cv2.putText(
            frame,
            f"Yawn Duration: {yawn_duration:.2f}s",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # ---------------- DISPLAY YAWN ALERT ----------------

        if yawn_status == "YAWNING":

            cv2.putText(
                frame,
                "YAWNING DETECTED!",
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # ---------------- DRAW FACIAL LANDMARKS ----------------

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

        # ---------------- DRAW MOUTH LANDMARKS ----------------

        for point in mouth_points:

            x = int(point[0] * frame.shape[1])
            y = int(point[1] * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 0, 255),
                -1
            )

    else:

        # Reset eye timer
        eye_closed_start = None
        eye_closed_duration = 0.0

        # Reset yawning timer
        yawn_start = None
        yawn_duration = 0.0

        drowsiness_status = "NORMAL"

    # ---------------- DISPLAY WEBCAM ----------------

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# Stop webcam
webcam.stop()

# Close MediaPipe detector
face_detector.detector.close()

# Close OpenCV windows
cv2.destroyAllWindows()
