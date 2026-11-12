
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.model_path = "models/face_landmarker.task"

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.detector.detect_for_video(
            mp_image,
            timestamp
        )

        return results.face_landmarks
