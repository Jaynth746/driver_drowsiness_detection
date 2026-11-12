
import numpy as np


class EyeDetector:

    def __init__(self):

        self.right_eye_indices = [
            33, 160, 158, 133, 153, 144
        ]

        self.left_eye_indices = [
            362, 385, 387, 263, 373, 380
        ]

    def extract_eye_points(self, face_landmarks, eye_indices):

        eye_points = []

        for index in eye_indices:

            landmark = face_landmarks[index]

            eye_points.append([
                landmark.x,
                landmark.y
            ])

        return np.array(eye_points)

    def euclidean_distance(self, point1, point2):

        return np.linalg.norm(point1 - point2)

    def calculate_ear(self, eye_points):

        p1 = eye_points[0]
        p2 = eye_points[1]
        p3 = eye_points[2]
        p4 = eye_points[3]
        p5 = eye_points[4]
        p6 = eye_points[5]

        vertical_1 = self.euclidean_distance(p2, p6)
        vertical_2 = self.euclidean_distance(p3, p5)

        horizontal = self.euclidean_distance(p1, p4)

        ear = (
            vertical_1 + vertical_2
        ) / (2 * horizontal)

        return ear
