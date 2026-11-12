
import numpy as np


class MouthDetector:

    def __init__(self):

        # Mouth landmark indices
        self.mouth_indices = [
            61,   # Left mouth corner
            291,  # Right mouth corner
            13,   # Upper inner lip
            14    # Lower inner lip
        ]

    def extract_mouth_points(self, face_landmarks):

        mouth_points = []

        for index in self.mouth_indices:

            landmark = face_landmarks[index]

            mouth_points.append([
                landmark.x,
                landmark.y
            ])

        return np.array(mouth_points)

    def euclidean_distance(self, point1, point2):

        return np.linalg.norm(point1 - point2)

    def calculate_mar(self, mouth_points):

        left_corner = mouth_points[0]
        right_corner = mouth_points[1]
        upper_lip = mouth_points[2]
        lower_lip = mouth_points[3]

        vertical_distance = self.euclidean_distance(
            upper_lip,
            lower_lip
        )

        horizontal_distance = self.euclidean_distance(
            left_corner,
            right_corner
        )

        mar = vertical_distance / horizontal_distance

        return mar
