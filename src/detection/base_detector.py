
class BaseDetector:

    def __init__(self):
        pass

    # Common landmark extraction method
    def extract_points(self, face_landmarks, landmark_indices):

        points = []

        for index in landmark_indices:

            landmark = face_landmarks[index]

            points.append([
                landmark.x,
                landmark.y
            ])

        return points
