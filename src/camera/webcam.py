import cv2 

class Webcam:
    def __init__(self):
        self.camera_index = 0
        self.camera = None

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION) # only for macOS

    def read_frame(self):
        success, frame = self.camera.read()

        if not success :
            return None

        return frame
    def stop(self):
        self.camera.release()

