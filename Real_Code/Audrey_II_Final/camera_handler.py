import cv2

class CameraHandler:
    def __init__(self, camera_index=1):
        #Initializes CameraHandler
        #My camera index was 1 for front facing laptop camera

        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            return None
        return frame
    
    def release_camera(self):

        if self.cap.isOpened():
            self.cap.release()
            print("Camera released")