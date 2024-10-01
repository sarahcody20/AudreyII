import numpy as np
import cv2
import math
import threading

class ColorTracker:
    def __init__(self, lower_color, upper_color):
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.cap = cv2.VideoCapture(1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.active_distance = None
        self.running = True  # Control variable for the thread

    def capture_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask

    def find_centroids(self, mask, thresh_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > thresh_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    centroids.append((cX, cY))
        return centroids

    def draw_results(self, result, centroids):
        distance = None
        if len(centroids) >= 2:
            (x1, y1) = centroids[0]
            (x2, y2) = centroids[1]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            text_distance = f"Distance: {distance:.2f}"
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(result, text_distance, (midpoint[0] + 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return distance

    def run(self):
        while self.running:
            frame = self.capture_frame()
            if frame is None:
                continue
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            result, mask = self.process_frame(frame)
            centroids = self.find_centroids(mask)

            # Get the distance from draw_results
            self.active_distance = self.draw_results(result, centroids)

            cv2.imshow('frame', result)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
