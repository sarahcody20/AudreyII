import numpy as np
import cv2
import math

class ColorTracker:
    def __init__(self, lower_color, upper_color):
        """
        Initialize the ColorTracker with specific color ranges.

        Parameters:
        lower_color (np.array): Lower HSV bound for the color.
        upper_color (np.array): Upper HSV bound for the color.
        """
        self.lower_color = lower_color
        self.upper_color = upper_color

        # Initialize video capture
        self.cap = cv2.VideoCapture(1)

    def capture_frame(self):
        """Capture a video frame."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def process_frame(self, frame):
        """Convert frame to HSV and create a mask based on color range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask

    def find_centroids(self, mask, thresh_area=500):
        """Find centroids of the objects in the mask."""
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
                else:
                    centroids.append((0, 0))
        return centroids

    def draw_results(self, result, centroids):
        """Draw results on the frame, including distance and lines between centroids."""
        if len(centroids) >= 2:
            (x1, y1) = centroids[0]
            (x2, y2) = centroids[1]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Draw the distance on the frame
            text_distance = f"Distance: {distance:.2f}"
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(result, text_distance, (midpoint[0] + 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw a line between the two centroids
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def write_hsv(self, result, centroids, hsv_frame):
        """Write HSV values of the centroids on the frame."""
        for (x, y) in centroids:
            # Ensure centroid is within the image bounds
            if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                hsv_values = hsv_frame[y, x]
                hsv_text = f"H: {hsv_values[0]}, S: {hsv_values[1]}, V: {hsv_values[2]}"
                # Display the HSV values at the centroid
                cv2.putText(result, hsv_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def run(self):
        """Start the tracking process."""
        while True:
            frame = self.capture_frame()
            if frame is None:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            result, mask = self.process_frame(frame)
            centroids = self.find_centroids(mask)

            self.draw_results(result, centroids)
            self.write_hsv(result, centroids, hsv_frame)

            cv2.imshow('frame', result)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
