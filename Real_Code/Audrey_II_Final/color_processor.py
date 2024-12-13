import cv2
import numpy as np

class ColorProcessor:
    def __init__(self, lower_color, upper_color):

        #Initialized ColorProcessor with color bounds
        #Numpy array
        self.lower_color = np.array(lower_color)
        self.upper_color = np.array(upper_color)

    def process_frame(self, frame):

        #Processes a frame to create a mask and result
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        results = cv2.bitwise_and(frame, frame, mask=mask)
        return results, mask

    def find_centroids(self, mask, thresh_area=500):

        #Finds the centroids of contours that are large enough to meet the threshold
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
    
    