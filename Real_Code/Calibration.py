import numpy as np
import cv2
import math

class ImageCalibration:
    def __init__(self, lower_color, upper_color):
        # Initialize with specific color ranges
        self.lower_color = lower_color
        self.upper_color = upper_color

        self.video_width = 480
        self.video_height = 640

        # Read in images (MUST be closed.jpg and open.jpg)
        self.closed_image = cv2.imread(r'C:\Users\sarah\VSCode Projects\AudreyII\Real_Code\closed.jpg')
        self.open_image = cv2.imread(r'C:\Users\sarah\VSCode Projects\AudreyII\Real_Code\open.jpg')

        if self.closed_image is None:
            print("Error: Could not read closed image. Is it called closed.jpg?")
        if self.open_image is None:
            print("Error: Could not read open image. Is it called open.jpg?")

        # Only resize if images are valid
        if self.closed_image is not None:
            self.closed_image = cv2.resize(self.closed_image, (self.video_width, self.video_height))
        if self.open_image is not None:
            self.open_image = cv2.resize(self.open_image, (self.video_width, self.video_height))

        # Initialize distance variables
        self.closed_distances = []
        self.open_distances = []

    def process_image(self):
        if self.closed_image is None or self.open_image is None:
            return None, None

        # Convert images into HSV
        closed_hsv = cv2.cvtColor(self.closed_image, cv2.COLOR_BGR2HSV)
        open_hsv = cv2.cvtColor(self.open_image, cv2.COLOR_BGR2HSV)

        # Create masks
        mask1 = cv2.inRange(closed_hsv, self.lower_color, self.upper_color)
        mask2 = cv2.inRange(open_hsv, self.lower_color, self.upper_color)

        return mask1, mask2

    def find_centroids(self, mask, thresh_area=500):
        if mask is None:
            return []

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

    def find_distance(self, centroids):
        distances = []
        if len(centroids) >= 2:
            (x1, y1) = centroids[0]
            (x2, y2) = centroids[1]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)

        return distances

    def run(self):
        # Process the images
        mask1, mask2 = self.process_image()

        if mask1 is None or mask2 is None:
            print("Error: Masks could not be generated due to missing images.")
            return

        # Find centroids
        closed_centroids = self.find_centroids(mask1)
        open_centroids = self.find_centroids(mask2)

        # Calculate distances
        self.closed_distances = self.find_distance(closed_centroids)
        self.open_distances = self.find_distance(open_centroids)

        # Print distances
        print("Closed Image Distances:", self.closed_distances)
        print("Open Image Distances:", self.open_distances)
