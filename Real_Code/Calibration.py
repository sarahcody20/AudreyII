import numpy as np
import cv2 


class ImageCalibration:
    def __init__(self, lower_color, upper_color):
        #Initalize with specific color ranges
        self.lower_color = lower_color
        self.upper_color = upper_color

        #Read in images (MUST be closed.jpg and open.jpg)
        self.closed_image = cv2.imread( r'C:\Users\sarah\VSCode Projects\AudreyII\Real_Code\closed.jpg')
        self.open_image = cv2.imread(r'C:\Users\sarah\VSCode Projects\AudreyII\Real_Code\open.jpg')

        if self.closed_image is None:
            print(f"Error: Could not read image from {closed_file}")
        if self.open_image is None:
            print(f"Error: Could not read image from {open_file}")

    def process_image(self)
        self.closed_hsv = cv2.cvtColor(closed_image, cv2.COLOR_BGR2HSV)
        self.open_hsv = cv2.cvtColor(open_image, cv2.COLOR_BGR2HSV)


