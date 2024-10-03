import numpy as np
import cv2
from Color_Tracker import ColorTracker
from Calibration import ImageCalibration

def get_color_ranges(color_name):
    color_ranges = {
        'blue': (np.array([70, 90, 90]), np.array([120, 255, 255])),
        'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
    }

    if color_name in color_ranges:
        return color_ranges[color_name]
    else:
        raise ValueError(f"Color '{color_name}' not recognized. Available colors: {list(color_ranges.keys())}")

def get_inputs():
    color_name = input("Color Name (blue/red): ").strip().lower()
    cap = cv2.VideoCapture(1)
    cap.release()
    return color_name

def main():
    try:
        color = get_inputs()
        lower_color, upper_color = get_color_ranges(color)

        # Run the calibration to get baseline distances
        calibration = ImageCalibration(lower_color, upper_color)
        calibration.run()

        # Access the instance variables directly
        tracker = ColorTracker(lower_color, upper_color, calibration.closed_distances, calibration.open_distances)
        tracker.run()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()