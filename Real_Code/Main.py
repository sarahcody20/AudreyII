import numpy as np
from Color_Tracker import ColorTracker
from Calibration import ImageCalibration

def get_color_ranges(color_name):
    color_ranges = {
        'blue': (np.array([70, 90, 90]), np.array([120, 255, 255])),
        'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
        # Add other colors as needed
    }
    
    if color_name in color_ranges:
        return color_ranges[color_name]
    else:
        raise ValueError(f"Color '{color_name}' not recognized. Available colors: {list(color_ranges.keys())}")

def get_inputs():
    color_name = input("Color Name: ")  

    cap = cv2.VideoCapture(1)
    width =int(cap.get(3))
    height =int(cap.get(4))
    return color_name, width, height

def main():
    # Get inputs
    color = get_inputs()
    print(color, width, height)

    # Get the color ranges
    lower_color, upper_color = get_color_ranges(color)

    #Run the calibration to get baseline distances
    calibration = ImageCalibration(lower_color, upper_color, video_width, video_height)
    calibration.run()

    closed_distances = calibrator.closed_distances
    open_distances = calibrator.open_distances


    # Initialize the ColorTracker with the color ranges
    tracker = ColorTracker(lower_color, upper_color)

    # Run the tracker
    tracker.run()

if __name__ == "__main__":
    main()