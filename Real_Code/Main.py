import numpy as np
import cv2
from Color_Tracker import ColorTracker
from Calibration import ImageCalibration
import threading
import time

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
    color_name = input("Color Name: ")  
    cap = cv2.VideoCapture(1)
    width = int(cap.get(3))
    height = int(cap.get(4))
    cap.release()  # Release the video capture object
    return color_name, width, height

def main():
    color, width, height = get_inputs()
    lower_color, upper_color = get_color_ranges(color)

    # Run the calibration to get baseline distances
    calibration = ImageCalibration(lower_color, upper_color, width, height)
    calibration.run()

    closed_distances = calibration.closed_distances
    open_distances = calibration.open_distances

    # Initialize the ColorTracker with the color ranges
    tracker = ColorTracker(lower_color, upper_color)

    # Start the tracker in a separate thread
    tracker_thread = threading.Thread(target=tracker.run)
    tracker_thread.start()

    # Continuously print active distance
    try:
        while True:
            time.sleep(1)  # Adjust the frequency of distance checks
            
            # Check if active_distance is valid
            if tracker.active_distance is not None:
                gripper_distance = (tracker.active_distance - closed_distances[0]) / (open_distances[0] - closed_distances[0])*100
                gripper_distance = max(0, min(100, gripper_distance))  # Clamp between 0 and 100
                print(f'Gripper Distance: {gripper_distance}')
            else:
                print('Active distance not yet available.')

    except KeyboardInterrupt:
        # Stop the tracker when the script is interrupted
        tracker.stop()
        tracker_thread.join()

if __name__ == "__main__":
    main()
