import numpy as np
from Color_Tracker import ColorTracker

def get_color_ranges(color_name):
    """
    Return the HSV color ranges for a given color name.

    Parameters:
    color_name (str): Name of the color.

    Returns:
    tuple: (lower_color, upper_color)
    """
    color_ranges = {
        'blue': (np.array([70, 90, 90]), np.array([120, 250, 250])),
        'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
        # Add more colors as needed
    }
    
    if color_name in color_ranges:
        return color_ranges[color_name]
    else:
        raise ValueError(f"Color '{color_name}' not recognized. Available colors: {list(color_ranges.keys())}")

def main():
    # Define the color to track
    color_name = 'blue'  # Change to 'red' or other color as needed

    # Get the color ranges
    lower_color, upper_color = get_color_ranges(color_name)

    # Initialize the ColorTracker with the color ranges
    tracker = ColorTracker(lower_color, upper_color)

    # Run the tracker
    tracker.run()

if __name__ == "__main__":
    main()
