import numpy as np
import cv2
import time
from mqtt_publisher import MqttPublisher
from camera_handler import CameraHandler
from color_processor import ColorProcessor
from distance_calculator import DistanceCalculator
from visualizer import Visualizer
from calibration import ImageCalibration

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
    return color_name

def main():
    try:
        color = get_inputs()
        lower_color, upper_color = get_color_ranges(color)
        calibration = ImageCalibration(lower_color, upper_color)
        calibration.run()

        # Initialize components
        mqtt_publisher = MqttPublisher("10.243.85.48", 1885, "vscode")  # Ensure topic is correct
        camera_handler = CameraHandler(camera_index=1)
        color_processor = ColorProcessor(lower_color, upper_color)
        distance_calculator = DistanceCalculator(
            closed_distances=calibration.closed_distances,
            open_distances=calibration.open_distances
        )
        visualizer = Visualizer()

        mqtt_publisher.connect()

        # Main loop
        while True:
            frame = camera_handler.capture_frame()
            if frame is None:
                continue

            result, mask = color_processor.process_frame(frame)
            centroids = color_processor.find_centroids(mask)

            if len(centroids) >= 2:
                point1, point2 = centroids[:2]
                distance = distance_calculator.calculate_distance(point1, point2)
                normalized_distance = distance_calculator.normalize()

                if normalized_distance is not None:
                    mqtt_publisher.publish("mouth", int(normalized_distance))
                    time.sleep(0.3)

                visualizer.draw_distance(result, point1, point2, distance)

            visualizer.draw_centroids(result, centroids)
            visualizer.display_frame("Visualizer", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera_handler.release()
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
