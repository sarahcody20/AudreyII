import numpy as np
import cv2
import math
import paho.mqtt.client as mqtt

class ColorTracker:
    def __init__(self, lower_color, upper_color, closed_distances, open_distances):
        self.cap = cv2.VideoCapture(1)
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.closed_distances = closed_distances
        self.open_distances = open_distances
        self.active_distance = None
        self.client = None  # Store MQTT client here

    def mqtt_connection(self):
        broker_address = "10.5.10.72"
        broker_port = 1885

        # Create a new MQTT client instance
        self.client = mqtt.Client("computer")

        # Connect to the broker
        self.client.connect(broker_address, broker_port)

        # Publish a test message
        self.client.publish("test", "mqtt connection working")
        print("message sent")

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
            self.active_distance = distance
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(result, text_distance, (midpoint[0] + 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def normalize(self):
        if self.active_distance is None:
            return None

        if not self.closed_distances or not self.open_distances:
            print("Error: Calibration distances not set properly.")
            return None

        x0 = self.closed_distances[0]
        x1 = self.open_distances[0]
        x = self.active_distance

        gripper_distance = ((x - x0) / (x1 - x0)) * 100
        gripper_distance = max(0, min(gripper_distance, 100))

        return gripper_distance

    def run(self):
        self.mqtt_connection()  # Establish MQTT connection before loop
        while True:
            frame = self.capture_frame()
            if frame is None:
                continue
            
            result, mask = self.process_frame(frame)
            centroids = self.find_centroids(mask)
            self.draw_results(result, centroids)

            # Optionally normalize active distance
            normalized_distance = self.normalize()
            if normalized_distance is not None:
                self.client.publish("test", normalized_distance)
                print(f"Normalized Distance: {normalized_distance:.2f}")

            cv2.imshow('Result', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
