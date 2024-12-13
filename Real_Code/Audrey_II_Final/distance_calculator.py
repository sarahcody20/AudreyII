import math

class DistanceCalculator():
    def __init__(self, closed_distances, open_distances):
    
        #Initizalies DistanceCalculator with calibration distances

        if not closed_distances or not open_distances:
            raise ValueError("Cannot access calibration distances")
        
        self.closed_distances = closed_distances
        self.open_distances = open_distances
        self.active_distance = None

    def calculate_distance(self, point1, point2):

        #Calculates distance between 2 points

        (x1, y1) = point1
        (x2, y2) = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.active_distance = distance
        return distance
    
    def normalize(self):

        #Normalizes the active distances betwen 0 (closed) and 100 (open)

        if self.active_distance is None:
            print("Active distance not set")
            return None
        
        x0 = self.closed_distances[0]
        x1 = self.open_distances[0]
        x = self.active_distance

        if x1 == x0:
            print("Error: Calibration distances are identical, normalization impossible.")
            return None

        gripper_distance = ((x - x0) / (x1 - x0)) * 100
        gripper_distance = max(0, min(gripper_distance, 100))
        return 100 - gripper_distance