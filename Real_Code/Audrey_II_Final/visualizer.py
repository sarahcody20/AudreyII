import cv2

class Visualizer:
    def __init__(self):
        pass

    def draw_centroids(self, frame, centroids):

        #Draws the centroids on a given frame
        for (x,y) in centroids:
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)

    def draw_distance(self, frame, point1, point2, distance):

        #Draws a line between the 2 centroids and displays calculated distance
        midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
        cv2.line(frame, point1, point2, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}", (midpoint[0] + 10, midpoint[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    def display_frame(self, window_name, frame):

        #Displays a frame in a window
        cv2.imshow(window_name, frame)

        