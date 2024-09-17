import numpy as np
import cv2
import math

cap = cv2.VideoCapture(1)

while True: 
    ret, frame = cap.read()
    if not ret: 
        break

    #Get frame dimensions
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Define color range for blue in HSV
    lower_blue = np.array([70, 100, 100])
    upper_blue = np.array([100, 250,250 ])

    #Create a mask to isolate blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Apply mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #Find contours in the mask 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    for contour in contours:
        #Calculate moments of the contour
        M = cv2.moments(contour)

        #Calculate the centroid of the contour
        if M['m00'] !=0:
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            centroids.append((cX, cY))

            #Put centroid coordinates on the frame
            cv2.circle(result, (cX, cY), 7, (255, 0, 0), -1)

            # Extract the HSV value at the centroid
            hsv_value = hsv[cY, cX]
            h, s, v = hsv_value
            
            # Draw the HSV value on the frame
            text = f"HSV: ({h}, {s}, {v})"
            # cv2.putText(result, text, (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if len(centroids) >= 2:
                # Compute the distance between the first two centroids
                (x1, y1) = centroids[0]
                (x2, y2) = centroids[1]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Draw the distance on the frame
                text_distance = f"Distance: {distance:.2f}"
                midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.putText(result, text_distance, (midpoint[0] + 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Draw a line between the two centroids
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('frame', result)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()