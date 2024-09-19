import numpy as np
import cv2

# Define wide range of color you are looking for in HSV
lower_color = np.array([70, 90, 90])
upper_color = np.array([120, 250, 255])

# Initialize video capture
cap = cv2.VideoCapture(1)

while True:
    # Capture video frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    # Turn frame to HSV and create a mask based on colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find the largest unmasked objects
    centroids = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: 
        area = cv2.contourArea(contour)
        if area > 500:  # Can change this if necessary
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centroids.append((cX, cY))

    # Write HSV values on centroids
    for (x, y) in centroids:
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            hsv_values = hsv[y, x]
            hsv_text = f"H: {hsv_values[0]}, S: {hsv_values[1]}, V: {hsv_values[2]}"
            cv2.putText(result, hsv_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('HSV Detector', result)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
