#Via Tech With Tim
import numpy as np
import cv2

#0 gives back camera, 1 gives front
cap = cv2.VideoCapture(1)

while True:
    #Returns the frame (image), and ret (aka did this work properly)
    ret, frame = cap.read()

    width =int(cap.get(3))
    height =int(cap.get(4))

    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    image[:height//2,:width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) #Top left
    image[height//2:,:width//2] = smaller_frame #Bottom left
    image[:height//2,width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180) #Top Right
    image[height//2:,width//2:] = smaller_frame #Bottom Right

    cv2.imshow('frame', image)

    #cv2.waitKey will wait 1 ms and move on unless you press q
    #ord means ASCII value
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    
