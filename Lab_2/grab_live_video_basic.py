# import required libraries
import numpy as np
import cv2 as cv

# create a VideoCapture object
cap = cv.VideoCapture(0)

# main loop
while(True):

    # capture new frame
    ret, frame = cap.read()

    # convert from colour to grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # visualise image
    cv.imshow('frame', frame)

    # wait for the user to press 'q' to close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture object
cap.release()

# destroy windows to free memory
cv.destroyAllWindows()
