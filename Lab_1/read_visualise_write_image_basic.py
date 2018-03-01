# import required libraries
import numpy as np
import cv2

# read in image from file
img_in = cv2.imread('fortaleza_ceara_001.jpg', cv2.IMREAD_COLOR) # alternatively, you can use cv2.IMREAD_GRAYSCALE

# create a new window for image visualisation purposes
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL

# visualise input and output images
cv2.imshow("input image", img_in)

# wait for the user to press a key
key = cv2.waitKey(0)

# convert image to grayscale
img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# if user presses 's', the grayscale image is write to an image file
if key == ord("s"):
    cv2.imwrite('grayscale_image.png', img_out)
    print('output image has been saved in current working directory')

# destroy windows to free memory
cv2.destroyAllWindows()
print('...image window has been closed properly - bye!')
exit()
