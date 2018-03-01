# import required libraries
import numpy as np
import cv2
import argparse

# parse command line arguments
parser = argparse.ArgumentParser('Read, visualise and write image into disk')
parser.add_argument('-i', '--in_image_name', help='input image name', required=True)
parser.add_argument('-u', '--inputcolourspace', help='input grayscale', required=True)
parser.add_argument('-o', '--out_image_name', help='output image name', required=True)
args = vars(parser.parse_args())

# retrieve name of input and output images given as arguments from command line
img_in_name = args['in_image_name']
inputcolourspace = args['inputcolourspace']
img_out_name= args['out_image_name']

# read in image from file
if inputcolourspace=="grayscale":
	img_in = cv2.imread(img_in_name, cv2.IMREAD_GRAYSCALE) # alternatively, you can use cv2.IMREAD_GRAYSCALE
	img_out = img_in;
else:
	img_in = cv2.imread(img_in_name, cv2.IMREAD_COLOR)
	# convert input image from colour to grayscale
	img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# verify that image exists
if img_in is None:
    print('ERROR: image ', img_in_name, 'could not be read')
    exit()

# create a new window for image purposes
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL
cv2.namedWindow("output image", cv2.WINDOW_AUTOSIZE) # that option will allow you for window resizing

# visualise input and output image
cv2.imshow("input image", img_in)
cv2.imshow("output image", img_out)

# wait for the user to press a key
key = cv2.waitKey(0)

# if user pressed 's', the grayscale image is write to disk
if key == ord("s"):
    cv2.imwrite(img_out_name, img_out)
    print('output image has been saved in current working directory')

# destroy windows to free memory
cv2.destroyAllWindows()
print('windows have been closed properly')
exit()
