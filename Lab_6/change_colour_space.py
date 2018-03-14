"""
	change_colour_space.py
	
	This code segments blue color objects. Then makes an 
	AND-bitwise operation between the mask and input images.

	author: Miguel Benavides, Laura Morales
	date created: 13 Marz 2018
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# initialise a video capture object
cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
	print('Unable to open the camera')
	exit()

while(True):

	# grab current frame
	cf, frame = cap.read()

	# convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


	# ----- Tune these parameters so that blue-colour  ------ #
	# ----- objects can be detected                    ------ #
	h_val_l = 80
	h_val_h = 120
	s_val_l = 100
	v_val_l = 100
	lower_blue = np.array([h_val_l,s_val_l, v_val_l])
	upper_blue = np.array([h_val_h, 255, 255])
	# ------------------------------------------------------- #

	# ----- Tune these parameters so that red-colour   ------ #
	# ----- objects can be detected                    ------ #
	h_val_l = 160
	h_val_h = 200
	s_val_l = 100
	v_val_l = 100
	lower_red = np.array([h_val_l,s_val_l, v_val_l])
	upper_red = np.array([h_val_h, 255, 255])
	# ------------------------------------------------------- #

	# ----- Tune these parameters so that yellow-colour------ #
	# ----- objects can be detected                    ------ #
	h_val_l = 10
	h_val_h = 50
	s_val_l = 100
	v_val_l = 100
	lower_yellow = np.array([h_val_l,s_val_l, v_val_l])
	upper_yellow = np.array([h_val_h, 255, 255])
	# ------------------------------------------------------- #

	# ----- Tune these parameters so that violet-colour------ #
	# ----- objects can be detected                    ------ #
	h_val_l = 130
	h_val_h = 170
	s_val_l = 100
	v_val_l = 100
	lower_violet = np.array([h_val_l,s_val_l, v_val_l])
	upper_violet = np.array([h_val_h, 255, 255])
	# ------------------------------------------------------- #


	# threshold the hsv image so that only the respective colour pixels are kept
	maskblue = cv2.inRange(hsv, lower_blue, upper_blue)
	maskred = cv2.inRange(hsv, lower_red, upper_red)
	maskyellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	maskviolet = cv2.inRange(hsv, lower_violet, upper_violet)

	# AND-bitwise operation between the mask and input images
	blue_object_img = cv2.bitwise_and(frame, frame, mask=maskblue)
	red_object_img = cv2.bitwise_and(frame, frame, mask=maskred)
	yellow_object_img = cv2.bitwise_and(frame, frame, mask=maskyellow)
	violet_object_img = cv2.bitwise_and(frame, frame, mask=maskviolet)

	# ADD operation between the 4 colours images
	multicolour_object_img = cv2.add(blue_object_img, red_object_img)
	multicolour_object_img = cv2.add(multicolour_object_img, yellow_object_img)
	multicolour_object_img = cv2.add(multicolour_object_img, violet_object_img)

	# visualise current frame
	cv2.imshow('frame',frame)

	# visualise mask image
	cv2.imshow('maskblue', maskblue)
	cv2.imshow('maskred', maskred)
	cv2.imshow('maskyellow', maskyellow)
	cv2.imshow('maskviolet', maskviolet)

	# visualise segmented blue object
	cv2.imshow('blue object', blue_object_img)
	cv2.imshow('red object', red_object_img)
	cv2.imshow('yellow object', yellow_object_img)
	cv2.imshow('violet object', violet_object_img)
	cv2.imshow('multicolour object', multicolour_object_img)

	# Display the resulting frame
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
