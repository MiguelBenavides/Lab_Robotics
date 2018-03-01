"""
	split_and_merge_image_channels.py
	
	This code splits channels of an image and merges them to form an RGB
	image, using cv2.split() and cv2.merge().

	author: Miguel Benavides Banda 
	date created: 1 May 2018
	universidad de monterrey
"""

# import required libraries
import numpy as numpy
import matplotlib.pyplot as plt
import cv2

# read image
image_name = 'vehicular_traffic.jpg'
img_colour = cv2.imread(image_name, cv2.IMREAD_COLOR)
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# verify that image exists
if img_colour is None:
	print('ERROR: image ', image_name, 'could not be read')
	exit()

# visualise image using matplotlib
plt.figure(1)
plt.imshow(img_colour)
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# split colour image channels
r_channel,g_channel,b_channel = cv2.split(img_colour)

# visualise each channel
plt.figure(2)
plt.imshow(r_channel,cmap='gray')
plt.title('RED channel')
plt.figure(3)
plt.imshow(g_channel,cmap='gray')
plt.title('GREEN channel')
plt.figure(4)
plt.imshow(b_channel,cmap='gray')
plt.title('BLUE channel')

# merge channels 
img_original = cv2.merge((r_channel, g_channel, b_channel)) 

plt.figure(5)
plt.imshow(img_original)
plt.title('Merged channels')

# set blue channel to zero
b_channel[:,:] = 0

# merge channels after setting blue channel to zero
img_bzero = cv2.merge((r_channel, g_channel, b_channel)) 

plt.figure(6)
plt.imshow(img_bzero)
plt.title('Merged channels after setting BLUE channel to zero')

# display figures
plt.show()
