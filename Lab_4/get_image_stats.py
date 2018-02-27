"""
	get_image_stats.py

	This code lets you retrieve the dimensions of an image,
	meaning height, width, and number of channels.

	author: Miguel Benavides, Laura Morales
	date created: 27 February 2018
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

# -------------------- PROCESS COLOUR IMAGE -------------------- #

# get image size
img_size = img_colour.shape

# print image size
print('\nCOLOUR IMAGE STATS:')
print('colour image size: ', img_size)

# retrieve image width resolution
print('image width resolution: ', img_size[0])

# retrieve image height resolution
print('image height resolution: ', img_size[1])

# retrieve number of channels
print('number of channels: ', img_size[2])

# minimum pixel value in image
print('minimum intensity value: ', img_colour.min())

# minimum pixel value in image
print('max intensity value: ', img_colour.max())

# maximum intensity value in image
print('meam intensity value: ', img_colour.mean())

# print type of image
print('type of image: ', img_colour.dtype)

# visualise colour image
plt.figure(1)
plt.imshow(img_colour)
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')


# ------------------ PROCESS GREYSCALE IMAGE ------------------- #

# convert the input colour image into a grayscale image
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)

# get greyscale image size
img_size = img_greyscale.shape

# retrieve size of greyscale image
print('\nGREYSCALE IMAGE STATS:')
print('greyscale image size', img_size)

# retrieve image width resolution
print('greyscale image width resolution: ', img_size[0])

# retrieve image height resolution
print('greyscale image height resolution: ', img_size[1])

# retrieve number of channels
print('number of channels: ', 1)

# minimum pixel value in image
print('minimum intensity value: ', img_greyscale.min())

# minimum pixel value in image
print('max intensity value: ', img_greyscale.max())

# maximum intensity value in image
print('meam intensity value: ', img_greyscale.mean())

# print type of image
print('type of image: ', img_greyscale.dtype)

# visualise image using matplotlib
plt.figure(2)
plt.imshow(img_greyscale, cmap='gray')
plt.title('OUTPUT IMAGE: GREYSCALE')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# display both windows with the colour and greyscale images
plt.show()
