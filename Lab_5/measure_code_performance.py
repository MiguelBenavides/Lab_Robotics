"""
	measure_code_performance.py
	
	The code below shows you how to measure the performance of 
	the code used for image properties printing and image 
	visualisation in Lab 4.

	author: Miguel Benavides 
	date created: 3 Marz 2018
	universidad de monterrey
"""

# import required libraries
import numpy as numpy
import matplotlib.pyplot as plt
import cv2

# print image statistics
def print_image_statistics(img, head_string):

	"""
		This function generates the statistics for a given 
		image and plots the figure
	"""
	# add your code below this line
	
	# get image size
	img_size = img.shape

	# print image size
	print('\nCOLOUR IMAGE STATS:')
	print('colour image size: ', img_size)

	# retrieve image width resolution
	print('image width resolution: ', img_size[0])

	# retrieve image height resolution
	print('image height resolution: ', img_size[1])

	# retrieve number of channels
	if len(img_size) < 3:
		print('number of channels: ', 1)
	else:
		print('number of channels: ', img_size[2])

	# minimum pixel value in image
	print('minimum intensity value: ', img.min())

	# minimum pixel value in image
	print('max intensity value: ', img.max())

	# maximum intensity value in image
	print('meam intensity value: ', img.mean())

	# print type of image
	print('type of image: ', img.dtype)
	
	return None

# visualise image
def visualise_image(img, fig_number, fig_title, iscolour):

	"""
		This code lets you visualize the image generated
	"""
	# add your code below this line

	# visualise colour image
	plt.figure(fig_number)
	if iscolour == 0:
		plt.imshow(img, cmap='gray')
	else:
		plt.imshow(img)
	plt.title(fig_title)
	plt.xlabel('x-resolution')
	plt.ylabel('y-resolution')
	
	return None

# read image
image_name = 'vehicular_traffic.jpg'
img_colour = cv2.imread(image_name, cv2.IMREAD_COLOR)

# verify that image exists
if img_colour is None:
	print('ERROR: image ', image_name, 'could not be read')
	exit()

# convert from BGR to RGB so that the image can be visualised using matplotlib
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# convert the input colour image into a grayscale image
e1 = cv2.getTickCount()
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nConversion from colour to greyscale took: ', time, 'seconds')

# print colour image stats and visualise it
e1 = cv2.getTickCount()
print_image_statistics(img_colour, 'COLOUR IMAGE STATS:')
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nCODE PERFORMANCE:')
print('printing colour image stats took: ', time, 'seconds')

e1 = cv2.getTickCount()
visualise_image(img_colour, 1, 'INPUT IMAGE: COLOUR', 1)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('visualising colour image took: ', time, 'seconds')

# print greyscale image stats and visualise it
e1 = cv2.getTickCount()
print_image_statistics(img_greyscale, 'GREYSCALE IMAGE STATS:')
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('\nCODE PERFORMANCE:')
print('Printing greyscale image stats took: ', time, 'seconds')

e1 = cv2.getTickCount()
visualise_image(img_greyscale, 2, 'OUTPUT IMAGE: GREYSCALE', 0)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('Visualising grey colour image took: ', time, 'seconds')

# visualise figures
plt.show()
