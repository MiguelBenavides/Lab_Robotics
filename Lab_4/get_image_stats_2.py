"""
	get_image_stats_2.py
	
	This code lets you have two function definitions, instead of one
	as the previous code, meaning one for image statistics and another
	for image visualisation.

	author: Miguel Benavides 
	date created: 28-Feb-2018
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
img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

# verify that image exists
if img_colour is None:
	print('ERROR: image ', image_name, 'could not be read')
	exit()

# convert the input colour image into a grayscale image
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)

# print colour image stats and visualise it
print_image_statistics(img_colour, 'COLOUR IMAGE STATS:')
visualise_image(img_colour, 1, 'INPUT IMAGE: COLOUR', 1)

# print greyscale image stats and visualise it
print_image_statistics(img_greyscale, 'GREYSCALE IMAGE STATS:')
visualise_image(img_greyscale, 2, 'OUTPUT IMAGE: GREYSCALE', 0)

# visualise figures
plt.show()
