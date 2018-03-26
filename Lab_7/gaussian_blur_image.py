"""
	gaussian_blur_image.py
	
	This code will use the function gaussianblur(src, ksize, sigmaX)
	to blur an image with an intesity of the gaussan kernel, which
	is affected by an equation where the sigma affects the blurriness.
	When only the x sigma is specified, the y sigma is given the same
	value. Same parameters apply as in blur() except the sigma. If both 
	sigmas are zeros, they are computed from ksize.width and ksize.heigh.

	author: Miguel Benavides, Laura Martinez 
	date created: 26 Marz 2018
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# read image
img_name = 'cavalo_motorizado.jpg'
img = cv2.imread(img_name)

# verify that image `img` exist
if img is None:
	print('ERROR: image ', img_name, 'could not be read')
	exit()

# blur image using `cv2.blur()`
kernel_size = (11,11)
blurred_image = cv2.GaussianBlur(img, kernel_size, 0)

# plot input and blurred images
plt.figure(1)
plt.imshow(img)
plt.title('Input image')
plt.xticks([]) 
plt.yticks([])

plt.figure(2)
plt.imshow(blurred_image)
plt.title('Output image using cv2.blur(%i,%i)' % (kernel_size[0], kernel_size[1]))
plt.xticks([]) 
plt.yticks([])

plt.show()
