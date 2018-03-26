"""
	convolve_image_with_kernel.py
	
	This code correlates and convolves an image 
	with a 5x5 kernel of ones, making a blurred image as a result.

	cv2.flip() = this function flips an array (an image is 
	also an array of pixel intensity values) by the means of
	0 being horizontal flip, 1 being vertical flip, and 
	-1 being on both axis.

	cv2.filter2D() = this function convolves an image with
	the kernel, the first parameter src is the image or input
	array, the second parameter dst is the output image
	or output array (this one has a parameter ddepth to be 
	specified, in our case we want the output image to be the same
	depth as the RGB input image so we use -1), and the last 
	parameter kernel which is the convolution kernel.

	As this code smooths the image by replacing the center
	pixel with the average of its neightborhoood pixels (in a 
	5x5 array), replacing the center pixel with a 1x1 array 
	average will result in the same image as the input.

	Changing the kernel size will result on a different level 
	of blurriness. If the kernel is small, the image will blurr
	a little, and viceversa.

	The weighted average filter is not as heavy on hard edges,
	as the average of the image is severely affected by the 
	center pixel, and soft edges will be blurred more if it is 
	surrounded by hard edges.

	author: Miguel Benavides, Laura Martinez
	date created: 23 Marz 2018
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

# define a 5x5 kernel
# kernel = np.ones((31,31), np.float32)/961
kernel = np.array([[1, 2, 1], 
                   [2, 4, 2], 
                   [1, 2, 1]], np.float32)/16

dst_correlation = cv2.filter2D(img, -1, kernel)

# rotate kernel
kernel_rotated = cv2.flip(kernel, -1)
dst_convolution = cv2.filter2D(img, -1, kernel_rotated)

# plot input and convolved images
plt.figure(1)
plt.imshow(img)
plt.title('Input image')
plt.xticks([]) 
plt.yticks([])

plt.figure(2)
plt.imshow(dst_correlation)
plt.title('Output image using a 5x5 averaging filter (correlation)')
plt.xticks([]) 
plt.yticks([])

plt.figure(3)
plt.imshow(dst_convolution)
plt.title('Output image using a 5x5 averaging filter (convolution)')
plt.xticks([]) 
plt.yticks([])

plt.show()


