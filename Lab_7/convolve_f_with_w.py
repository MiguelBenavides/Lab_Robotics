"""
	convolve_f_with_w.py
	
	This code demonstrates the correct funcionality of
	convolution and correlation bia a simple example
	of an array f and a kernel w

	author: Miguel Benavides, Laura Martinez 
	date created: 23 Marz 2018
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# define function `f`
f = np.array([[0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0], 
	      [0, 0, 1, 0, 0], 
              [0, 0, 0, 0, 0],
	      [0, 0, 0, 0, 0]], np.float32)

# define a 5x5 kernel
kernel = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]], np.float32)

# compute correlation
correlation = cv2.filter2D(f, -1, kernel)

# re-rotate the kernel `w` by 180 deg using cv2.flip()
kernel_rotated = cv2.flip(kernel, -1)

# conpute convolution
convolution = cv2.filter2D(f, -1, kernel_rotated)

# display correlation values
print('\nCorrelation:\n', correlation)

# display convolution values
print('\nConvolution:\n', convolution)
