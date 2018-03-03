"""
	blend_images.py
	
	This code lets you blend two images instead of adding them, 
	trough an equation img = a * img1 + b* img2 + c, 
	using cv.addWeighted().

	author: Miguel Benavides 
	date created: 3 Marz 2018
	universidad de monterrey
"""

# import required libraries
import numpy as numpy
import matplotlib.pyplot as plt
import cv2

# specify names of image files
image_name_1 = 'salinas_beach_queijo_asado.jpg'
image_name_2 = 'salinas_beach_bird.jpg'

# load image 1
img_1 = cv2.imread(image_name_1, cv2.IMREAD_COLOR)

# verify that img_1 exists
if img_1 is None:
	print('ERROR: image ', image_name_1, 'could not be read')
	exit()

# convert BGR to RGB
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

# load image 2
img_2 = cv2.imread(image_name_2, cv2.IMREAD_COLOR)

# verify that img_1 exists
if img_2 is None:
	print('ERROR: image ', image_name_2, 'could not be read')
	exit()

# convert BGR to RGB
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

# blend images
img_blended = cv2.addWeighted(img_1, 0.3, img_2, 0.7, 0)

# visualise images
plt.figure(1)
plt.imshow(img_1)
plt.title('Image 1')

plt.figure(2)
plt.imshow(img_2)
plt.title('Image 2')

plt.figure(3)
plt.imshow(img_blended)
plt.title('Image blended using Image 1 and Image 2')

# display figures
plt.show()
