"""
	get_pixel_intensity_value.py

	This code reads an image on colour, then converts it to a 
	greyscale image. It retrieves the pixel intensity at the row 100
	column 100 by either image array method and item() method. Then, 
	it prints the information for both colour and greyscale image, 
	and modifies the intensity of that given pixel to 0, 0, 0, meaning
	black. It gives you the option to modify it bia array or item() methods. 

	author: Miguel Benavides
	date created: 28 February 2018
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

# convert the input colour image into a grayscale image
img_greyscale = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)

# visualise image using matplotlib
plt.figure(1)
plt.imshow(img_colour)
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# visualise image using matplotlib
plt.figure(2)
plt.imshow(img_greyscale, cmap='gray')
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# retrieve pixel intensity value at given row/col of colour image
row, col = 100, 100

# ----------------------- READ PIXEL VALUE ----------------------------- #
print('COLOUR IMAGE:')
print('\n\tpixel intensity value at', '(', row, ',', col, '): ', img_colour[row, col])
print('\n\tAccessing pixel values using image array')
print('\t\tRED pixel intensity value at', '(', row, ',', col, '): ', img_colour[row, col, 0])
print('\t\tGREEN pixel intensity value at', '(', row, ',', col, '): ', img_colour[row, col, 1])
print('\t\tBLUE pixel intensity value at', '(', row, ',', col, '): ', img_colour[row, col, 2])
print('\n\tAccessing pixel values using pre-built method item()')
print('\t\tRED pixel intensity value at', '(', row, ',', col, '): ', img_colour.item(row, col, 0))
print('\t\tGREEN pixel intensity value at', '(', row, ',', col, '): ', img_colour.item(row, col, 1))
print('\t\tBLUE pixel intensity value at', '(', row, ',', col, '): ', img_colour.item(row, col, 2))

print('\nGREYSCALE IMAGE:')
print('\n\tpixel intensity value at', '(', row, ',', col, '): ', img_greyscale[row, col])
print('\n\tAccessing pixel values using image array')
print('\t\tgrey pixel intensity value at', '(', row, ',', col, '): ', img_greyscale[row, col])
print('\n\tAccessing pixel values using pre-built method item()')
print('\t\tgrey pixel intensity value at', '(', row, ',', col, '): ', img_greyscale.item(row, col))

# -------------------- MODIFY PIXEL VALUE ------------------------------ #
# modify pixel intensity values in colour image
img_colour[row, col] = [0, 0, 0]

# uncomment the lines below if you want to modify the pixel value using the pre-built method set()
#img_colour.itemset((row,col,0),0)
#img_colour.itemset((row,col,1),0)
#img_colour.itemset((row,col,2),0)

# visualise image using matplotlib
plt.figure(3)
plt.imshow(img_colour)
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')


# modify pixel intensity values in greyscale image
img_greyscale[row, col] = 0

# uncomment the lines below if you want to modify the pixel value using the pre-built method itemset()
#img_greyscale.itemset((row,col),0)

# visualise image using matplotlib
plt.figure(4)
plt.imshow(img_greyscale,cmap='gray')
plt.title('INPUT IMAGE: COLOUR')
plt.xlabel('x-resolution')
plt.ylabel('y-resolution')

# visualise figures
plt.show()
