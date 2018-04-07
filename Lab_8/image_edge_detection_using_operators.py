"""
	image_edge_detection_using_operators.py

	This code uses the sobel kernel to detect significant changes
	on the pixel intensity, allowing the code to identify the edge
	of the gifure inside the image.

	author: Miguel Benavides, Laura Moralez
	date: 7 April 2018
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2


# compute Sobel gradient
def compute_absolute_sobel_gradient(img, ax='x', ksize=3, threshold=(40,140)):

	# 1) check whether img is a colour or greyscale image
	if len(img.shape)>2:

		# convert from colour to greyscale image
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	else:
		# if greyscale grey=img
		grey = img

	# 2) take the derivate in 'ax' axis
	if ax.lower()=='x':

		# apply the Sobel operator along the x axis
		sobel_derivative = cv2.Sobel(grey, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=1)

	if ax.lower()=='y':

		# apply the Sobel operator along the y axis
		sobel_derivative = cv2.Sobel(grey, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=1)

	# 3) take the absolute value of the derivative
	sobel_absolute = np.absolute(sobel_derivative)

	# 4) scale to 8-bit (0-255), then convert to type = np.uint8
	sobel_scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))

	# 5) create a mask of 1's where threshold[0] < sobel_scaled < threshold[1]
	binary_output = np.zeros_like(sobel_scaled)
	threshold_min = threshold[0]
	threshold_max = threshold[1]
	binary_output[(sobel_scaled >= threshold_min) & (sobel_scaled <= threshold_max)] = 1

    # return binary_image with gradient being detected along 'ax' axis
	return sobel_derivative, binary_output

# compute Prewitt gradient
def compute_absolute_prewitt_gradient(img, ax='x'):

	# 1) check whether img is a colour or greyscale image
	if len(img.shape)>2:

		# convert from colour to greyscale image
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	else:
		# if greyscale grey=img
		grey = img

	# 2) use the gaussian blur on the grey image
	img_gaussian = cv2.GaussianBlur(grey,(3,3),0)

	# 3) create the prewitt kernel
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

	# 4) calculate the x or y edges
	if ax.lower()=='x':

		# apply the Prewitt operator along the x axis
		img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
		return img_prewittx

	if ax.lower()=='y':

		# apply the Prewitt operator along the y axis
		img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
		return img_prewitty

# compute Scharr gradient
def compute_absolute_scharr_gradient(img, ax='x'):

	# 1) check whether img is a colour or greyscale image
	if len(img.shape)>2:

		# convert from colour to greyscale image
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	else:
		# if greyscale grey=img
		grey = img

	# 2) use the gaussian blur on the grey image
	img_gaussian = cv2.GaussianBlur(grey,(3,3),0)

	# 3) create the scharr kernel
	kernelx = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
	kernely = np.array([[-3,-10,-3],[0,0,0],[3,10,3]])

	# 4) calculate the x or y edges
	if ax.lower()=='x':

		# apply the scharr operator along the x axis
		img_scharrx = cv2.filter2D(img_gaussian, -1, kernelx)
		return img_scharrx

	if ax.lower()=='y':

		# apply the scharr operator along the y axis
		img_scharry = cv2.filter2D(img_gaussian, -1, kernely)
		return img_scharry

# compute Roberts gradient
def compute_absolute_roberts_gradient(img, ax='x'):

	# 1) check whether img is a colour or greyscale image
	if len(img.shape)>2:

		# convert from colour to greyscale image
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	else:
		# if greyscale grey=img
		grey = img

	# 2) use the gaussian blur on the grey image
	img_gaussian = cv2.GaussianBlur(grey,(3,3),0)

	# 3) create the roberts kernel
	kernelx = np.array([[0,1],[-1,0]])
	kernely = np.array([[1,0],[0,-1]])

	# 4) calculate the x or y edges
	if ax.lower()=='x':

		# apply the roberts operator along the x axis
		img_robertsx = cv2.filter2D(img_gaussian, -1, kernelx)
		return img_robertsx

	if ax.lower()=='y':

		# apply the roberts operator along the y axis
		img_robertsy = cv2.filter2D(img_gaussian, -1, kernely)
		return img_robertsy

# combine x and y derivatives using AND operation
def combine_x_and_y_binary_derivatives(img_derivative_x, img_derivative_y, threshold=(40,120)):

	# verify that both image derivatives are the same size
	if img_derivative_x.shape != img_derivative_y.shape:
		print('ERROR [combine_x_and_y_binary_derivatives]: img_binary_x and img_binary_y images should be of same size')
		exit()

	# 1) take the absolute value of the derivative
	absolute_x = np.absolute(img_derivative_x)
	absolute_y = np.absolute(img_derivative_y)

	# 2) scale to 8-bit (0-255), then convert to type = np.uint8
	absolute_scaled_x = np.uint8(255 * absolute_x / np.max(absolute_x))
	absolute_scaled_y = np.uint8(255 * absolute_y / np.max(absolute_y))

	# 3) create a mask of 1's where threshold[0] < sobel_scaled < threshold[1]
	binary_combined_gradients = np.zeros_like(absolute_scaled_x)
	indx_x = (absolute_scaled_x >= threshold[0]) & (absolute_scaled_x <= threshold[1])
	indx_y = (absolute_scaled_y >= threshold[0]) & (absolute_scaled_y <= threshold[1])
	binary_combined_gradients[indx_x|indx_y] = 1

	# return combined binary x-and-y derivatives
	return binary_combined_gradients


# compute magnitude of x and y derivatives
def compute_magnitude_of_derivatives(img_derivative_x, img_derivative_y, ksize=3, threshold=(40,120)):

	# 1) calculate the magnitude
    gradient_magnitude = np.sqrt(np.power(img_derivative_x, 2) + np.power(img_derivative_y, 2))

    # 2) scale to 8-bit (0 - 255) and convert to type = np.uint8
    gradient_scaled = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # 3) create a binary mask where mag thresholds are met
    binary_magnitude = np.zeros_like(gradient_scaled)
    binary_magnitude[(gradient_scaled >= threshold[0]) & (gradient_scaled <= threshold[1])] = 1

    return binary_magnitude


# compute orientation of magnitude of x and y derivatives
def compute_direction(img_derivative_x, img_derivative_y, threshold=(0, np.pi/2)):

	# 1) take the absolute value of the x and y gradients
    gradient_magnitude_x = np.absolute(img_derivative_x)
    gradient_magnitude_y = np.absolute(img_derivative_y)

    # 2) calculate the gradient magnitude direction
    #gradient_direction = np.arctan2(gradient_magnitude_y, gradient_magnitude_x)
    gradient_direction = np.arctan2(img_derivative_y, img_derivative_x)

    # 3) reate a binary mask where direction thresholds are met
    binary_direction = np.zeros_like(gradient_direction)
    binary_direction[(gradient_direction >= threshold[0]) & (gradient_direction <= threshold[1])] = 1

    return binary_direction

# Image visualisation
def visualise_image(img, fig_number, title, flag_colour_conversion, conversion_colour, cmap):
  """
  INPUTS:

  img: image to be displayed
  fig_number: figure number
  title: figure's title
  flag_colour_conversion: True if image should be converted from one colour space to any other
  conversion_colour: BGR2RGB, BGR2GRAY, etc
  cmap: colour map to be used for visualisation purposes, e.g., cmap='gray'

  RETURNS:

  None
  """

  plt.figure(fig_number)

  if flag_colour_conversion:
    if conversion_colour == "BGR2RGB":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conversion_colour == "BGR2GRAY":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  plt.imshow(img, cmap)
  plt.title(title)
  plt.xticks([])
  plt.yticks([])

  return None

# pipeline
def run_pipeline(img_name):

	# read image
	img = cv2.imread(img_name)

	# verify that image `img` exist
	if img is None:
		print('ERROR: image ', img_name, 'could not be read')
		exit()

	# compute sobel derivative along x axis
	img_derivative_x, img_binary_x = compute_absolute_sobel_gradient(img, ax='x', ksize=3, threshold=(40,140))

	# compute sobel derivative along y axis
	img_derivative_y, img_binary_y = compute_absolute_sobel_gradient(img, ax='y', ksize=3, threshold=(40,140))

	# combine x and y derivatives
	img_combined_derivatives = combine_x_and_y_binary_derivatives(img_derivative_x, img_derivative_y, threshold=(50,140))

	# compute magnitude of gradient
	img_magnitude_gradient = compute_magnitude_of_derivatives(img_derivative_x, img_derivative_y, ksize=3, threshold=(40,140))

	# compute direction of gradient
	thresh_min = 0
	thresh_max = 2
	img_direction_gradient = compute_direction(img_derivative_x, img_derivative_y, threshold=(np.radians(thresh_min), np.radians(thresh_max)))

	# plot input and output images
	# this should visualise the colour input image figure shown above
	visualise_image(img, 1, 'Colour input image', True, 'BGR2RGB', 'gray')

	# this should visualise the x derivative image figure shown above
	visualise_image(img_derivative_x, 2, 'x derivative', False, '', 'gray')

	# this should visualise the binary x derivative image figure shown above
	visualise_image(img_binary_x, 3, 'binary x derivative', False, '', 'gray')
	
	# this should visualise the y derivative image figure shown above
	visualise_image(img_derivative_y, 4, 'y derivative', False, '', 'gray')

	# this should visualise the binary y derivative image figure shown above
	visualise_image(img_binary_y, 5, 'binary y derivative', False, '', 'gray')

	# this should visualise the combined x-and-y derivatives image figure shown above
	visualise_image(img_combined_derivatives, 6, 'combined x-and-y derivatives', False, '', 'gray')

	# this should visualise the gradient magnitude image figure shown above
	visualise_image(img_magnitude_gradient, 7, 'gradient magnitude', False, '', 'gray')

	# this should visualise the gradient direction image figure shown above
	visualise_image(img_direction_gradient, 8, 'gradient direction', False, '', 'gray')

	prewittx = compute_absolute_prewitt_gradient(img, ax='x')
	prewitty = compute_absolute_prewitt_gradient(img, ax='y')
	img_prewitt = prewittx + prewitty
	visualise_image(prewittx, 9, 'Prewitt x', False, '', 'gray')
	visualise_image(prewitty, 10, 'Prewitt y', False, '', 'gray')
	visualise_image(img_prewitt, 11, 'Prewitt', False, '', 'gray')

	scharrx = compute_absolute_scharr_gradient(img, ax='x')
	scharry = compute_absolute_scharr_gradient(img, ax='y')
	img_scharr = scharrx + scharry
	visualise_image(scharrx, 12, 'Scharr x', False, '', 'gray')
	visualise_image(scharry, 13, 'Scharr y', False, '', 'gray')
	visualise_image(img_scharr, 14, 'Scharr', False, '', 'gray')

	robertsx = compute_absolute_roberts_gradient(img, ax='x')
	robertsy = compute_absolute_roberts_gradient(img, ax='y')
	img_roberts = robertsx + robertsy
	visualise_image(robertsx, 15, 'Roberts x', False, '', 'gray')
	visualise_image(robertsy, 16, 'Roberts y', False, '', 'gray')
	visualise_image(img_roberts, 17, 'Roberts', False, '', 'gray')

	plt.show()

# uncomment the corresponding line to try a particular image
#img_name = 'opera_house_vivid_sydney.jpg'
img_name = 'sydney_harbour.jpg'
#img_name = 'vehicular_traffic.jpg'

# run pipeline
run_pipeline(img_name)
