import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# select a region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    #cv2.namedWindow('Line detection', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Line detection', 900, 550)	    
    #cv2.imshow('Line detection',masked_image)

    return masked_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=3):

    """
    Draws lines on image.

    Once the line is detected, it will render the detected lines
    back onto the image itself.
    """    

    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # Make a copy of the original image.
    img = np.copy(img)
    
    # If there are no lines to draw, exit.
    if lines is None:
        return

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    
    # Return the modified image.
    return img

def pipeline(image):
    """
    An image processing pipeline which will output
    an image with the lane lines annotated.
    """
    
    # 1.- Read image
    img_colour = image

    # verify that image `img` exist
    if img_colour is None:
        print('ERROR: image ', img_name, 'could not be read')
        exit()

	# 2. Convert from BGR to RGB then from RGB to greyscale
    img_colour_rgb = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)

	# 3.- Apply Gaussuan smoothing
    kernel_size = (7,7)
    blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)

	# 4.- Apply Canny edge detector
    low_threshold = 10
    high_threshold = 70
    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)

	# 5.- Define a polygon-shape like region of interest
    img_shape = grey.shape

    # uncomment the following lines when extracting lines around the whole image
    '''
    img_size = img_shape
    bottom_left = (0, img_size[0])
    top_left = (0, 0)
    top_right = (img_size[1], 0)
    bottom_right = (img_size[1], img_size[0])
    '''

	# comment the following lines when extracting lines around the roi
    bottom_left = (430, 840)
    top_left = (900, 580)
    top_right = (1020, 580)
    bottom_right = (1530, 838)

    # create a vertices array that will be used for the roi
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)

	# 6.- Get a region of interest using the just created polygon. This will be
	#     used together with the Hough transform to obtain the estimated Hough lines
    masked_edges = region_of_interest(edges, vertices)

	# 7.- Apply Hough transform for lane lines detection
    rho = 1                       # distance resolution in pixels of the Hough grid
    theta = np.pi/180             # angular resolution in radians of the Hough grid
    threshold = 40                # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 5              # minimum number of pixels making up a line
    max_line_gap = 5              # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_colour)*0   # creating a blank to draw lines on
    hough_lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
 
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (2.8 / 5))
    max_y = int(image.shape[0] * (3.9 / 5))

    if left_line_x is not None:
        if left_line_y is not None:
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))
 
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    if left_line_x is not None:
        if left_line_y is not None:
            poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
 
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )
    return line_image

#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#white_output = 'highway_right_solid_white_line_short_with_red_lines.mp4'
#clip1 = VideoFileClip("highway_right_solid_white_line_short.mp4")
#white_clip = clip1.fl_image(pipeline)
#white_clip.write_videofile(white_output, audio=False)



# visualise output video
# create a VideoCapture object and specify video file to be read
cap = cv2.VideoCapture('highway_right_solid_white_line_short.mp4')

# main loop
while(cap.isOpened()):

    # read current frame
    ret, frame = cap.read()

    # convert from colour to grayscale image
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # run pipeline
    line_image = pipeline(frame)
    
    cv2.namedWindow('Line detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Line detection', 900, 550)	    
    cv2.imshow('Line detection',line_image)

    # wait for the user to press 'q' to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture object
cap.release()

# destroy windows to free memory
cv2.destroyAllWindows()
