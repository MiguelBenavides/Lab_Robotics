""" draw_shapes_on_image.py

    example:

    line:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -s line -lp1 615 70 -lp2 695 70 -c 255 0 0 -t 2

    rectangle:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -s rectangle -lp1 610 530 -lp2 780 700 -c 0 0 255 -t 2

    circle:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -s circle -xy 690 610 -r 100 -c 0 255 0 -t 2

    ellipse:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -s ellipse -xy 300 450 -hw 100 50 -rot 45 -ang1 130 -ang2 270 -c 0 0 255 -t 2

    polygon:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -a 1048 180 1121 220 1099 314 995 314 976 220 -s polygon -b True -c 0 255 0 -t 2

    text:
    python3.5 draw_shapes_on_image.py -i vehicular_traffic.jpg -s text -xy 640 530 -txt "Test text" -c 255 0 0 -t 2


    This script draws different geometric shapes on an image.

    This includes:
        - lines
        - rectangles
        - circles
        - ellipses
        - polygons
        - text

    author: Miguel Benavides Banda
    date created: 18-02-2018
    universidad de monterrey.
"""

# import required libraries
import numpy as np
import cv2
import argparse


# ------------------------------------- #
# -------- UTILITY FUNCTIONS ---------- #
# ------------------------------------- #

# define function to draw a line
def draw_a_line(img, p1, p2, colour, thickness, linetype=cv2.LINE_8):

    # draw a line on image
    cv2.line(img, p1, p2, colour, thickness, linetype)

    # return img
    return img


# define function to draw a rectangle
def draw_a_rectangle(img, p1, p2, colour, thickness, linetype=cv2.LINE_8):

    # draw a line on image
    cv2.rectangle(img, p1, p2, colour, thickness, linetype)

    # return img
    return img


# define function to draw a circle
def draw_a_circle(img, xy, radius, colour, thickness):

    # draw a line on image
    cv2.circle(img, xy, radius, colour, thickness)

    # return img
    return img


# define function to draw an ellipse
def draw_an_ellipse(img, xy, hw, rotation, ang1, ang2, colour, thickness):

    # draw a line on image
    cv2.ellipse(img, xy, hw, rotation, ang1, ang2, colour, thickness)

    # return img
    return img

# define function to draw a polygon
def draw_a_polygon(img, array, boolean, colour, thickness):

    # draw a line on image
    cv2.polylines(img, array, boolean, colour, thickness)

    # return img
    return img

# define function to put text
def put_Text(img, text, xy, font, font_size, colour, thickness, linetype=cv2.LINE_8):

    # draw a line on image
    cv2.putText(img, text, xy, font, font_size, colour, thickness, linetype)

    # return img
    return img

# ------------------------------------- #
# ------------- MAIN CODE ------------- #
# ------------------------------------- #

# parse command line arguments
parser = argparse.ArgumentParser('Draw geometric shapes on an image')
parser.add_argument('-i', '--image',
                    help='name of input image', type=str, required=True)
parser.add_argument('-s', '--shape',
                    help='geometric shape to be drawn on the input image', type=str, required=True)
parser.add_argument('-lp1', '--line_p1', nargs='*',
                    help='x,y coordinate of point 1', required=False)
parser.add_argument('-lp2', '--line_p2', nargs='*',
                    help='x,y coordinate of point 2', required=False)
parser.add_argument('-xy', '--coordinate_xy', nargs='*',
                    help='center coordinates (x, y)', required=False)
parser.add_argument('-hw', '--axis_hw', nargs='*',
                    help='length of the minor and major axes (h, w)', required=False)
parser.add_argument('-r', '--radius',
                    help='radius of circle', type=int, required=False)
parser.add_argument('-rot', '--rotation',
                    help='rotation angle of the ellipse (calculated counterclockwise)', type=int, required=False)
parser.add_argument('-ang1', '--start_angle',
                    help='starting angle (calculated clockwise)', type=int, required=False)
parser.add_argument('-ang2', '--final_angle',
                    help='final angle (calculated clockwise)', type=int, required=False)
parser.add_argument('-c', '--colour', nargs='*',
                    help='stroke colour in BGR (not RGB, be careful)', required=False)
parser.add_argument('-t', '--thickness',
                    help='stroke thickness  (in pixels)', type=int, required=False)
parser.add_argument('-b', '--boolean',
                    help='True, if it is a closed line (Polygon)', type=bool, required=False)
parser.add_argument('-a', '--array', nargs='*',
                    help='array of coordinates for polygon', type=list, required=False)
parser.add_argument('-txt', '--text',
                    help='the text to be written', type=str, required=False) 
args = vars(parser.parse_args())

# retrieve name of input image given as argument from command line
img_in_name = args['image']

# read in image from disk
img_in = cv2.imread(img_in_name, cv2.IMREAD_COLOR) # alternatively, you can use cv2.IMREAD_GRAYSCALE

# verify that image exists
if img_in is None:
    print('ERROR: image ', img_in_name, 'could not be read')
    exit()

# retrieve geometric shape name
geometric_shape = args['shape']

# if geometric shape is a line or a rectangle
if (geometric_shape == 'line') or (geometric_shape == 'rectangle'):

    # retrieve line features
    line_p1 = args['line_p1']
    line_p2 = args['line_p2']
    colour = args['colour']
    thickness = args['thickness']

    # if '--line' is specified, but either '--line_p1' or
    # '--line_p2' is missing, ask the user to enter
    # the corresponding coordinate
    if (line_p1 is None) or (line_p2 is None) or (colour is None) or (thickness is None):

        # ask user enter line coordinates
        print('ERROR: line coordinate or info missing')
        exit()

    # otherwise
    else:

        # retrieve line coordinates
        line_p1 = tuple(list(map(int, line_p1)))
        line_p2 = tuple(list(map(int, line_p2)))
        colour = tuple(list(map(int, colour)))

        # check that each coordinate is of length 2
        if len(line_p1) == 2 and len(line_p2)==2 and len(colour)==3:

            # if drawing a line
            if geometric_shape == 'line':

                # call 'draw_a_line'
                img_in = draw_a_line(img_in, line_p1, line_p2, colour, thickness)

            # if drawing a rectangle
            elif geometric_shape == 'rectangle':

                # call 'draw_a_rectangle'
                img_in = draw_a_rectangle(img_in, line_p1, line_p2, colour, thickness)

        # otherwise	
        else:

            # ask the user enter a valid line coordinate
            print('ERROR: both p1 and p2 coordinates must be of length 2 and colour length is 3')
            exit()


# if geometric shape is a circle
if (geometric_shape == 'circle'):

    # retrieve circle features
    coordinate_xy = args['coordinate_xy']
    radius = args['radius']
    colour = args['colour']
    thickness = args['thickness']

    # if '--circle' is specified, but '--coordinate_xy ' is missing,
    # ask the user to enter the corresponding coordinate
    if (coordinate_xy is None) or (radius is None) or (colour is None) or (thickness is None):

        # ask user enter line coordinates
        print('ERROR: x,y coordinates or info missing')
        exit()

    # otherwise
    else:

        # retrieve circle coordinates
        coordinate_xy = tuple(list(map(int, coordinate_xy)))
        colour = tuple(list(map(int, colour)))

        # check that each coordinate is of length 2
        if len(coordinate_xy) == 2 and len(colour)==3:

            img_in = draw_a_circle(img_in, coordinate_xy, radius, colour, thickness)

        # otherwise
        else:

            # ask the user enter a valid line coordinate
            print('ERROR: xy coordinates must be of length 2')
            exit()


# if geometric shape is an ellipse
if (geometric_shape == 'ellipse'):

    # retrieve ellipse features
    coordinate_xy = args['coordinate_xy']
    axis_hw = args['axis_hw']
    rotation = args['rotation']
    start_angle = args['start_angle']
    final_angle = args['final_angle']
    colour = args['colour']
    thickness = args['thickness']

    # if '--ellipse' is specified, but '--coordinate_xy ' is missing,
    # ask the user to enter the corresponding coordinate
    if (coordinate_xy is None) or (axis_hw is None) or (rotation is None) or (start_angle is None) or (final_angle is None) or (colour is None) or (thickness is None):

        # ask user enter line coordinates
        print('ERROR: either x,y coordinates or h,w axis or info are missing')
        exit()

    # otherwise
    else:

        # retrieve ellipse coordinates
        coordinate_xy = tuple(list(map(int, coordinate_xy)))
        axis_hw = tuple(list(map(int, axis_hw)))
        colour = tuple(list(map(int, colour)))

        # check that each coordinate is of length 2
        if len(coordinate_xy) == 2 and len(axis_hw)==2 and len(colour)==3:

            img_in = draw_an_ellipse(img_in, coordinate_xy, axis_hw, rotation, start_angle, final_angle, colour, thickness)

        # otherwise
        else:

            # ask the user enter a valid line coordinate
            print('ERROR: either xy coordinates and hw axis must be of length 2 and color length 3')
            exit()


# if geometric shape is a polygon
if (geometric_shape == 'polygon'):

    # retrieve polygon features
    array_list = args['array']
    boolean = args['boolean']
    colour = args['colour']
    thickness = args['thickness']

    # if '--polygon' is specified, but '--array ' is missing,
    # ask the user to enter the corresponding coordinate
    if (array is None) or (boolean is None) or (colour is None) or (thickness is None):

        # ask user enter line coordinates
        print('ERROR: either array coordinates or  info are missing')
        exit()

    # otherwise
    else:

        # retrieve polygon coordinates
        array = np.array(array_list, np.int32)
        array = array.reshape(-1,1,2)
        vrx = np.array([array], np.int32)
        vrx = vrx.reshape((-1,1,2))
        colour = tuple(list(map(int, colour)))

        # check that each coordinate is of length 2
        if len(colour)==3:

            img_in = draw_a_polygon(img_in, [vrx], boolean, colour, thickness)

        # otherwise
        else:

            # ask the user enter a valid array coordinate
            print('ERROR: array coordinates must be of length 5 and color length 3')
            exit()


# if put text is selected
if (geometric_shape == 'text'):

    # retrieve polygon features
    text = args['text']
    xy = args['coordinate_xy']
    colour = args['colour']
    thickness = args['thickness']
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_size=0.8

    # if '--polygon' is specified, but '--array ' is missing,
    # ask the user to enter the corresponding coordinate
    if (text is None) or (xy is None) or (colour is None) or (thickness is None):

        # ask user enter line coordinates
        print('ERROR: either xy coordinates or  info are missing')
        exit()

    # otherwise
    else:

        # retrieve polygon coordinates
        xy = tuple(list(map(int, xy)))
        colour = tuple(list(map(int, colour)))

        # check that each coordinate is of length 2
        if len(xy)==2 and len(colour)==3:

            img_in = put_Text(img_in, text, xy, font, font_size, colour, thickness)

        # otherwise
        else:

            # ask the user enter a valid array coordinate
            print('ERROR: xy coordinates must be of length 2 and color length 3')
            exit()


# create a new window for image purposes
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL

# visualise input and output image
cv2.imshow("input image", img_in)

# wait for the user to press a key
key = cv2.waitKey(0)

# destroy windows to free memory
cv2.destroyAllWindows()
print('windows have been closed properly')
exit()
