"""
	testing_code.py
	
	This code segments blue color objects. Then makes an 
	AND-bitwise operation between the mask and input images. 
	With the resulting blue mask image then creates a roi, 
	inside this region numbers can be detected.

	author: Miguel Benavides, Laura Morales
	date created: 9 May 2018
	universidad de monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
 
#######   training part   ############# 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))
 
model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)
 
#######   testing part    #############

#Frame width & Height
w=640
h=480

def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        maxWidth = w/2
        maxHeight = h/2

        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

def resize_and_threshold_warped(image):
        #Resize the corrected image to proper size & convert it to grayscale
        #warped_new =  cv2.resize(image,(w/2, h/2))
        warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Smoothing Out Image
        blur = cv2.GaussianBlur(warped_new_gray,(5,5),0)

        #Calculate the maximum pixel and minimum pixel value & compute threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        threshold = (min_val + max_val)/2

        #Threshold the image
        ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)

        #return the thresholded image
        return warped_processed

#Font Type
font = cv2.FONT_HERSHEY_SIMPLEX

# create a VideoCapture object
cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
	print('Unable to open the camera')
	exit()

# main loop
while(True):

    # capture new frame
    ret, frame = cap.read()
 
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ----- Tune these parameters so that blue-colour  ------ #
    # ----- objects can be detected                    ------ #
    h_val_l = 80
    h_val_h = 120
    s_val_l = 100
    v_val_l = 100
    lower_blue = np.array([h_val_l,s_val_l, v_val_l])
    upper_blue = np.array([h_val_h, 255, 255])
    # ------------------------------------------------------- #

    # threshold the hsv image so that only the respective colour pixels are kept
    maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

    # AND-bitwise operation between the mask and input images
    blue_object_img = cv2.bitwise_and(frame, frame, mask=maskblue)

    # visualise current frame
    cv2.imshow('frame',frame)

    # visualise mask image
    cv2.imshow('maskblue', maskblue)

    # visualise segmented blue object
    cv2.imshow('blue object', blue_object_img)

#######   Use the mask to create roi   #######
    blurred = cv2.GaussianBlur(maskblue,(3,3),0)

    #Detecting Edges
    edges = auto_canny(blurred)

    #Contour Detection & checking for squares based on the square area
    cntr_frame, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    smallerArea = 0
    smallerContours = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        if len(approx)==4:
            area = cv2.contourArea(approx)
            
            if smallerArea == 0:
                smallerArea = area

            if area <= smallerArea:
                smallerArea = area
                smallerContours = [approx]
     
            if smallerArea > 5000 and smallerArea < 15000:
                cv2.drawContours(frame,smallerContours,0,(0,0,255),2)
    
    cv2.imshow('Edges', edges)
    cv2.imshow('Square detection', frame)

    ###Create black image to use as mask
    img = np.zeros([480,640,1],dtype=np.uint8)

    if smallerContours != 0:
        roi = np.array(smallerContours)
        roi = roi.reshape(-1)
        img[roi[3]+5:roi[5]-5, roi[4]+5:roi[6]-5] = 255

    cv2.imshow('mask_image',img)
    
    img_num = cv2.bitwise_and(frame, frame, mask=img)
 
    cv2.imshow('cropped_image',img_num)

    im = img_num

    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
 
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	 
    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                if string != '7':
                    print (string)
                    cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
                    cv2.imshow('im',im)
                    cv2.imshow('out',out)
                    cv2.waitKey(0)

    # wait for the user to press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture object
cap.release()

# destroy windows to free memory
cv2.destroyAllWindows()

