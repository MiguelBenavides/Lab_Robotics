# import required libraries
import numpy as np
import cv2


def configure_videoCapture(device_index):

    """
    Configure video capture object to handle video device.
    
    Parameters
        device_index: int value indicating the index number to access camera
        
    Returns
        cap: videoCapture-type object 
    
    """

    # create a videoCapture object and returns either a True or False
    cap = cv2.VideoCapture(device_index)

    # if camera could not be opened, it displays an error and exits
    if not cap.isOpened():
        print("ERROR: Camera could not be opened")
        exit()

    # return videoCapture object 'cap'
    return cap


def print_video_frame_specs(cap):
    
    """
    Print video specifications such as video frame width and height, fps, 
    brightness, contrast, saturation, gain, and exposure.
    
    Parameters
        cap: video capture object
        
    Returns
        None: this definition only prints information on the command line
              window.
    """    

    # retrieve video properties
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]

    # verify that frame was properly captured
    if ret == False:
        print("ERROR: current frame could not be read")
        exit()

    else: # if so, video frame stats are displayed
        
        # print video frames specifications
        print('\nVideo specifications:')
        print('\tframe width: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('\tframe height: ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('\tframe rate: ', cap.get(cv2.CAP_PROP_FPS))
        print('\tbrightness: ', cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print('\tcontrast: ', cap.get(cv2.CAP_PROP_CONTRAST))
        print('\tsaturation: ', cap.get(cv2.CAP_PROP_SATURATION))
        print('\thue: ', cap.get(cv2.CAP_PROP_GAIN))
        print('\texposure: ', cap.get(cv2.CAP_PROP_EXPOSURE))
        
    # return None
    return None


def capture_and_process_video(cap):

    """
    Capture live video from a camera connected to your computer. Each frame is
    flipped and visualised together with the original frame on separate windows.
    
    Parameters
        cap: video capture object
        
    Returns
        None: none
    
    """

    # create a new window for image purposes
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # alternatively, you can use cv2.WINDOW_NORMAL
    cv2.namedWindow("output image", cv2.WINDOW_AUTOSIZE) # that option will allow you for window resizing


    # main loop
    print('\nCapturing video ...')
    while(cap.isOpened()):

        # capture frame by frame
        ret, frame = cap.read()

	    # verify that frame was properly captured
        if ret == False:
            print("ERROR: current frame could not be read")
            break

        # if frame was properly captured, it is converted 
        # from a colour to a grayscale image
        frame_out = cv2.flip(frame,0)

        # visualise current frame and grayscale frame
        cv2.imshow("input image", frame)
        cv2.imshow("output image", frame_out)

       
        # wait for the user to press a key 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # return none
    return None

    
def free_memory(cap):

    """
    Free memory by releasing videoCapture 'cap' and by destroying/closing all 
    open windows.
    
    Parameters
        cap: video capture object
        
    Returns
        None: none
    """

    # when finished, release the VideoCapture object and close windows to free memory
    print('closing camera ...')
    cap.release()
    print('camera closed')
    cv2.destroyAllWindows()
    print('program finished - bye!\n')
    
    # return none
    return None


def run_pipeline(arg=None):
    """
    Run pipeline to capture, process and visualise both the original frame and 
    processed frame.
    
    Parameters
        arg: None
        
    Returns
        arg: None
        
    """
    
    # pipeline
    device_index = 0
    cap = configure_videoCapture(device_index)
    print_video_frame_specs(cap)
    capture_and_process_video(cap)
    free_memory(cap)
    
    # return none
    return arg
    
    
# run pipeline    
run_pipeline()
