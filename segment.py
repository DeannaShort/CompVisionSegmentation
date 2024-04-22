# Author: Deanna Short
# Course: CMPS 473
# Project: Object Counting: Instance Segmentation Using the Traditional Technique, Watershed Segmentation
# Resources: 
#   * https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
#   * https://www.youtube.com/watch?v=3MUxPn3uKSk

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def read_image(file_name):
    img = cv.imread(f'images/{file_name}')
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return [img, imgRGB]

def plot_grey(subplot, img, title):
    plt.subplot(subplot)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    
def plot(subplot, img, title):
    plt.subplot(subplot)
    plt.imshow(img)
    plt.title(title)
    
def display_count(contours, name):
    count = len(contours)
    print(f'[{name}] Object count: {count}') # display approximated counts for countours 
    
def display_count_watershed(markers):
    unique_markers= np.unique(markers)
    count = len(unique_markers) - 2
    print(f'[Watershed] Object count: {count}')

def pre_process(img):
    ## SMOOTH IMAGE ##
    img = cv.bilateralFilter(img, 9, 75, 75) # apply bilateral filter to smooth picture while preserving edges
    
    ## CREATE BINARY IMAGE ##
    _, img_threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU) # apply inital thresholding
    threshold_contours, _ = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    plot_grey(232, img_threshold, 'Thresholded Input Image')
    
    ## FILL GAPS WITH DILATION ##
    kernel = np.ones((3,3), np.uint8)
    img_dilate = cv.morphologyEx(img_threshold,cv.MORPH_OPEN,kernel, iterations = 3) # use morphology open (erosion & dilation) to fill in gaps between white spaces
    dilated_contours, _ = cv.findContours(img_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    plot_grey(233, img_dilate, 'Dilated Thresholded Image')

    ## SEPARATE OVERLAPPING OBJECTS ##
    distance_transformed = cv.distanceTransform(img_dilate, cv.DIST_L2, 5)  # apply distance transform and thresholding for image partitioning
    plot_grey(234, distance_transformed, 'Distance Transformed Image')
    
    ## CREATE BINARY IMAGE FROM DISTANCE TRANSFORM ##
    _,distance_threshhold = cv.threshold(distance_transformed, 2, 255, cv.THRESH_BINARY) # apply thresholding for binary mask
    plot_grey(235, distance_threshhold, 'Thresholded Distance Transformed Image')
    
    ## CREATE MARKERS ##
    distance_threshhold = np.uint8(distance_threshhold)
    disance_transformed_countours, _ = cv.findContours(distance_threshhold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    _,markers = cv.connectedComponents(distance_threshhold)
    plot_grey(236, markers, 'Connected Components')
    
    ## SET UNKNOWN AREA AS BACKGROUND ##
    unknown = cv.subtract(img_dilate, distance_threshhold) # dilated image as background and thresholded distance-transform image as foreground
    markers += 1 # add one to all labels so that sure background is not 0, but 1
    markers[unknown==255] = 0 # set region of unknown with zero
        
    ## DISPLAY COUNTS FROM IMAGES ## 
    display_count(threshold_contours, 'Thresholded Input Image')
    display_count(dilated_contours, 'Dilated Thresholded Image')
    display_count(disance_transformed_countours, 'Thresholded Distance Transformed Image')
    
    return markers # return markers to be used in watershed

def watershed(): 
    img, imgRGB = read_image("traffic.png")

    ## DISPLAY ORIGINAL IMAGE ##
    plt.figure('Pre-Processing')
    plot(231, imgRGB, 'Input Image')
    
    ## PRE-PROCESS & OBTAIN MARKERS ##
    markers = pre_process(img)
    
    ## IMPLEMENT WATERSHED ##
    plt.figure('Watershed Technique') 
    markers = np.int32(markers)
    markers = cv.watershed(imgRGB,markers) # apply watershed
    plot(121, markers, 'Watershed')

    ## VISUALIZE BOUNDARIES ##
    imgRGB[markers == -1] = [255,0,0]
    plot(122, imgRGB, 'Segmented Image')

    ## DISPLAY COUNTS FROM WATERSHED ##
    display_count_watershed(markers)
    
    plt.show()
    
watershed() 