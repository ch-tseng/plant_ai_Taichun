import subprocess
import cv2

import sys, traceback
import numpy as np
import string
from plantcv import plantcv as pcv

def takePicture():
    bashCommand = "fswebcam -r 1280x720 --no-banner cam.jpg"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    img = cv2.imread("cam.jpg")
    return img

def displayImage(img, bgimg, winName="test", waitTime=0):
    if(bgimg != None):
        bg = cv2.imread(bgimg)
        bg[240:240+img.shape[0], 0:img.shape[1]] = img
    else:
        bg = img

    cv2.imshow(winName, bg)
    cv2.waitKey(waitTime)

counter = 0
debug = None

while True:
    img = takePicture()

#--> PLANT CV
    counter, s = pcv.rgb2gray_hsv(img, 's', counter, debug)
    counter, s_thresh = pcv.binary_threshold(s, 90, 255, 'light', counter, debug)
    counter, s_mblur = pcv.median_blur(s_thresh, 5, counter, debug)

     # Convert RGB to LAB and extract the Blue channel
    counter, b = pcv.rgb2gray_lab(img, 'b', counter, debug)

    # Threshold the blue image
    counter, b_thresh = pcv.binary_threshold(b, 145, 255, 'light', counter, debug)
    counter, b_cnt = pcv.binary_threshold(b, 145, 255, 'light', counter, debug)
    # Join the thresholded saturation and blue-yellow images
    counter, bs = pcv.logical_or(s_mblur, b_cnt, counter, debug)
    counter, masked = pcv.apply_mask(img, bs, 'white', counter, debug)

    #----------------------------------------
    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    counter, masked_a = pcv.rgb2gray_lab(masked, 'a', counter, debug)
    counter, masked_b = pcv.rgb2gray_lab(masked, 'b', counter, debug)

    # Threshold the green-magenta and blue images
    counter, maskeda_thresh = pcv.binary_threshold(masked_a, 115, 255, 'dark', counter, debug)
    counter, maskeda_thresh1 = pcv.binary_threshold(masked_a, 135, 255, 'light', counter, debug)
    counter, maskedb_thresh = pcv.binary_threshold(masked_b, 128, 255, 'light', counter, debug)

    # Join the thresholded saturation and blue-yellow images (OR)
    counter, ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh, counter, debug)
    counter, ab = pcv.logical_or(maskeda_thresh1, ab1, counter, debug)
    counter, ab_cnt = pcv.logical_or(maskeda_thresh1, ab1, counter, debug)

    # Fill small objects
    counter, ab_fill = pcv.fill(ab, ab_cnt, 200, counter, debug)

    # Apply mask (for vis images, mask_color=white)
    counter, masked2 = pcv.apply_mask(masked, ab_fill, 'white', counter, debug)

    displayImage(masked2, "images/bg_a1.jpg", "Plant Image", 3)
#<--- END PLANT CV
