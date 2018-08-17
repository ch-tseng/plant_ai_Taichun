import subprocess
import cv2

import sys, traceback, os
import numpy as np
import string
from plantcv import plantcv as pcv

from plant_detection.PlantDetection import PlantDetection

from pydarknet import Detector, Image
import imutils

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

def plantDetect(imagePath, imgname, typeDisplay=0):

    if(typeDisplay==0):
        PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=True,
            clump_buster=False, draw_contours=False, circle_plants=False)
        print("Remove the soil area (去除土壤區域)")
    elif(typeDisplay==1):
        PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=True,
            clump_buster=False, draw_contours=True, circle_plants=False)
        print("Draw the outline of plants (描繪植物外框)")
    elif(typeDisplay==2):
        PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=True,
            clump_buster=True, draw_contours=False, circle_plants=False)
        print("Plant area segnemt (植物區域切分)")
    elif(typeDisplay==3):
        PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=True,
            clump_buster=False, draw_contours=False, circle_plants=True)
        print("Area for each plant (植物本秼區域)")
    elif(typeDisplay==4):
        PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=True,
            clump_buster=True, draw_contours=True, circle_plants=True)
        print("End...")
    #elif(typeDisplay==5):
    #    PD = PlantDetection(image=imagePath, verbose=False, text_output=False, grey_out=False,
    #        clump_buster=False, draw_contours=False, circle_plants=False)


    try:
        PD.detect_plants()
        #print (PD.detect_plants())
        #print("python test.py -i " + picPath + ".jpg")
        return cv2.imread(imgname+'_marked.jpg')

    except:
        #GPIO.cleanup()
        pass

def yoloDetect(img):
    net = Detector(bytes("cfg.taichun/yolov3-tiny.cfg", encoding="utf-8"),
        bytes("cfg.taichun/weights/yolov3-tiny_3600.weights", encoding="utf-8"), 0,
        bytes("cfg.taichun/obj.data",encoding="utf-8"))

    img2 = Image(img)

    results = net.detect(img2)

    for cat, score, bounds in results:
        cat = cat.decode("utf-8")
        if(cat == "Pteris_cretica"):
            boundcolor = (0, 238, 252)
        elif(cat == "Echeveria_Minibelle"):
            boundcolor = (227, 252, 2)
        elif(cat == "Crassula_capitella"):
            boundcolor = (249, 77, 190)

        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)

        boundbox = cv2.imread("images/"+cat+".jpg")
        print("read:","images/"+cat+".jpg")
        print(y, boundbox.shape[0],x , boundbox.shape[1])
        #img[ int(y-h/2):int(y-h/2)+boundbox.shape[0], int(x-w/2):int(x-w/2)+boundbox.shape[1]] = boundbox
        img[ int(y):int(y+boundbox.shape[0]), int(x):int(x+boundbox.shape[1])] = boundbox

    return img



counter = 0
debug = None

while True:
    img = takePicture()

    #for i in range(0,5):
    #    plant1 = plantDetect("cam.jpg", "cam", i)
    #    displayImage(plant1, "images/bg_a1.jpg", "Plant Image", 3000)


#--> PLANT CV
    counter, s = pcv.rgb2gray_hsv(img, 's', counter, debug)
    counter, s_thresh = pcv.binary_threshold(s, 145, 255, 'light', counter, debug)
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

    #-------------------------------------------------------
     # Identify objects
    #counter, id_objects,obj_hierarchy = pcv.find_objects(masked2, ab_fill, counter, debug)
    # Define ROI
    #counter, roi1, roi_hierarchy= pcv.define_roi(masked2, 'rectangle', counter, None, 'default', debug, True, 550, 0, -500, -1900)
    # Decide which objects to keep
    #counter, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi1, roi_hierarchy, id_objects, obj_hierarchy, counter, debug)
    # Object combine kept objects
    #counter, obj, mask = pcv.object_composition(img, roi_objects, hierarchy3, counter, debug)

    zeros = np.zeros(masked2.shape[:2], dtype = "uint8")
    merged = cv2.merge([zeros, ab_fill, zeros])

    for i in range(2):
        displayImage(img, "images/bg_a1.jpg", "Plant Image", 3000)
        displayImage(merged, "images/bg_a1.jpg", "Plant Image", 3000)
        displayImage(masked2, "images/bg_a1.jpg", "Plant Image", 3000)

    #<--- END PLANT CV

    displayImage(img, "images/bg_a2.jpg", "Plant Image", 1)
    yoloimage = yoloDetect(img)
    displayImage(yoloimage, "images/bg_a2.jpg", "Plant Image", 6000)
