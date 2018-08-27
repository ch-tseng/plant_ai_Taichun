#!/home/pi/.virtualenvs/dp/bin/python
import subprocess
import cv2
import gc

import time, sys, traceback, os
import numpy as np
import string
from plantcv import plantcv as pcv

from plant_detection.PlantDetection import PlantDetection

from pydarknet import Detector, Image
import imutils

def takePicture():
    bashCommand = "fswebcam -r 640x360 --no-banner cam.jpg"
    #process = subprocess.Popen(bashCommand.split(),shell=True, close_fds=True, stdout=subprocess.PIPE)
    #output, error = process.communicate()
    os.system(bashCommand)
    time.sleep(5)
    img = cv2.imread("cam.jpg")
    return img

def displayImage(img, bgimg, winName="test", waitTime=0):
    if(isinstance(img, np.ndarray)):
        print(img.shape)
        img = imutils.resize(img, height=720)
        print(img.shape)

        if(bgimg != None):
            bg = cv2.imread(bgimg)
            bg[240:240+img.shape[0], 0:img.shape[1]] = img
        else:
            bg = img

    else:
        bg = cv2.imread(bgimg)

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

def plant_cv(img):
    counter = 0
    debug = None

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

    zeros = np.zeros(masked2.shape[:2], dtype = "uint8")
    merged = cv2.merge([zeros, ab_fill, zeros])

    return merged, masked2

def yolo_plants(img):
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

def yolo_insects(img):
    net = Detector(bytes("cfg.insects/yolov3-tiny.cfg", encoding="utf-8"),
            bytes("cfg.insects/weights/yolov3-tiny_95000.weights", encoding="utf-8"), 0,
            bytes("cfg.insects/obj.data",encoding="utf-8"))

    img2 = Image(img)

    results = net.detect(img2)

    for cat, score, bounds in results:
        cat = cat.decode("utf-8")
        if(cat == "0_ladybug"):
            boundcolor = (4, 5, 250)
        elif(cat == "1_Camellia"):
            boundcolor = (215, 158, 2)
        elif(cat == "2_Pieridae"):
            boundcolor = (57, 182, 6)
        elif(cat == "3_Lindinger"):
            boundcolor = (5, 70, 111)
        elif(cat == "4_Papilio_1_4"):
            boundcolor = (6, 148, 195)
        elif(cat == "5_Papilio_5"):
            boundcolor = (6, 148, 195)
        elif(cat == "6_ant"):
            boundcolor = (249, 7, 132)


        x, y, w, h = bounds
        print(x, y, w, h)

        xx = int(round(x, 0))
        yy = int(round(y, 0))
        ww = int(round(w, 0))
        hh = int(round(h, 0))
        cv2.rectangle(img, (int(xx - (ww / 2)), int(yy - (hh / 2))), (int(xx + (ww / 2)), int(yy + (hh / 2))), (255, 0, 0), thickness=2)

        boundbox = cv2.imread("images/"+cat+".jpg")
        print("read:","images/"+cat+".jpg")
        print(boundbox.shape)
        print(yy, yy+boundbox.shape[0], xx, xx+boundbox.shape[1])
        #img[ int(y-h/2):int(y-h/2)+boundbox.shape[0], int(x-w/2):int(x-w/2)+boundbox.shape[1]] = boundbox
        img[ yy:yy+boundbox.shape[0], xx:xx+boundbox.shape[1] ] = boundbox
        
    return img


def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def ndvi1(image):
    # use Standard NDVI method, smaller for larger area
    thRED1 = 150
    thYELLOW1 = 60
    thGREEN1 = 0

    #image = cv2.bitwise_and(image, image, mask=cv2.imread("9_or_joined.png"))
    r, g, b = cv2.split(image)
    divisor = (r.astype(float) + b.astype(float))
    divisor[divisor == 0] = 0.01  # Make sure we don't divide by zero!

    ndvi = (b.astype(float) - r) / divisor

    #Paint the NDVI image
    ndvi2 = contrast_stretch(ndvi)
    ndvi2 = ndvi2.astype(np.uint8)

    redNDVI = cv2.inRange(ndvi2, thRED1, 255)
    yellowNDVI = cv2.inRange(ndvi2, thYELLOW1, thRED1)
    greenNDVI = cv2.inRange(ndvi2, thGREEN1, thYELLOW1)
    merged = cv2.merge([yellowNDVI, greenNDVI, redNDVI])

    #text = '[Max]: {m} '.format(m=round(ndvi.max(),1))
    #text = text + '[Mean]: {m} '.format(m=round(ndvi.mean(),1))
    #text = text + '[Median]: {m} '.format(m=round(np.median(ndvi),1))
    #text = text + '[Min]: {m}'.format(m=round(ndvi.min(),1))
    return merged


cv2.namedWindow("Plant Image", cv2.WND_PROP_FULLSCREEN)        # Create a named window
cv2.setWindowProperty("Plant Image", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

ii = 0
while True:
    displayImage(None, "images/bg_ndvi0.jpg", "Plant Image", 3000)

    img = takePicture()
    displayImage(img, "images/bg_pic.jpg", "Plant Image", 3000)

    img = ndvi1(img)
    displayImage(img, "images/bg_ndvi1.jpg", "Plant Image", 5000)

    del img
#--> PLANT CV
    displayImage(None, "images/bg_plantcv0.jpg", "Plant Image", 3000)

    img = takePicture()
    displayImage(img, "images/bg_pic.jpg", "Plant Image", 3000)

    merged, masked2 = plant_cv(img)

    for i in range(2):
        displayImage(img, "images/bg_a1.jpg", "Plant Image", 3000)
        displayImage(merged, "images/bg_a1.jpg", "Plant Image", 3000)
        displayImage(masked2, "images/bg_a1.jpg", "Plant Image", 3000)

    del img
    del merged
    del masked2
#<--- END PLANT CV

#--> Yolo for plant classes
    try:
        displayImage(None, "images/bg_a2.jpg", "Plant Image", 3000)

        img = takePicture()
        img = yolo_plants(img)
        #displayImage(img, "images/bg_a1.jpg", "Plant Image", 1000)
        #img = yolo_plants(img)
        displayImage(img, "images/bg_a3.jpg", "Plant Image", 500)

    except:
        pass

#--> Yolo for insect classes
    try:
        img = takePicture()
        img = yolo_insects(img)
        #displayImage(img, "images/bg_a1.jpg", "Plant Image", 1000)
        displayImage(None, "images/bg_b2.jpg", "Plant Image", 5000)
        #img = yolo_plants(img)
        displayImage(img, "images/bg_b3.jpg", "Plant Image", 10000)

        del img
        gc.collect()
        ii += 1

    except:
        pass

    if(ii>5):
        #os.kill(os.getpid(), 9)
        os.execv('/home/pi/taichun/main.py', [''])
