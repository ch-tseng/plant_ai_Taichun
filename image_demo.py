from pydarknet import Detector, Image
import argparse
import cv2
import os
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the source image")
args = vars(ap.parse_args())

if __name__ == "__main__":
    # net = Detector(bytes("cfg/densenet201.cfg", encoding="utf-8"), bytes("densenet201.weights", encoding="utf-8"), 0, bytes("cfg/imagenet1k.data",encoding="utf-8"))

    net = Detector(bytes("cfg.taichun/yolov3-tiny.cfg", encoding="utf-8"), 
        bytes("cfg.taichun/weights/yolov3-tiny_3600.weights", encoding="utf-8"), 0, 
        bytes("cfg.taichun/obj.data",encoding="utf-8"))

    img = cv2.imread(args["image"])

    img2 = Image(img)

    # r = net.classify(img2)
    results = net.detect(img2)
    print(results)

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

        #cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))

    cv2.imwrite("output.jpg", img)
    #img = imutils.resize(img, width=700)
    #cv2.imshow("output", img)
    # img2 = pydarknet.load_image(img)

    #cv2.waitKey(0)
