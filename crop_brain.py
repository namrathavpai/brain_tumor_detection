import cv2
import numpy as np
import os
from os import listdir
import imutils

# path = "Resources"

def brain_crop_resize(img):
    img = cv2.imread(img, 0)
    #img = cv2.GaussianBlur(img,(3,3),1)
    #remove noise
    threshold = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=2)
    #find contours
    cnts= cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    #get the corner points of the skull/brain
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
    # cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(img, extRight, 8, (0, 255, 0), -1)
    # cv2.circle(img, extTop, 8, (255, 0, 0), -1)
    # cv2.circle(img, extBot, 8, (255, 255, 0), -1)

    #crop image according to the cornerpoint
    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    #resize thr brain_image
    new_image = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)

    return new_image



# cv2.imshow("out", img)
# cv2.imshow("out1", threshold)
# cv2.imshow("final", new_image)
# cv2.waitKey(0)


