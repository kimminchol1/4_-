import cv2
import numpy as np
from numpy.core.numeric import True_

cap = cv2.VideoCapture("./Highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=125,detectShadows=False)
while True:
    ret, frame = cap.read()
    
    height, width, _ = frame.shape
    roi = frame[400: 700,450: 800]
    mask = object_detector.apply(roi)
    mask = cv2.bitwise_not(mask)
    # mask = cv2.GaussianBlur(mask, (3,3),0)
    _, mask = cv2.threshold(mask, 254 ,255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >90 :
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 2)
            print(x,y,w,h)

    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("roi", roi)
    key = cv2.waitKey(30)
    if key ==27:
        break

cap.release()