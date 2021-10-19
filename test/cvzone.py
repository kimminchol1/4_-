import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
cap = cv2.VideoCapture(url)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

while True :
    _, frame = cap.read()
    frameOut = segmentor.removeBG(frame,(255,0,255))
    cv2.imshow("frame", frame)
    cv2.imshow("frameOut", frameOut)
    cv2.waitKey(1)