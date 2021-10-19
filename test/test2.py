import numpy as np 
import cv2 

url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
cap = cv2.VideoCapture(url) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) 
fgbg = cv2.createBackgroundSubtractorKNN()
while(1): 
    ret, frame = cap.read() 
    frame = cv2.resize(frame,(800,600)) 
    fgmask = fgbg.apply(frame) 
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel) 

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    # for index, centroid in enumerate(centroids):
    #     if stats[index][0] == 0 and stats[index][1] == 0:
    #         continue
    #     if np.any(np.isnan(centroid)):
    #         continue


    #     x, y, width, height, area = stats[index]
    #     centerX, centerY = int(centroid[0]), int(centroid[1])

    #     if area > 100:
    #         cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
    #         cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
    cv2.imshow('orig',frame) 
    cv2.imshow('frame',fgmask) 
    k = cv2.waitKey(30) & 0xff 
    if k == 27 : 
        break 
cap.release() 
cv2.destroyAllWindows()
