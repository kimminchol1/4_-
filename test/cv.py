import cv2
import numpy as np

url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
cap = cv2.VideoCapture(url)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
moveCap = cv2.createBackgroundSubtractorMOG2(varThreshold=170)
 
while True:
    
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # canny = cv2.Canny(blurred_frame, 100, 150)
    moveMask = moveCap.apply(frame)
    # moveMask = cv2.threshold(moveMask, 127,255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(moveMask)
    
# 기본 영상 필터링
    # cv2.imshow('grayVideo',hsv)
    # cv2.imshow('blurred_frame',blurred_frame)
    # cv2.imshow('canny',canny)

    # cv2.namedWindow('canny',cv2.WINDOW_NORMAL)
# 라벨링 
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))

    
    

    cv2.imshow('frame',frame)
    cv2.imshow('moveMask',moveMask)
    if cv2.waitKey(5) & 0xFF == 27:
        break


cv2.destroyAllWindows()

