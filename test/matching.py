# import cv2 as cv
# import numpy as np


# src = cv.imread('sh1.png')
# template = cv.imread('sh.png')
# template = cv.resize(template,dsize=(0,0), fx = 0.9, fy = 0.9)

# th, tw = template.shape[:2]
# cv.imshow('template', template)

# #3가지 메서드 순회
# methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', \
#                                      'cv.TM_SQDIFF_NORMED']
# for i, method_name in enumerate(methods):
#     src_draw = src.copy()
#     method = eval(method_name)
#     res = cv.matchTemplate(src,template, method)
#     minv, maxv, minloc, maxloc = cv.minMaxLoc(res)

# #TM_SQDIFF는 최소값이 좋은 매칭. 나머지는 최대값이 좋은매칭
#     if method in [cv.TM_SQDIFF_NORMED, cv.TM_SQDIFF]:
#         top_left = minloc
#         match_val = minv
#     else :
#         top_left = maxloc
#         match_val = maxv
#     bottom_right = (top_left[0] + tw, top_left[1] + th)
#     cv.rectangle(src_draw, top_left, bottom_right, (0,0,255),2)
#     # 매칭 포인트 표시
#     cv.putText(src_draw, str(match_val), top_left, \
#                 cv.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv.LINE_AA)
#     cv.imshow(method_name, src_draw)
# cv.waitKey(0)
# cv.destroyAllWindows()

    



##########위는 이미지 to 이미지 매칭 #############
import cv2
import numpy as np

url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
cap = cv2.VideoCapture(url)

# template = cv2.imread('tt.png',cv2.IMREAD_GRAYSCALE)
template = cv2.imread('ss.png',cv2.IMREAD_GRAYSCALE)
w,h = template.shape[::-1]

while True:
    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # cv2.add()
    res = cv2.matchTemplate(frame_gray, template ,cv2.TM_CCOEFF_NORMED)
    # 오후 2시쯤 밝기가 밝을때 threshold가 0.8이면 잘잡힘
    # 해가 졌을 때는 threshold가 0.95면 잘잡힘 
    threshold = 0.8
    # threshold = 0.95
    # if len(res):
    loc = np.where(res>=threshold)

    for pts in zip(*loc[::-1]):
        cv2.rectangle(frame,pts,(pts[0]+w,pts[1]+h),(0,0,255),1)

    cv2.imshow('detected',frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release() 
###### 동영상 to 이미지 매칭######