import cv2
import numpy as np
from matplotlib import pyplot as plt

url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
cap = cv2.VideoCapture(url)
def templMat():
    imageFile = "sh.jpg"
    templateFile = cap

    img1 = cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE)
    img2 = img1.copy()

    temp = cv2.imread(templateFile,cv2.IMREAD_GRAYSCALE)
    # 템플릿 이미지의 가로세로 폭을 정의
    w,h = temp.shape[::-1]

    # for문을 돌리기위해 matchTemplate 함수의 method 인자를 지정
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # 메소드 배열에 대한 for문
    for meth in methods:
        # 이미지가 메소드당 한번씩 출력됨. 현재보고있는 이미지를 끄면 그다음 for문의 메소드 결과값 출력
        img1 = img2.copy()
        # for 문이 돌아갈때마다 다른 method 의 값을 method 함수에 담아줌
        method = eval(meth)

        # 각각의 method 값에 의한 템플릿 매칭 시작
        try:
            res = cv2.matchTemplate(img1,temp,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        except:
            print("error",meth)
            continue

        # TM_SQDIFF 와 TM_SQDIFF_NORMED 는 최소값을 사용함. 그외는 다 최대값
        # top_left의 좌표가 해당 템플릿이 위치한 왼쪽 위의 좌표
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # 템플릿의 오른쪽아래끝 좌표는 top_left 값에서 템플릿이미지의 가로세로폭을
        # 각좌표에 더해주면됨.
        bottom_right = (top_left[0]+w, top_left[1]+h)
        # 찾은 템플릿이미지에 흰색 사각형을 쳐줌
        cv2.rectangle(img1, top_left, bottom_right, 255, 2)

        # matplotlib 를 이용한 비교 gui 작성
        plt.subplot(121)
        plt.imshow(res,cmap='gray')
        plt.title('Matching Result')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.imshow(img1,cmap='gray')
        plt.title('Detect Point')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(meth)
##python3 detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg 
        plt.show()

templMat()
