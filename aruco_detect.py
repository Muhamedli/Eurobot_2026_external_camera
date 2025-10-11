import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Image path for detection
IMG_NAME = 'img_archive/f4.bmp'

img = cv.imread(IMG_NAME, cv.IMREAD_GRAYSCALE)
img_rgb = cv.imread(IMG_NAME, cv.IMREAD_COLOR_RGB)
blur = cv.GaussianBlur(img,(3,3),0)

th1 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,55,4)

ret2, th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
cl1 = clahe.apply(blur)
ret3, th3 = cv.threshold(cl1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

images = [img, img_rgb, th1, th2, th3]
titles = ['Original GRAYSCALE Image', 'Original RGB Image', "adaptiveThreshold", "Otsu's Thresholding", "Otsu's Thresholding + CLAHE"]

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

for i in images:
    corners, ids, rejected = detector.detectMarkers(i)
    if ids is not None:
        cv.aruco.drawDetectedMarkers(i, corners, ids, (125, 125, 125))

for i in range(4):
    plt.subplot(4,2,i*2+1),plt.imshow(images[0], 'gray')
    plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(4,2,i*2+2),plt.imshow(images[i+1], 'gray')
    plt.title(titles[i+1]), plt.xticks([]), plt.yticks([])
plt.show()