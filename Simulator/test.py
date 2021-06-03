import cv2
import numpy as np
import math
from scipy import ndimage
import argparse
from math import atan2, cos, sin, sqrt, pi

def metodo_moment(imagen):
    print("############################MOMENTS##########################################")

    cv2.imshow("Imagen",imagen)
    cv2.waitKey(0)
    imgHSV=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    #filtro para clasificar la elipse
    minH=np.array([6,0,0])
    maxH=np.array([255,255,255])
    imagenBit=cv2.inRange(imgHSV,minH,maxH)
    moments=cv2.moments(imagenBit,True)
    print("Area de la elipse :",moments['m00']) #En pixeles al cuadrado
    print("Area total        : ",imagenBit.shape[0]*imagenBit.shape[1])
    xcenter=int(moments['m10']/moments['m00'])
    ycenter=int(moments['m01']/moments['m00'])
    cv2.circle(imagen, (xcenter, ycenter), 5, (255, 0, 0), -1)
    cv2.imshow("Centro",imagen)
    cv2.waitKey(0)
    print("xcenter           : ",xcenter)
    print("ycenter           : ",ycenter)
    print("ancho             : ",imagenBit.shape[0])
    print("largo             : ",imagenBit.shape[1])
    print("cx_normalizado    :",xcenter/imagenBit.shape[1])
    print("cy_normalizado    :",ycenter/imagenBit.shape[0])
    #angulo de rotacion de la imagen
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(imagen, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = -1*math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    print(f"Angulo            : {median_angle:.04f}")

def drawAxis(img, p_, q_, colour, scale):
    i=1
    p = list(p_)
    q = list(q_)
    
    angle = math.atan2(p[1] - q[1], p[0] - q[0])
    if i==1:
        angulo = (angle-pi/2)*180/pi
        print(angulo)
        i=i+1
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    return angulo

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    centro = (int(mean[0,0]), int(mean[0,1]))
    # Draw the principal components
    cv2.circle(img, centro, 3, (255, 0, 255), 2)
    p1 = (centro[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], centro[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (centro[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], centro[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])

    angulo1=drawAxis(img, centro, p1, (255, 255, 255), 1)
    a=drawAxis(img, centro, p2, (0, 0, 0), 5)
    angle = math.degrees(atan2(eigenvectors[0,1], eigenvectors[0,0])) 
    return centro,a

def metodo_pca(src):
    print("###############################PCA##########################################")
    cv2.imshow('src', src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    band=1
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        area_total=src.shape[0]*src.shape[1];
        if band==1:
            print("Area elipse   :",area)
            print("Area total    :",area_total)
            band=0
        if area < 1e2 or 1e5 < area:
            continue

        cv2.drawContours(src, contours, i, (0, 0, 255), 2)
        centro,angulo1= getOrientation(c, src)

    cv2.imshow('output', src)
    cv2.waitKey()
    xcenter=centro[0]
    ycenter=centro[1]
    print("xcenter       : ",xcenter)
    print("ycenter       : ",ycenter)
    print("ancho         : ",bw.shape[0])
    print("largo         : ",bw.shape[1])
    print("cx_normalizado:",xcenter/bw.shape[1])
    print("cy_normalizado:",ycenter/bw.shape[0])
    print("Angulo        : ",angulo1)







#DESCOMENTE LA IMAGEN Y EL METODO A UTILIZAR
imagen=cv2.imread("elipse.png")
#imagen=cv2.imread("rotado.png")
metodo_moment(imagen)

metodo_pca(imagen)





