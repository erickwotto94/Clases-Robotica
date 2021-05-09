import cv2
import numpy as np
from pyzbar import pyzbar
import math
#Funciones a implementar

def getQRS(img):
    return [{
        'polygon':QR.polygon,
        'rect':QR.rect,
        'text':QR.data.decode('utf-8')} 
        for QR in pyzbar.decode(img)]

def filterImage(image,minH,maxH):
    imageHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    minHSV=np.array([minH,0,0])
    maxHSV=np.array([maxH,255,255])
    return cv2.inRange(imageHSV,minHSV,maxHSV)

def getPosNorm(frame):
    minQR=95
    maxQR=158
    #Para filtrar de tal manera que se vea el contorno
    minPiso=12
    maxPiso=29
    imgQR= filterImage(frame,minQR,maxQR)
    imgPiso= filterImage(frame,minPiso,maxPiso)
    cv2.imshow("Original",frame)
    cv2.imshow("QR",imgQR)
    cv2.imshow("Piso",imgPiso)
    cv2.waitKey(0) #cv2.waitKey devuelve el ascii que se presiono
    
    #Lectura del texto QR
    #imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    for QR in getQRS(frame):
        print("El texto en el QR dice: ",QR['text'])

    #Contorno del QR
    contornoQR=frame
    parametros=getQRS(contornoQR)
    puntos=[]
    for i in parametros[0]['polygon']:
        puntos.append((i.x,i.y))
    contornoQR=cv2.rectangle(contornoQR,puntos[0],puntos[2],(255,100,1),2)
    cv2.imshow("ContornoQR",contornoQR)
    x1,y1=puntos[0] #205 101
    x2,y2=puntos[1] #206 237
    x3,y3=puntos[2] #282 237
    x4,y4=puntos[3]
    h=y2-y1
    w=x3-x1


    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray,100,300)
    cv2.imshow("Canny",imgCanny)
    cv2.waitKey(0)

    #Visualizacoin del contorno
    contours,jerarquia=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(frame,[contour],-1,(0,0,255),5)
        cv2.imshow("Contorno",frame)
        cv2.waitKey(0)

    #print(contours)
    #Imprimiendo los contornos se pueden obtener los vertices del cuadrado en 
    #donde se encuentra el robot, llamado aqui como imgPiso
    x1=480
    y1=20
    x2=27
    y2=28
    x3=27
    y3=472
    x4=480
    y4=472
    
    #Con esto se puede verificar los bordes del suelo
    #cv2.circle(frame,(x1,y1),15,(244,0,0),6)
    #cv2.imshow("frame",frame)
    #cv2.waitKey(0)
    
    #Dimensiones del piso
    #print("Largo 1:",y3-y2)
    #print("Largo 2:",y4-y1)
    #print("Ancho 1:",x1-x2)
    #print("Ancho 2:",x4-x3)
    ancho_medio=((x1-x2)+(x4-x3))/2
    largo_medio=((y3-y2)+(y4-y1))/2

    #Calculo de areas y centros
    moments_1=cv2.moments(imgQR,True)
    moments_2=cv2.moments(imgPiso,True)

    #Centro de masa
    xcenter=moments_1['m10']/moments_1['m00']
    ycenter=moments_1['m01']/moments_1['m00']
    cv2.destroyAllWindows()    

    return[{
        'cx': xcenter,
        'cy': ycenter,
        'h': h,
        'w': w,
        'text':QR['text'],
        'Ht': largo_medio,
        'Wt': ancho_medio}]

#########################################################

frame=cv2.imread("camera.png")
for TP in getPosNorm(frame):
    print("cx:",TP['cx'])
    print("cy:",TP['cy'])
    print("h:",TP['h'])
    print("w:",TP['w'])
    print("Texto QR:",TP['text'])
    print("H:",TP['Ht'])
    print("W:",TP['Wt'])

posicionNormalizada=[TP['cx']/TP['Wt'],TP['cy']/TP['Ht']]
dimensionNormalizada=[TP['w']/TP['Wt'],TP['h']/TP['Ht']]

print("Posicion normalizada <cx/W,cy/H>:",posicionNormalizada)
print("Dimension normalizada <w/W,h/H>:",dimensionNormalizada)
#########################################################################