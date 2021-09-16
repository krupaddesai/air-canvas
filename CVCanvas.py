import cv2
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "HeaderImages"
myList = os.listdir(folderPath)
overlayList = []
for path in myList:
    image = cv2.imread(f'{folderPath}/{path}')
    overlayList.append(image)

header = overlayList[0]
color = (22, 22 ,255)
brushThickness = 20
eraserThickness = 60

vid = cv2.VideoCapture(0)
vid.set(3, 1280)
vid.set(4, 720)
detector = htm.handDetector()

imgCanvas= np.zeros((720, 1280, 3), np.uint8)
while True:
    # 1 Import Image
    success, img = vid.read()
    img = cv2.flip(img, 1)

    # 2 Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    
    #position of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3 Check which fingers are up
        fingers = detector.fingersUp()
        # 4 If selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 200 < x1 < 375:
                    header = overlayList[0]
                    color = (22, 22, 255)
                if 385 < x1 < 555:
                    header = overlayList[1]
                    color = (77, 145, 255)
                if 565 < x1 < 735:
                    header = overlayList[2]
                    color = (89, 222, 255)
                if 745 < x1 < 915:
                    header = overlayList[3]
                    color = (55, 128, 0)
                if 925 < x1 < 1095:
                    header = overlayList[4]
                    color = (173, 75, 0)
                if 1105 < x1 < 1280:
                    header = overlayList[5]
                    color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-25), (x2,y2+25), color, cv2.FILLED)
            
            #print("selection mode")
        # 5 If Drawing Mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), color, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), color, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, brushThickness)
            
            xp,yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)      
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 1, imgCanvas, 1, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
