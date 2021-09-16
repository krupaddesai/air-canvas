import cv2
import time
import mediapipe as mp

vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands= mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = vid.read()
    img = cv2.flip(img, 1)

    # must convert to rgb bc class hands only reads rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # allows process for multiple hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # retrieve id and landmark information
            for id, lm in enumerate(handLms.landmark):
            
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time() # gives current time
    fps = 1/(cTime-pTime)
    pTime = cTime 
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()






