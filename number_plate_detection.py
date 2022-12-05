import cv2
import numpy as np 

frame_width = 640
frame_height = 480
cascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
min_area = 500
counter = 0

cap = cv2.VideoCapture(0)
cap.set(3,frame_width)   
cap.set(4,frame_height)   
cap.set(10,150)           ##Changing brightness to 150

while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """Using pretrained OpenCV number plate cascade"""

    number_plates = cascade.detectMultiScale(img_gray, 1.1, 5)

    """Creating boundary boxes around detected number plates"""

    for (x,y,w,h) in number_plates:
        area = w*h
        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255),2)
            cv2.putText(img, "Number Plate",(x,y-5), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)
            
            """Picking the number plate from the caption"""

            img_roi = img[y:y+h, x:x+w]
            cv2.imshow("Region of Interest", img_roi)

    cv2.imshow("Capture", img)

    """Saving the captured number plate to the Scanned file by pressing the 's' key"""

    if cv2.waitKey(1) & 0xFF == ord("s"):
        cv2.imwrite("Resources/Scanned/NoPlate_"+str(counter)+".jpg",img_roi)

        """Creating a message that will show on the screen when a number plate successfully saved"""

        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)== ord("q")
        counter +=1 

cap.release()
cv2.destroyAllWindows()
