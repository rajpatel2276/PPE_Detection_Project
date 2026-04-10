from ultralytics import YOLO
import cv2
import cvzone
import os
import math

os.environ["QT_QPA_PLATFORM"] = "xcb"

model = YOLO("ppe.pt")

cap = cv2.VideoCapture("../Project3-PPE_Detection/Videos/ppe-3.mp4")

# url = "http://10.143.164.141:4747/video" 
# cap = cv2.VideoCapture(url)
# cap.set(3,1280)
# cap.set(4,720)

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask',
               'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck',
                 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer',
                   'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0,0,255)
while True:
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame. Check if the video path is correct.")
        break

    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            # cvzone.cornerRect(img,(x1,y1,w,h))
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask" or currentClass == 'Gloves':
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1),max(35,y1)),
                               scale=1,thickness=1,colorB=myColor,colorT=(255,255,255),colorR=myColor,offset=4)
                cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()