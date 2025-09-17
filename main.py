from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

def ppe_detection(file): 
    if file is None : 
        cap = cv2.VideoCapture(0)  
        cap.set(3, 1280)
        cap.set(4, 720)
    else : 
        cap = cv2.VideoCapture(file) 
    
    # Initialize YOLO model with GPU support
    model = YOLO("best.pt")
    
    # Check if CUDA is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to GPU if available
    model.to(device)

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)
    while True:
        success, img = cap.read()
        # Run inference on GPU if available
        results = model(img, stream=True, device=device)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1


                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf>0.5:
                    if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0,255) 
                    elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                        myColor =(0,255,0) 
                    else:
                        myColor = (255, 0, 0)  

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                    colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    file = r"F:\Computer_vision\PPE_detection_YOLO\Videos\ppe-2.mp4"
    ppe_detection(file)

