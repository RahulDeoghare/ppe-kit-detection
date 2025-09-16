from ultralytics import YOLO
import cv2
import cvzone
import math
import json
import os
from datetime import datetime

def save_violation_to_json(violation_type, confidence, bbox, timestamp, frame_number=None):
    """
    Save violation data to a JSON file
    """
    violation_data = {
        "timestamp": timestamp,
        "violation_type": violation_type,
        "confidence": confidence,
        "bounding_box": {
            "x1": bbox[0],
            "y1": bbox[1], 
            "x2": bbox[2],
            "y2": bbox[3]
        }
    }
    
    if frame_number is not None:
        violation_data["frame_number"] = frame_number
    
    # Create violations directory if it doesn't exist
    violations_dir = "violations"
    if not os.path.exists(violations_dir):
        os.makedirs(violations_dir)
    
    # Generate filename with timestamp
    filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.json"
    filepath = os.path.join(violations_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(violation_data, f, indent=4, default=str)
    
    print(f"Violation saved to: {filepath}")

def ppe_detection(file): 
    if file is None : 
        cap = cv2.VideoCapture(0)  
        cap.set(3, 1280)
        cap.set(4, 720)
    else : 
        cap = cv2.VideoCapture(file) 
    model = YOLO("best.pt")

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        frame_count += 1
        results = model(img, stream=True)
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
                        # Save violation to JSON
                        timestamp = datetime.now()
                        save_violation_to_json(
                            violation_type=currentClass,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            frame_number=frame_count
                        )
                    elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                        myColor =(0,255,0) 
                    else:
                        myColor = (255, 0, 0)  

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                    colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file = "test.mp4"  # Updated to use local video file
    ppe_detection(file)

