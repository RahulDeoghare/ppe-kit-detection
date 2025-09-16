import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
import cv2
import math
import time
import csv
from twilio.rest import Client
import logging
import json
import os
from datetime import datetime


last_email_time = 0
email_cooldown = 20  


logging.basicConfig(filename='alert_timing.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def save_violation_to_json(violation_type, confidence, bbox, timestamp, person_id=None, frame_number=None):
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
    
    if person_id is not None:
        violation_data["person_id"] = person_id
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
    return filepath

violation_tips = {
    'Hardhat': 'Wearing a hardhat protects you from head injuries caused by falling objects or impact.',
    'Mask': 'Wearing a mask helps protect you and others from airborne hazards and infectious agents.',
    'NO-Hardhat': 'Not wearing a hardhat can lead to severe head injuries due to falling objects or impact.',
    'NO-Mask': 'Not wearing a mask increases the risk of exposure to airborne hazards and infectious agents.',
    'NO-Safety Vest': 'Not wearing a safety vest makes you less visible, increasing the risk of accidents in low-light conditions.',
    'Safety Vest': 'Wearing a safety vest ensures that you are visible to others, especially in low-light conditions.',
    'Person': 'Ensure all safety gear is worn properly to avoid injuries.',
    'Safety Cone': 'Safety cones help in marking safe areas and guiding pedestrian or vehicular traffic.',
    'machinery': 'Machinery should be operated with care, ensuring all safety protocols are followed.',
    'vehicle': 'Vehicles should be operated carefully in designated areas to prevent accidents.'
}

def log_time_taken(action, start_time):
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"{action} took {duration:.2f} seconds")
    print(f"{action} took {duration:.2f} seconds")


def adjust_cooldown(violations):
    severe_violations = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
    if any(v in violations for v in severe_violations):
        return 30  
    else:
        return 60 

def send_email_alert(subject, body, to_email):
    start_time = time.time()  
    sender_email = "3021113@extc.fcrit.ac.in"
    sender_password = "3021113"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        print("Email sent successfully.")
        log_time_taken("Email sending", start_time)
    except Exception as e:
        logging.error(f"Failed to send email. Error: {e}")

def send_sms_alert(body, to_number):
    start_time = time.time() 
    # Load Twilio credentials from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'YOUR_ACCOUNT_SID_HERE')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'YOUR_AUTH_TOKEN_HERE')
    client = Client(account_sid, auth_token)

    from_number = os.getenv('TWILIO_PHONE_NUMBER', '+1234567890')  # Replace with your Twilio phone number

    try:
        message = client.messages.create(
            body=body,
            from_=from_number,
            to=to_number
        )
        print(f"SMS sent successfully. SID: {message.sid}")
        log_time_taken("SMS sending", start_time)
    except Exception as e:
        logging.error(f"Failed to send SMS. Error: {e}")


def aggregate_violations(persons_violations):
    aggregated_violations = {}
    for person_id, violations in persons_violations.items():
        if person_id not in aggregated_violations:
            aggregated_violations[person_id] = []
        aggregated_violations[person_id].extend(violations)
    return aggregated_violations

def log_detection_to_csv(person_id, detected_items):

    with open('detection_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        items_str = ', '.join(detected_items)
        writer.writerow([timestamp, person_id, items_str])

def video_detection(path_x, email_recipient, sms_recipient):
    global last_email_time
    global email_cooldown  

    with open('detection_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Person ID', 'Items Detected'])

    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    if not cap.isOpened():
        print(f"Error: Could not open video {path_x}")
        return
    model = YOLO("YOLO-Weights/ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']

    aggregated_violations = {}
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to read frame from video.")
            break

        frame_count += 1
        person_count = 0
        persons_violations = {}

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'

                current_time = time.time()

                if conf > 0.5:
                    if class_name == 'Person':
                        person_count += 1
                        persons_violations[person_count] = []
                    elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                        if person_count in persons_violations:
                            persons_violations[person_count].append(class_name)
                        
                        # Save violation to JSON
                        timestamp = datetime.now()
                        save_violation_to_json(
                            violation_type=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            person_id=person_count,
                            frame_number=frame_count
                        )
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        if persons_violations:

            aggregated_violations = aggregate_violations(persons_violations)


        for person_id, detected_items in aggregated_violations.items():
            log_detection_to_csv(person_id, detected_items)

        if 'email_cooldown' not in globals():
            email_cooldown = 60 

        if aggregated_violations and current_time - last_email_time > email_cooldown:
            email_cooldown = adjust_cooldown([v for sublist in aggregated_violations.values() for v in sublist])
            subject = f"PPE Violation Detected - {len(aggregated_violations)} Person(s)"
            body_lines = []
            for person_id, violations in aggregated_violations.items():
                violation_messages = [f"{v}: {violation_tips[v]}" for v in violations]
                body_lines.append(f"Person {person_id} not detected items:\n" + "\n".join(violation_messages))
            body = "\n\n".join(body_lines)
            print(f"Sending email and SMS: {subject} | {body}")
            send_email_alert(subject, body, email_recipient)
            send_sms_alert(body, sms_recipient)
            last_email_time = current_time

        yield img

    cap.release()
    cv2.destroyAllWindows()

