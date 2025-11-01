import os
from ultralytics import YOLO
import cv2
import math
import time
import csv
import logging
import json
import os
from datetime import datetime
import torch
from database_manager import get_db_manager, wait_for_db


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

def video_detection(path_x):
    # Initialize database
    if not wait_for_db():
        print("Warning: Database not available, falling back to CSV logging only")
        db_manager = None
    else:
        db_manager = get_db_manager()
    
    # Generate session name
    timestamp = datetime.now()
    session_name = f"Detection_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    source_type = "webcam" if path_x == 0 else "video_file" if isinstance(path_x, str) else "unknown"
    source_path = str(path_x) if path_x != 0 else None
    
    # Keep CSV logging for backward compatibility
    with open('detection_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Person ID', 'Items Detected'])

    # Check if the input is an image or video
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    is_single_image = isinstance(path_x, str) and path_x.lower().endswith(image_extensions)
    
    try:
        if is_single_image:
            # For single images, process just that one image
            img = cv2.imread(path_x)
            if img is None:
                print(f"Error: Could not load image {path_x}")
                return
            
            # Process single image
            yield from process_single_image(img, path_x, session_name, source_type, source_path, db_manager)
        else:
            # For videos or webcam
            yield from process_video_stream(path_x, session_name, source_type, source_path, db_manager)
    finally:
        print(f"Detection session '{session_name}' completed")

def process_single_image(img, path_x, session_name, source_type, source_path, db_manager=None):
    """Process a single image file"""
    
    # Initialize YOLO model with GPU support
    model = YOLO("YOLO-Weights/ppe.pt")
    
    # Check if CUDA is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to GPU if available
    model.to(device)
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']

    # Validate and process the image
    if img is None or img.size == 0:
        print("Warning: Empty image received")
        return

    # Ensure image has exactly 3 channels (RGB) for YOLO model
    if len(img.shape) == 3 and img.shape[2] == 4:  # If image has 4 channels (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to 3 channels
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    person_count = 0
    persons_violations = {}

    try:
        # Run inference on GPU if available
        results = model(img, stream=True, device=device)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'

                if conf > 0.5:
                    if class_name == 'Person':
                        person_count += 1
                        persons_violations[person_count] = []
                    
                    elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                        if person_count in persons_violations:
                            persons_violations[person_count].append(class_name)
                        
                        # Save violation to JSON (keep for backward compatibility)
                        timestamp = datetime.now()
                        save_violation_to_json(
                            violation_type=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            person_id=person_count,
                            frame_number=1  # Single image, so frame 1
                        )
                        
                        # Save violation with screenshot to database - SIMPLIFIED APPROACH
                        if db_manager:
                            try:
                                # Extract the region of interest (bounding box area) for screenshot
                                violation_image = img[y1:y2, x1:x2].copy()
                                
                                db_manager.save_violation_with_image(
                                    session_name=session_name,
                                    violation_type=class_name,
                                    person_id=person_count,
                                    frame_number=1,
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                    image_frame=violation_image,
                                    whole_frame=img.copy(),
                                    severity="HIGH" if conf > 0.8 else "MEDIUM",
                                    source_type=source_type,
                                    source_path=source_path
                                )
                                print(f"Saved violation: {class_name} for person {person_count}")
                            except Exception as e:
                                print(f"Failed to save violation to database: {e}")
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    else:
                        # Non-violation objects - just draw green box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Handle violations for single image (CSV logging)
        if persons_violations:
            aggregated_violations = aggregate_violations(persons_violations)
            
            for person_id, detected_items in aggregated_violations.items():
                log_detection_to_csv(person_id, detected_items)

        # For single image, yield the processed image continuously to maintain the stream
        for _ in range(60):  # Show the image for approximately 60 frames
            yield img
            
    except Exception as e:
        print(f"Error during image processing: {e}")
        return

def process_video_stream(path_x, session_name, source_type, source_path, db_manager=None):
    """Process video file or webcam stream"""

    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    
    # Set video capture properties for better compatibility
    if isinstance(video_capture, str):  # If it's a file path
        # Try different backends for better codec support
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_capture, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_capture, cv2.CAP_DSHOW)
            
    if not cap.isOpened():
        print(f"Error: Could not open video {path_x}")
        return
    
    # Get video properties for debugging
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Initialize YOLO model with GPU support
    model = YOLO("YOLO-Weights/ppe.pt")
    
    # Check if CUDA is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to GPU if available
    model.to(device)
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']

    aggregated_violations = {}
    frame_count = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from video.")
                break

            # Validate frame
            if img is None or img.size == 0:
                print("Warning: Empty frame received, skipping...")
                continue

            frame_count += 1
            person_count = 0
            persons_violations = {}

            # Ensure image has exactly 3 channels (RGB) for YOLO model
            if len(img.shape) == 3 and img.shape[2] == 4:  # If image has 4 channels (RGBA)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to 3 channels
            elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # If grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

            try:
                # Run inference on GPU if available
                results = model(img, stream=True, device=device)
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    label = f'{class_name}{conf}'

                    if conf > 0.5:
                        if class_name == 'Person':
                            person_count += 1
                            persons_violations[person_count] = []
                        
                        elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                            if person_count in persons_violations:
                                persons_violations[person_count].append(class_name)
                            
                            # Save violation to JSON (keep for backward compatibility)
                            timestamp = datetime.now()
                            save_violation_to_json(
                                violation_type=class_name,
                                confidence=conf,
                                bbox=(x1, y1, x2, y2),
                                timestamp=timestamp,
                                person_id=person_count,
                                frame_number=frame_count
                            )
                            
                            # Save violation with screenshot to database - SIMPLIFIED APPROACH
                            if db_manager:
                                try:
                                    # Extract the region of interest (bounding box area) for screenshot
                                    violation_image = img[y1:y2, x1:x2].copy()
                                    
                                    db_manager.save_violation_with_image(
                                        session_name=session_name,
                                        violation_type=class_name,
                                        person_id=person_count,
                                        frame_number=frame_count,
                                        bbox=(x1, y1, x2, y2),
                                        confidence=conf,
                                        image_frame=violation_image,
                                        whole_frame=img.copy(),
                                        severity="HIGH" if conf > 0.8 else "MEDIUM",
                                        source_type=source_type,
                                        source_path=source_path
                                    )
                                    print(f"Saved violation: {class_name} for person {person_count} at frame {frame_count}")
                                except Exception as e:
                                    print(f"Failed to save violation to database: {e}")
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        else:
                            # Non-violation objects - just draw green box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            if persons_violations:
                aggregated_violations = aggregate_violations(persons_violations)

            # CSV logging for backward compatibility
            for person_id, detected_items in aggregated_violations.items():
                log_detection_to_csv(person_id, detected_items)

            yield img
    finally:
        cap.release()
        cv2.destroyAllWindows()

