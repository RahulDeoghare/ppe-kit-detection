import os
from ultralytics import YOLO
import cv2
import math
import time
import csv
import logging
import json
from datetime import datetime
import torch
from database_manager import get_db_manager, wait_for_db
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='alert_timing.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration flags from environment
SAVE_TO_JSON = os.getenv('SAVE_VIOLATIONS_TO_JSON', 'false').lower() == 'true'
SAVE_TO_CSV = os.getenv('SAVE_VIOLATIONS_TO_CSV', 'true').lower() == 'true'
SAVE_TO_DB = os.getenv('SAVE_VIOLATIONS_TO_DB', 'true').lower() == 'true'

def save_violation_to_json(violation_type, confidence, bbox, timestamp, person_id=None, frame_number=None):
    """
    Save violation data to a JSON file (optional backup logging)
    """
    if not SAVE_TO_JSON:
        return
        
    violation_data = {
        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
        "violation_type": violation_type,
        "confidence": confidence,
        "bbox": bbox,
        "person_id": person_id,
        "frame_number": frame_number,
    }

    try:
        with open('violations.jsonl', 'a') as jf:
            jf.write(json.dumps(violation_data) + '\n')
    except Exception as e:
        logger.error(f"Failed to write violation JSON: {e}")

def log_detection_to_csv(person_id, detected_items):
    """Log detection to CSV (optional backup logging)"""
    if not SAVE_TO_CSV:
        return
        
    try:
        with open('detection_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            items_str = ', '.join(detected_items)
            writer.writerow([timestamp, person_id, items_str])
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}")


def aggregate_violations(persons_violations):
    """Simple aggregator for persons_violations dict."""
    aggregated = {}
    for pid, items in persons_violations.items():
        # keep items as-is; ensure it's a list
        aggregated[pid] = list(items) if items is not None else []
    return aggregated

def video_detection(path_x):
    # Initialize database connection
    db_manager = None
    if SAVE_TO_DB:
        if not wait_for_db():
            logger.error("Database not available! Violations will not be saved to database.")
            if not (SAVE_TO_CSV or SAVE_TO_JSON):
                raise Exception("No logging method available - database unavailable and backup logging disabled")
        else:
            db_manager = get_db_manager()
            logger.info("✅ Database connection established - violations will be saved directly to database")
    
    # Generate session name
    timestamp = datetime.now()
    session_name = f"Detection_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    source_type = "webcam" if path_x == 0 else "video_file" if isinstance(path_x, str) else "unknown"
    source_path = str(path_x) if path_x != 0 else None
    
    logger.info(f"🔍 Starting detection session: {session_name}")
    logger.info(f"📊 Logging config - DB: {SAVE_TO_DB}, CSV: {SAVE_TO_CSV}, JSON: {SAVE_TO_JSON}")
    
    # Initialize CSV if enabled
    if SAVE_TO_CSV:
        try:
            with open('detection_log.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Person ID', 'Items Detected'])
        except Exception as e:
            logger.error(f"Failed to initialize CSV logging: {e}")

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
    classNames = ['Hardhat', 'NO-Hardhat', 'NO-Safety Vest',
                  'Safety Vest',]

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
                # Defensive extraction of box coordinates / conf / class
                try:
                    # coords may be nested (tensor) or direct list
                    coords = None
                    if hasattr(box, 'xyxy'):
                        try:
                            coords = box.xyxy[0]
                        except Exception:
                            coords = box.xyxy
                    if coords is None:
                        raise ValueError('No coordinates in box')

                    x1, y1, x2, y2 = map(int, map(float, coords))

                    # confidence
                    try:
                        conf_val = float(box.conf[0])
                    except Exception:
                        conf_val = float(box.conf) if hasattr(box, 'conf') else 0.0
                    conf = math.ceil((conf_val * 100)) / 100

                    # class index
                    try:
                        cls_idx = int(box.cls[0])
                    except Exception:
                        cls_idx = int(box.cls) if hasattr(box, 'cls') else -1

                    if cls_idx < 0 or cls_idx >= len(classNames):
                        print(f"⚠️  Skipping unknown class index: {cls_idx}")
                        continue

                    class_name = classNames[cls_idx]
                except Exception as e:
                    print(f"⚠️  Skipping malformed detection box: {e}")
                    continue

                label = f'{class_name}{conf}'

                if conf > 0.5:
                    if class_name == 'Person':
                        person_count += 1
                        persons_violations[person_count] = []
                    
                    elif class_name in ['NO-Hardhat', 'NO-Safety Vest']:
                        if person_count in persons_violations:
                            persons_violations[person_count].append(class_name)
                        
                        timestamp = datetime.now()
                        violation_saved = False
                        
                        # PRIMARY: Save violation to database
                        if db_manager and SAVE_TO_DB:
                            try:
                                # Extract the region of interest (bounding box area) for screenshot
                                violation_image = img[y1:y2, x1:x2].copy()
                                
                                violation_id = db_manager.save_violation_with_image(
                                    session_name=session_name,
                                    violation_type=class_name,
                                    person_id=person_count,
                                    frame_number=1,
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                    image_frame=violation_image,
                                    whole_frame=img.copy(),
                                    severity="high" if conf > 0.8 else "medium",
                                    source_type=source_type,
                                    source_path=source_path
                                )
                                logger.info(f"✅ Saved violation to DB: {class_name} for person {person_count} (ID: {violation_id})")
                                violation_saved = True
                            except Exception as e:
                                logger.error(f"❌ Failed to save violation to database: {e}")
                        
                        # BACKUP: Save to JSON if enabled or if database save failed
                        if not violation_saved or SAVE_TO_JSON:
                            save_violation_to_json(
                                violation_type=class_name,
                                confidence=conf,
                                bbox=(x1, y1, x2, y2),
                                timestamp=timestamp,
                                person_id=person_count,
                                frame_number=1
                            )
                        
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
    """Process video file or webcam stream with improved RTSP handling"""

    video_capture = path_x
    cap = None
    
    # Try different backends for RTSP streams
    if isinstance(video_capture, str) and video_capture.startswith('rtsp://'):
        print(f"Attempting to open RTSP stream: {video_capture}")
        
        # Try FFMPEG backend first for RTSP
        cap = cv2.VideoCapture(video_capture, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("FFMPEG backend failed, trying default backend...")
            cap = cv2.VideoCapture(video_capture)
            
        if not cap.isOpened():
            print(f"Failed to open RTSP stream: {video_capture}")
            print("Possible issues:")
            print("  - Camera not reachable (check network connectivity)")
            print("  - Incorrect RTSP URL or credentials")
            print("  - Camera not powered on or configured")
            print("  - Firewall blocking RTSP port (554)")
            return
    else:
        # For regular video files or webcam
        cap = cv2.VideoCapture(video_capture)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {path_x}")
        return
    
    # Get video properties for debugging
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Set buffer size for RTSP streams to reduce latency
    if isinstance(video_capture, str) and video_capture.startswith('rtsp://'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize YOLO model with GPU support
    model = YOLO("YOLO-Weights/ppe.pt")
    
    # Check if CUDA is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to GPU if available
    model.to(device)
    classNames = ['Hardhat', 'NO-Hardhat', 'NO-Safety Vest',
                  'Safety Vest',]

    aggregated_violations = {}
    frame_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10

    try:
        while True:
            success, img = cap.read()
            if not success:
                consecutive_errors += 1
                print(f"Error: Failed to read frame from video (attempt {consecutive_errors}/{max_consecutive_errors}).")
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive frame read errors. Stopping stream.")
                    break
                continue
            
            consecutive_errors = 0  # Reset error counter on successful read
            img = cv2.resize(img, (1280, 720))

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
                    # Defensive extraction for video processing as well
                    try:
                        coords = None
                        if hasattr(box, 'xyxy'):
                            try:
                                coords = box.xyxy[0]
                            except Exception:
                                coords = box.xyxy
                        if coords is None:
                            raise ValueError('No coordinates in box')

                        x1, y1, x2, y2 = map(int, map(float, coords))

                        try:
                            conf_val = float(box.conf[0])
                        except Exception:
                            conf_val = float(box.conf) if hasattr(box, 'conf') else 0.0
                        conf = math.ceil((conf_val * 100)) / 100

                        try:
                            cls_idx = int(box.cls[0])
                        except Exception:
                            cls_idx = int(box.cls) if hasattr(box, 'cls') else -1

                        if cls_idx < 0 or cls_idx >= len(classNames):
                            print(f"⚠️  Skipping unknown class index: {cls_idx}")
                            continue

                        class_name = classNames[cls_idx]
                    except Exception as e:
                        print(f"⚠️  Skipping malformed detection box (video): {e}")
                        continue

                    label = f'{class_name}{conf}'

                    if conf > 0.5:
                        if class_name == 'Person':
                            person_count += 1
                            persons_violations[person_count] = []
                        
                        elif class_name in ['NO-Hardhat', 'NO-Safety Vest']:
                            if person_count in persons_violations:
                                persons_violations[person_count].append(class_name)
                            
                            timestamp = datetime.now()
                            violation_saved = False
                            
                            # PRIMARY: Save violation to database
                            if db_manager and SAVE_TO_DB:
                                try:
                                    # Extract the region of interest (bounding box area) for screenshot
                                    violation_image = img[y1:y2, x1:x2].copy()
                                    
                                    violation_id = db_manager.save_violation_with_image(
                                        session_name=session_name,
                                        violation_type=class_name,
                                        person_id=person_count,
                                        frame_number=frame_count,
                                        bbox=(x1, y1, x2, y2),
                                        confidence=conf,
                                        image_frame=violation_image,
                                        whole_frame=img.copy(),
                                        severity="high" if conf > 0.8 else "medium",
                                        source_type=source_type,
                                        source_path=source_path
                                    )
                                    logger.info(f"✅ Saved violation to DB: {class_name} for person {person_count} at frame {frame_count} (ID: {violation_id})")
                                    violation_saved = True
                                except Exception as e:
                                    logger.error(f"❌ Failed to save violation to database: {e}")
                            
                            # BACKUP: Save to JSON if enabled or if database save failed
                            if not violation_saved or SAVE_TO_JSON:
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
        if cap:
            cap.release()
        cv2.destroyAllWindows()

