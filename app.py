from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
import json
from datetime import datetime
from YOLO_Video import video_detection
from dotenv import load_dotenv
from database_manager import get_db_manager, wait_for_db
import threading
import time
from queue import Queue
import atexit

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'konsberg')
app.config['UPLOAD_FOLDER'] = os.getenv('FLASK_UPLOAD_FOLDER', 'static/files')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

# Global variables for background camera processing
camera_threads = {}  # Dictionary to store camera threads
camera_queues = {}   # Dictionary to store frame queues for each camera
stop_flags = {}      # Dictionary to store stop flags for each camera

def background_camera_processor(camera_id, rtsp_url):
    """Background thread function to process camera streams continuously"""
    print(f"🎥 Starting background processing for Camera {camera_id}: {rtsp_url}")
    
    stop_flags[camera_id] = False
    camera_queues[camera_id] = Queue(maxsize=10)  # Buffer up to 10 frames
    
    try:
        # Create a session name for this camera
        timestamp = datetime.now()
        session_name = f"Camera_{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Get database manager
        db_manager = get_db_manager() if wait_for_db(max_retries=3) else None
        
        # Start video detection
        frame_generator = video_detection(rtsp_url)
        
        for frame in frame_generator:
            if stop_flags.get(camera_id, True):
                print(f"🛑 Stopping background processing for Camera {camera_id}")
                break
                
            # Add frame to queue (non-blocking)
            try:
                camera_queues[camera_id].put(frame, block=False)
            except:
                # Queue is full, remove oldest frame
                try:
                    camera_queues[camera_id].get_nowait()
                    camera_queues[camera_id].put(frame, block=False)
                except:
                    pass  # If we can't manage the queue, just continue
                    
    except Exception as e:
        print(f"❌ Error in background processing for Camera {camera_id}: {e}")
    finally:
        print(f"🏁 Background processing ended for Camera {camera_id}")
        # Cleanup
        if camera_id in camera_threads:
            del camera_threads[camera_id]
        if camera_id in camera_queues:
            del camera_queues[camera_id]
        if camera_id in stop_flags:
            del stop_flags[camera_id]

def start_camera_thread(camera_id, rtsp_url):
    """Start a background thread for a camera if not already running"""
    if camera_id in camera_threads and camera_threads[camera_id].is_alive():
        print(f"📹 Camera {camera_id} is already running")
        return
    
    # Stop any existing thread first
    stop_camera_thread(camera_id)
    
    # Start new thread
    thread = threading.Thread(
        target=background_camera_processor, 
        args=(camera_id, rtsp_url),
        daemon=True
    )
    camera_threads[camera_id] = thread
    thread.start()
    print(f"🚀 Started background thread for Camera {camera_id}")

def stop_camera_thread(camera_id):
    """Stop a camera thread"""
    if camera_id in stop_flags:
        stop_flags[camera_id] = True
    
    # Wait a bit for thread to stop
    if camera_id in camera_threads:
        thread = camera_threads[camera_id]
        thread.join(timeout=2.0)
        if thread.is_alive():
            print(f"⚠️  Thread for Camera {camera_id} did not stop gracefully")

def stop_all_camera_threads():
    """Stop all camera threads"""
    print("🛑 Stopping all camera threads...")
    
    # First, signal all threads to stop
    for camera_id in list(stop_flags.keys()):
        stop_flags[camera_id] = True
    
    # Wait for all threads to stop and collect them
    threads_to_wait = []
    for camera_id in list(camera_threads.keys()):
        if camera_id in camera_threads:
            threads_to_wait.append((camera_id, camera_threads[camera_id]))
    
    # Wait for each thread
    for camera_id, thread in threads_to_wait:
        try:
            thread.join(timeout=2.0)
            if thread.is_alive():
                print(f"⚠️  Thread for Camera {camera_id} did not stop gracefully")
        except Exception as e:
            print(f"Error stopping thread for Camera {camera_id}: {e}")
    
    # Clear dictionaries after all threads have been waited on
    camera_threads.clear()
    camera_queues.clear()
    stop_flags.clear()
    print("✅ All camera threads stopped")

def generate_frames_from_queue(camera_id):
    """Generate frames from the camera's queue for live viewing"""
    while True:
        if camera_id not in camera_queues:
            # Camera not running, show error message
            error_frame = cv2.putText(
                cv2.zeros((480, 640, 3), dtype=cv2.uint8), 
                f"Camera {camera_id} not active", 
                (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            ref, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)  # Wait before showing error again
            continue
            
        try:
            # Get frame from queue with timeout
            frame = camera_queues[camera_id].get(timeout=1.0)
            ref, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except:
            # Queue empty or timeout, show waiting message
            waiting_frame = cv2.putText(
                cv2.zeros((480, 640, 3), dtype=cv2.uint8), 
                f"Waiting for Camera {camera_id}...", 
                (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            ref, buffer = cv2.imencode('.jpg', waiting_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.5)

def generate_frames(path_x=''):
    try:
        print(f"Starting video detection for: {path_x if path_x else 'webcam'}")
        yolo_output = video_detection(path_x)
        for detection_ in yolo_output:
            try:
                ref, buffer = cv2.imencode('.jpg', detection_)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
    except Exception as e:
        print(f"Error in video detection for {path_x}: {e}")
        # Create a more informative error message frame
        error_msg = f"Stream Error: {str(e)[:100]}..."
        if "No route to host" in str(e) or "Connection refused" in str(e):
            error_msg = "Camera not reachable. Check network connection and camera IP."
        elif "Could not open video" in str(e):
            error_msg = "Cannot open video stream. Check URL and camera status."
        
        # Create error frame
        error_frame = cv2.imread('static/images/error.jpg') if os.path.exists('static/images/error.jpg') else None
        if error_frame is None:
            # Create a blank error frame
            error_frame = cv2.putText(
                cv2.zeros((480, 640, 3), dtype=cv2.uint8), 
                error_msg, 
                (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        # Yield error frame continuously
        while True:
            try:
                ref, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                break

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


# Hardcoded RTSP URLs for multi-camera live feed
RTSP_URLS = [
    "rtsp://admin:India123%23@192.168.9.101:554/cam/realmonitor?channel=1&subtype=0",  # Camera 1
    "rtsp://admin:India123%23@192.168.9.102:554/cam/realmonitor?channel=1&subtype=0",  # Camera 2
    "rtsp://admin:India123%23@192.168.9.103:554/cam/realmonitor?channel=1&subtype=0"   # Camera 3
]  # Replace with your actual RTSP URLs

# Multi-camera support: UI for starting RTSP feeds
@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    if request.method == 'POST':
        # Start background threads for all RTSP URLs
        print("🎬 Starting multi-camera background processing...")
        for i, rtsp_url in enumerate(RTSP_URLS, 1):
            start_camera_thread(i, rtsp_url)
        
        session['rtsp_urls'] = RTSP_URLS
        return render_template('live_feed.html', rtsp_urls=RTSP_URLS)
    return render_template('ui.html')

@app.route("/live_feed", methods=['GET', 'POST'])
def live_feed():
    session.clear()
    return render_template('live_feed.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')



# Generic RTSP stream endpoint for any URL (for multi-stream live_feed.html)
@app.route('/rtsp_stream')
def rtsp_stream():
    rtsp_url = request.args.get('url', '').strip()
    camera_id = request.args.get('camera_id', type=int)
    
    if camera_id and camera_id in camera_queues:
        # Use background thread queue
        return Response(generate_frames_from_queue(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif rtsp_url:
        # Fallback to direct processing (for compatibility)
        return Response(generate_frames(path_x=rtsp_url),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No RTSP URL or camera_id provided", 400

@app.route('/violations')
def violations():
    """View violations dashboard"""
    return render_template('violations.html')

@app.route('/api/violations')
def api_violations():
    """API endpoint to get violations data with pagination"""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))  # Reduced from 1000 to 50
        
        # Wait for database and get manager
        if not wait_for_db(max_retries=3):
            return jsonify({
                'error': 'Database not available',
                'violations': [],
                'sessions': [],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': 0,
                    'pages': 0
                }
            }), 503
        
        db_manager = get_db_manager()
        
        # Get session statistics
        sessions = db_manager.get_session_statistics()
        
        # Get violations with pagination
        violations, total_count = db_manager.get_violations_paginated(page=page, per_page=per_page)
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        
        # Format data for JSON response
        sessions_data = []
        for session in sessions:
            sessions_data.append({
                'session_name': session['session_name'],
                'source_type': session['source_type'],
                'first_violation': session['first_violation'].isoformat() if session['first_violation'] else None,
                'last_violation': session['last_violation'].isoformat() if session['last_violation'] else None,
                'total_violations': session['total_violations'],
                'unacknowledged_violations': session['unacknowledged_violations'],
                'unique_persons': session['unique_persons'],
                'no_hardhat_count': session['no_hardhat_count'],
                'no_mask_count': session['no_mask_count'],
                'no_vest_count': session['no_vest_count'],
                'avg_confidence': float(session['avg_confidence']) if session['avg_confidence'] else 0.0
            })
        
        violations_data = []
        for violation in violations:
            violations_data.append({
                'violation_id': str(violation['violation_id']),
                'session_name': violation['session_name'],
                'violation_type': violation['violation_type'],
                'person_id': violation['person_id'],
                'frame_number': violation['frame_number'],
                'confidence': float(violation['confidence']),
                'timestamp': violation['timestamp'].isoformat() if violation['timestamp'] else None,
                'severity': violation['severity'],
                'bbox_x1': violation['bbox_x1'],
                'bbox_y1': violation['bbox_y1'],
                'bbox_x2': violation['bbox_x2'],
                'bbox_y2': violation['bbox_y2'],
                'image_path': violation['screenshot_path'],  # Bounding box screenshot
                'whole_frame_path': violation['whole_frame_path'],  # Whole frame screenshot
                'acknowledged': violation['acknowledged'],
                'acknowledged_by': violation['acknowledged_by'],
                'acknowledged_at': violation['acknowledged_at'].isoformat() if violation['acknowledged_at'] else None,
                'notes': violation['notes']
            })
        
        return jsonify({
            'violations': violations_data,
            'sessions': sessions_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'total_violations': total_count,
            'total_sessions': len(sessions_data)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'violations': [],
            'sessions': [],
            'pagination': {
                'page': 1,
                'per_page': 50,
                'total': 0,
                'pages': 0
            }
        }), 500

@app.route('/api/acknowledge_violation', methods=['POST'])
def acknowledge_violation():
    """API endpoint to acknowledge a violation"""
    try:
        data = request.get_json()
        violation_id = data.get('violation_id')
        acknowledged_by = data.get('acknowledged_by', 'web_user')
        notes = data.get('notes', '')
        
        if not violation_id:
            return jsonify({'error': 'Violation ID required'}), 400
        
        db_manager = get_db_manager()
        db_manager.acknowledge_violation(violation_id, acknowledged_by, notes)
        
        return jsonify({'success': True, 'message': 'Violation acknowledged'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera_status')
def camera_status():
    """API endpoint to get status of all cameras"""
    status = {}
    for i in range(1, len(RTSP_URLS) + 1):
        is_running = i in camera_threads and camera_threads[i].is_alive()
        status[f"camera_{i}"] = {
            "running": is_running,
            "rtsp_url": RTSP_URLS[i-1] if i-1 < len(RTSP_URLS) else None,
            "thread_alive": camera_threads.get(i).is_alive() if i in camera_threads else False,
            "queue_size": camera_queues[i].qsize() if i in camera_queues else 0
        }
    return jsonify(status)

@app.route('/api/stop_cameras', methods=['POST'])
def stop_cameras():
    """API endpoint to stop all camera threads"""
    stop_all_camera_threads()
    return jsonify({"message": "All cameras stopped"})

@app.route('/api/start_cameras', methods=['POST'])
def start_cameras():
    """API endpoint to start all camera threads"""
    for i, rtsp_url in enumerate(RTSP_URLS, 1):
        start_camera_thread(i, rtsp_url)
    return jsonify({"message": "All cameras started"})

@app.route('/api/whole_frame_image/<violation_id>')
def get_whole_frame_image(violation_id):
    """Serve whole frame image from filesystem with fallback to database"""
    try:
        db_manager = get_db_manager()
        # First try to get image path from database
        image_path = db_manager.get_whole_frame_image_path(violation_id)
        
        if image_path and os.path.exists(image_path):
            from flask import send_file
            return send_file(image_path, mimetype='image/jpeg')
        else:
            # Fallback to database stored image
            image_data = db_manager.get_whole_frame_image(violation_id)
            if image_data:
                return Response(image_data, mimetype='image/jpeg')
            else:
                return "Image not found", 404
            
    except Exception as e:
        return f"Error: {str(e)}", 500



# @app.teardown_appcontext  # Removed - was stopping threads on every request!
# def cleanup_background_threads(exception=None):
#     """Clean up background threads when the app shuts down"""
#     stop_all_camera_threads()

if __name__ == "__main__":
    # Register cleanup function to run only when the process exits
    atexit.register(stop_all_camera_threads)
    
    try:
        app.run(debug=True)
    finally:
        # Ensure threads are stopped when app exits
        stop_all_camera_threads()