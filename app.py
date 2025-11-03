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

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'konsberg')
app.config['UPLOAD_FOLDER'] = os.getenv('FLASK_UPLOAD_FOLDER', 'static/files')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    try:
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
        print(f"Error in video detection: {e}")
        # Return a simple error message frame
        error_frame = cv2.imread('static/images/error.jpg') if os.path.exists('static/images/error.jpg') else None
        if error_frame is not None:
            ref, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


# Multi-camera support: UI for entering up to 3 RTSP URLs
@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    if request.method == 'POST':
        # Get up to 3 RTSP URLs from form
        rtsp_urls = [
            request.form.get('rtsp_url1', '').strip(),
            request.form.get('rtsp_url2', '').strip(),
            request.form.get('rtsp_url3', '').strip()
        ]
        # Filter out empty URLs
        rtsp_urls = [url for url in rtsp_urls if url]
        session['rtsp_urls'] = rtsp_urls
        return render_template('multi_stream.html', rtsp_urls=rtsp_urls)
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
    if rtsp_url:
        return Response(generate_frames(path_x=rtsp_url),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No RTSP URL provided", 400

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

@app.route('/api/violation_image/<violation_id>')
def get_violation_image(violation_id):
    """Serve violation image from filesystem with fallback to database"""
    try:
        db_manager = get_db_manager()
        # First try to get image path from database
        image_path = db_manager.get_violation_image_path(violation_id)
        
        if image_path and os.path.exists(image_path):
            from flask import send_file
            return send_file(image_path, mimetype='image/jpeg')
        else:
            # Fallback to database stored image
            image_data = db_manager.get_violation_image(violation_id)
            if image_data:
                return Response(image_data, mimetype='image/jpeg')
            else:
                return "Image not found", 404
            
    except Exception as e:
        return f"Error: {str(e)}", 500

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



if __name__ == "__main__":
    app.run(debug=True)