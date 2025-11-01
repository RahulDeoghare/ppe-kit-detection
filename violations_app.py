"""
Simple PPE Detection Flask App
Single database table approach for violations with screenshots
"""

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
from violations_database import get_violations_db, wait_for_database

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'konsberg')
app.config['UPLOAD_FOLDER'] = os.getenv('FLASK_UPLOAD_FOLDER', 'static/files')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    """Generate video frames with PPE detection"""
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
    """Home page"""
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    """Webcam detection page"""
    session.clear()
    return render_template('ui.html')

@app.route("/live_feed", methods=['GET', 'POST'])
def live_feed():
    """Live feed page"""
    session.clear()
    return render_template('live_feed.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    """Video upload page"""
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
    """Video stream from uploaded file"""
    return Response(generate_frames(path_x=session.get('video_path', None)), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    """Webcam stream"""
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rtsp_stream')
def rtsp_stream():
    """RTSP stream"""
    rtsp_url = request.args.get('url', '')
    if rtsp_url:
        return Response(generate_frames(path_x=rtsp_url), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No RTSP URL provided", 400

@app.route('/violations')
def violations():
    """Main violations dashboard - shows ALL violations"""
    return render_template('violations_dashboard.html')

@app.route('/api/violations')
def api_violations():
    """API endpoint to get ALL violations"""
    try:
        # Wait for database
        if not wait_for_database(max_retries=3):
            return jsonify({
                'error': 'Database not available',
                'violations': [],
                'summary': {}
            }), 503
        
        db = get_violations_db()
        
        # Get violations with filters
        violation_type = request.args.get('type')
        severity = request.args.get('severity')
        acknowledged = request.args.get('acknowledged')
        session_name = request.args.get('session')
        limit = int(request.args.get('limit', 500))
        
        # Convert acknowledged parameter
        if acknowledged == 'true':
            acknowledged = True
        elif acknowledged == 'false':
            acknowledged = False
        else:
            acknowledged = None
        
        # Get violations
        if any([violation_type, severity, acknowledged is not None, session_name]):
            violations = db.search_violations(
                violation_type=violation_type,
                severity=severity,
                acknowledged=acknowledged,
                session_name=session_name,
                limit=limit
            )
        else:
            violations = db.get_all_violations(limit=limit)
        
        # Get summary
        summary = db.get_violation_summary()
        
        # Format violations for JSON
        violations_data = []
        for violation in violations:
            violations_data.append({
                'violation_id': str(violation['violation_id']),
                'violation_type': violation['violation_type'],
                'confidence': float(violation['confidence']),
                'severity': violation['severity'],
                'person_id': violation['person_id'],
                'bbox_x1': violation['bbox_x1'],
                'bbox_y1': violation['bbox_y1'],
                'bbox_x2': violation['bbox_x2'],
                'bbox_y2': violation['bbox_y2'],
                'source_type': violation['source_type'],
                'session_name': violation['session_name'],
                'frame_number': violation['frame_number'],
                'screenshot_path': violation['screenshot_path'],
                'timestamp': violation['timestamp'].isoformat() if violation['timestamp'] else None,
                'acknowledged': violation['acknowledged'],
                'acknowledged_at': violation['acknowledged_at'].isoformat() if violation['acknowledged_at'] else None,
                'acknowledged_by': violation['acknowledged_by'],
                'notes': violation['notes']
            })
        
        return jsonify({
            'violations': violations_data,
            'summary': summary,
            'total_returned': len(violations_data)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'violations': [],
            'summary': {}
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
        
        db = get_violations_db()
        db.acknowledge_violation(violation_id, acknowledged_by)
        
        # Add notes if provided
        if notes:
            # Add notes functionality could be added to database manager
            pass
        
        return jsonify({'success': True, 'message': 'Violation acknowledged'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_violation', methods=['POST'])
def delete_violation():
    """API endpoint to delete a violation"""
    try:
        data = request.get_json()
        violation_id = data.get('violation_id')
        
        if not violation_id:
            return jsonify({'error': 'Violation ID required'}), 400
        
        db = get_violations_db()
        db.delete_violation(violation_id)
        
        return jsonify({'success': True, 'message': 'Violation deleted'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/violation_summary')
def api_violation_summary():
    """API endpoint to get violation summary statistics"""
    try:
        if not wait_for_database(max_retries=3):
            return jsonify({'error': 'Database not available'}), 503
        
        db = get_violations_db()
        summary = db.get_violation_summary()
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create screenshots directory on startup
@app.before_first_request
def create_directories():
    """Create necessary directories"""
    try:
        db = get_violations_db()
        db.create_screenshots_directory()
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
