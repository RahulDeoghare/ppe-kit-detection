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

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
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

@app.route('/rtsp_stream')
def rtsp_stream():
    rtsp_url = request.args.get('url', '')
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
    """API endpoint to get violations data"""
    try:
        # Wait for database and get manager
        if not wait_for_db(max_retries=3):
            return jsonify({
                'error': 'Database not available',
                'violations': [],
                'sessions': []
            }), 503
        
        db_manager = get_db_manager()
        
        # Get recent sessions
        sessions = db_manager.get_session_statistics()
        
        # Get unacknowledged violations
        violations = db_manager.get_unacknowledged_violations()
        
        # Format data for JSON response
        sessions_data = []
        for session in sessions:
            sessions_data.append({
                'session_id': str(session['session_id']),
                'session_name': session['session_name'],
                'source_type': session['source_type'],
                'start_time': session['start_time'].isoformat() if session['start_time'] else None,
                'end_time': session['end_time'].isoformat() if session['end_time'] else None,
                'total_violations': session['total_violations'],
                'unacknowledged_violations': session['unacknowledged_violations']
            })
        
        violations_data = []
        for violation in violations:
            violations_data.append({
                'violation_id': str(violation['violation_id']),
                'violation_type': violation['violation_type'],
                'person_id': violation['person_id'],
                'confidence': float(violation['confidence']),
                'timestamp': violation['timestamp'].isoformat() if violation['timestamp'] else None,
                'severity': violation['severity'],
                'frame_number': violation['frame_number'],
                'session_name': violation['session_name']
            })
        
        return jsonify({
            'violations': violations_data,
            'sessions': sessions_data,
            'total_violations': len(violations_data),
            'total_sessions': len(sessions_data)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'violations': [],
            'sessions': []
        }), 500

@app.route('/api/acknowledge_violation', methods=['POST'])
def acknowledge_violation():
    """API endpoint to acknowledge a violation"""
    try:
        data = request.get_json()
        violation_id = data.get('violation_id')
        acknowledged_by = data.get('acknowledged_by', 'web_user')
        
        if not violation_id:
            return jsonify({'error': 'Violation ID required'}), 400
        
        db_manager = get_db_manager()
        db_manager.acknowledge_violation(violation_id, acknowledged_by)
        
        return jsonify({'success': True, 'message': 'Violation acknowledged'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)