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

app = Flask(__name__)

app.config['SECRET_KEY'] = 'konsberg'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x='', email_recipient='nidhish.waghmare88@gmail.com', sms_recipient='+918452992560'):
    yolo_output = video_detection(path_x, email_recipient, sms_recipient)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
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
    email_recipient = "recipient@example.com"
    sms_recipient = "+918452992560"
    return Response(generate_frames(path_x=session.get('video_path', None), email_recipient=email_recipient, sms_recipient=sms_recipient), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations_dashboard')
def violations_dashboard():
    """Serve the violations dashboard HTML page"""
    return render_template('violations.html')

@app.route('/violations')
def violations():
    """View all violation JSON files"""
    violations_dir = 'violations'
    violation_files = []
    
    if os.path.exists(violations_dir):
        files = os.listdir(violations_dir)
        json_files = [f for f in files if f.endswith('.json')]
        json_files.sort(reverse=True)  # Most recent first
        
        for filename in json_files:
            filepath = os.path.join(violations_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    violation_data = json.load(f)
                violation_data['filename'] = filename
                violation_files.append(violation_data)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return jsonify({
        'total_violations': len(violation_files),
        'violations': violation_files
    })

@app.route('/violations/<filename>')
def get_violation(filename):
    """Get specific violation JSON file"""
    violations_dir = 'violations'
    filepath = os.path.join(violations_dir, filename)
    
    if os.path.exists(filepath) and filename.endswith('.json'):
        try:
            with open(filepath, 'r') as f:
                violation_data = json.load(f)
            return jsonify(violation_data)
        except Exception as e:
            return jsonify({'error': f'Error reading file: {e}'}), 500
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/violations/summary')
def violations_summary():
    """Get summary statistics of violations"""
    violations_dir = 'violations'
    summary = {
        'total_violations': 0,
        'violation_types': {},
        'recent_violations': []
    }
    
    if os.path.exists(violations_dir):
        files = os.listdir(violations_dir)
        json_files = [f for f in files if f.endswith('.json')]
        summary['total_violations'] = len(json_files)
        
        # Get recent violations (last 10)
        json_files.sort(reverse=True)
        for filename in json_files[:10]:
            filepath = os.path.join(violations_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    violation_data = json.load(f)
                summary['recent_violations'].append({
                    'filename': filename,
                    'violation_type': violation_data.get('violation_type'),
                    'timestamp': violation_data.get('timestamp'),
                    'confidence': violation_data.get('confidence')
                })
                
                # Count violation types
                violation_type = violation_data.get('violation_type')
                if violation_type:
                    summary['violation_types'][violation_type] = summary['violation_types'].get(violation_type, 0) + 1
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return jsonify(summary)

if __name__ == "__main__":
    app.run(debug=True)