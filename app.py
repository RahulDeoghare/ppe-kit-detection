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

def generate_frames(path_x='', email_recipient='3021130@extc.fcrit.ac.in', sms_recipient='+918452992560'):
    try:
        yolo_output = video_detection(path_x, email_recipient, sms_recipient)
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
    email_recipient = "recipient@example.com"
    sms_recipient = "+918452992560"
    return Response(generate_frames(path_x=session.get('video_path', None), email_recipient=email_recipient, sms_recipient=sms_recipient), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rtsp_stream')
def rtsp_stream():
    rtsp_url = request.args.get('url', '')
    if rtsp_url:
        email_recipient = "recipient@example.com"
        sms_recipient = "+918452992560"
        return Response(generate_frames(path_x=rtsp_url, email_recipient=email_recipient, sms_recipient=sms_recipient), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No RTSP URL provided", 400



if __name__ == "__main__":
    app.run(debug=True)