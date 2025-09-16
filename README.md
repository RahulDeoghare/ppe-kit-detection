# PPE Kit Detection

A Personal Protective Equipment (PPE) detection system using YOLO (You Only Look Once) deep learning model to identify safety violations in real-time video streams.

## Features

- Real-time PPE detection from video streams
- Web-based interface for monitoring
- Violation logging and alerts
- Support for multiple video sources
- Detection of safety helmets, vests, and other PPE items

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO
- Flask (for web interface)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RahulDeoghare/ppe-kit-detection.git
cd ppe-kit-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLO model files:
```bash
# The yolov8n.pt model will be automatically downloaded when first run
# For custom models (best.pt, ppe.pt), you'll need to train them or obtain them separately
```

4. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Twilio credentials (if using SMS alerts)
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
```

## Usage

### Running the Web Application
```bash
python app.py
```

### Running Video Detection
```bash
python main.py
```

### Running YOLO Video Processing
```bash
python YOLO_Video.py
```

## Project Structure

```
PPE Detection/
├── app.py                 # Flask web application
├── main.py               # Main detection script
├── YOLO_Video.py         # YOLO video processing
├── best.pt               # Trained model weights
├── ppe.pt                # PPE-specific model weights
├── detection_log.csv     # Detection logs
├── static/               # Static web files
├── templates/            # HTML templates
├── Videos/               # Sample videos
├── YOLO-Weights/         # Model weights directory
└── violations/           # Generated violation logs (auto-created)
```

## Models

- `best.pt` - Custom trained model for PPE detection (not included in repo - too large)
- `ppe.pt` - PPE-specific YOLO model (not included in repo - too large)
- `yolov8n.pt` - Base YOLOv8 nano model (downloaded automatically)

**Note:** The large model files are excluded from the repository. You'll need to:
1. Train your own models, or
2. Download pre-trained models from appropriate sources
3. Place them in the root directory or YOLO-Weights folder

## Output

- Violation logs are saved in the `violations/` directory
- Detection results are logged in `detection_log.csv`
- Alert timing is tracked in `alert_timing.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
