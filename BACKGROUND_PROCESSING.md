# Background Camera Processing - Always-On Detection

## Overview
The PPE detection system now runs camera processing in **background threads** that continue operating even when you navigate away from the live feed page. This ensures continuous violation detection and database logging.

## How It Works

### Background Threads
- Each RTSP camera runs in its own dedicated background thread
- Threads process video frames continuously using YOLOv8
- Violations are detected and saved to database regardless of page navigation
- Live feed page taps into existing background processing (no duplicate processing)

### Thread Management
- **Start**: Click "Start Multi-Camera Feed" to launch background threads
- **Monitor**: Check camera status via API or live feed page
- **Stop**: Threads stop automatically on app shutdown or via API calls

## Key Benefits

### ✅ Continuous Detection
- Violations detected 24/7 while cameras are active
- Database logging continues even when viewing violations page
- No interruption when switching between pages

### ✅ Resource Efficient
- One YOLO model instance per camera thread
- Frame buffering prevents memory overflow
- Automatic cleanup on thread termination

### ✅ Live Monitoring
- Live feed displays real-time frames from background processing
- Status indicators show which cameras are active
- API endpoints for programmatic control

## API Endpoints

### Camera Status
```
GET /api/camera_status
```
Returns JSON with status of all cameras:
```json
{
  "camera_1": {
    "running": true,
    "rtsp_url": "rtsp://...",
    "thread_alive": true,
    "queue_size": 5
  }
}
```

### Control Cameras
```
POST /api/start_cameras  # Start all background threads
POST /api/stop_cameras   # Stop all background threads
```

## Usage Workflow

1. **Start Detection**: Visit `/webcam` → Click "Start Multi-Camera Feed"
2. **Background Processing**: Cameras run continuously in background threads
3. **Monitor Violations**: Navigate to `/violations` - detection continues
4. **View Live Feed**: Return to live feed anytime - taps into existing streams
5. **Stop When Needed**: Use API or restart app to stop processing

## Technical Details

- **Threading**: Each camera uses `threading.Thread` with daemon=True
- **Queue System**: Frames buffered in `Queue(maxsize=10)` for live viewing
- **Error Handling**: Automatic retry and graceful thread termination
- **Database**: Each camera session tracked separately with unique names
- **Cleanup**: Automatic thread cleanup on app shutdown

## Status Indicators

The live feed page shows real-time status:
- 🔴 **Multi-Camera RTSP Monitoring Active** - All cameras running
- ⚪ **No Cameras Active** - No background processing
- 📊 **Camera Status** - Individual camera thread status

## Troubleshooting

- **Cameras not connecting**: Check RTSP URLs and network connectivity
- **High CPU usage**: Reduce concurrent cameras or optimize YOLO settings
- **Memory issues**: Monitor queue sizes and adjust maxsize if needed
- **Thread not stopping**: Use API endpoints or restart the application