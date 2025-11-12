#!/usr/bin/env python3
"""
Standalone launcher for multi-window PPE detection camera viewer
This script launches only the multi-window OpenCV viewer without Flask
"""

import sys
import os
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

# Import and run the multi-window viewer
from multi_window_camera_viewer import MultiWindowCameraViewer
import signal

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

def main():
    """Main entry point"""
    # RTSP URLs - same as in your Flask app
    RTSP_URLS = [
        "rtsp://admin:India123%23@10.45.1.63:5545/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:India123%23@10.45.1.64:5543/cam/realmonitor?channel=1&subtype=0"
    ]
    
    print("🚀 PPE Detection - Multi-Window Camera Viewer")
    print("=" * 50)
    print(f"📹 Will open {len(RTSP_URLS)} separate camera windows")
    print("🎮 Global Controls:")
    print("   • ESC: Quit all cameras")
    print("   • SPACE: Restart all cameras") 
    print("   • 1-9: Restart specific camera")
    print("   • Individual windows: 'q' to close, 'r' to restart")
    print("=" * 50)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        viewer = MultiWindowCameraViewer(RTSP_URLS)
        viewer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        import cv2
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()