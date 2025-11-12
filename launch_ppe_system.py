#!/usr/bin/env python3

"""
Combined launcher for PPE Detection system
Launches both Flask web interface and OpenCV direct viewer
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

class PPELauncher:
    def __init__(self):
        self.flask_process = None
        self.opencv_process = None
        self.running = True
        
        # Get the directory where this script is located
        self.script_dir = Path(__file__).parent.absolute()
        
    def start_flask_app(self):
        """Start the Flask web application"""
        try:
            print("🌐 Starting Flask web application...")
            self.flask_process = subprocess.Popen(
                [sys.executable, 'app.py'],
                cwd=self.script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✅ Flask app started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start Flask app: {e}")
            return False
    
    def start_opencv_viewer(self, delay=3):
        """Start the OpenCV camera viewer with optional delay"""
        def delayed_start():
            if delay > 0:
                print(f"⏳ Waiting {delay} seconds before starting OpenCV viewer...")
                time.sleep(delay)
            
            try:
                print("🖥️ Starting OpenCV Camera Viewer...")
                self.opencv_process = subprocess.Popen(
                    [sys.executable, 'camera_grid_viewer.py'],
                    cwd=self.script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("✅ OpenCV viewer started successfully")
            except Exception as e:
                print(f"❌ Failed to start OpenCV viewer: {e}")
        
        # Start in a separate thread to avoid blocking
        threading.Thread(target=delayed_start, daemon=True).start()
    
    def stop_processes(self):
        """Stop all running processes"""
        print("🛑 Stopping all processes...")
        
        if self.opencv_process:
            try:
                self.opencv_process.terminate()
                self.opencv_process.wait(timeout=5)
                print("✅ OpenCV viewer stopped")
            except subprocess.TimeoutExpired:
                self.opencv_process.kill()
                print("⚠️ OpenCV viewer force killed")
            except Exception as e:
                print(f"❌ Error stopping OpenCV viewer: {e}")
        
        if self.flask_process:
            try:
                self.flask_process.terminate()
                self.flask_process.wait(timeout=5)
                print("✅ Flask app stopped")
            except subprocess.TimeoutExpired:
                self.flask_process.kill()
                print("⚠️ Flask app force killed")
            except Exception as e:
                print(f"❌ Error stopping Flask app: {e}")
    
    def run(self, launch_opencv=True, opencv_delay=3):
        """Run the complete PPE detection system"""
        print("🚀 PPE Detection System Launcher")
        print("=" * 50)
        
        try:
            # Start Flask app
            if not self.start_flask_app():
                return
            
            # Start OpenCV viewer if requested
            if launch_opencv:
                self.start_opencv_viewer(delay=opencv_delay)
            
            print("\n📋 System Status:")
            print("   🌐 Flask Web Interface: http://localhost:5000")
            if launch_opencv:
                print("   🖥️ OpenCV Grid Viewer: Starting...")
            print("\n🎯 Available Features:")
            print("   • Real-time PPE violation detection")
            print("   • Multi-camera RTSP monitoring")
            print("   • Violation logging and reporting")
            print("   • Web-based dashboard")
            if launch_opencv:
                print("   • Grid layout camera viewer")
            print("\n⌨️ Controls:")
            print("   • Ctrl+C: Stop all processes")
            if launch_opencv:
                print("   • In OpenCV window: 'q' to quit, 'r' to restart cameras")
            
            # Keep running and monitor processes
            while self.running:
                time.sleep(1)
                
                # Check if Flask process is still running
                if self.flask_process and self.flask_process.poll() is not None:
                    print("⚠️ Flask process stopped unexpectedly")
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.stop_processes()
            print("👋 PPE Detection System stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupt received, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPE Detection System Launcher')
    parser.add_argument('--no-opencv', action='store_true', 
                       help='Launch only Flask web interface (no OpenCV viewer)')
    parser.add_argument('--opencv-delay', type=int, default=3,
                       help='Delay in seconds before starting OpenCV viewer (default: 3)')
    parser.add_argument('--flask-only', action='store_true',
                       help='Launch only Flask app (same as --no-opencv)')
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Determine launch options
    launch_opencv = not (args.no_opencv or args.flask_only)
    
    # Create and run launcher
    launcher = PPELauncher()
    launcher.run(launch_opencv=launch_opencv, opencv_delay=args.opencv_delay)

if __name__ == "__main__":
    main()