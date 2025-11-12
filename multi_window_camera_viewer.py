import cv2
import threading
import time
import numpy as np
from YOLO_Video import video_detection
import signal
import sys

class MultiWindowCameraViewer:
    def __init__(self, rtsp_urls):
        self.rtsp_urls = rtsp_urls
        self.camera_threads = {}
        self.stop_flags = {}
        self.running = True
        
        # Window settings for individual cameras
        self.frame_width = 800
        self.frame_height = 600
        
        # Window positioning
        self.window_offset_x = 50
        self.window_offset_y = 50
        self.window_spacing_x = 850  # Space between windows horizontally
        self.window_spacing_y = 650  # Space between windows vertically
        
    def create_placeholder_frame(self, text, camera_id):
        """Create a placeholder frame with text"""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add gradient background for better visibility
        for i in range(self.frame_height):
            intensity = int(30 + (i / self.frame_height) * 20)
            frame[i, :] = [intensity, intensity, intensity]
        
        # Add main text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (255, 255, 255)
        thickness = 2
        
        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add camera ID in corner
        id_text = f"Camera {camera_id}"
        cv2.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)
        cv2.putText(frame, id_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def camera_thread(self, camera_id, rtsp_url, window_name):
        """Thread function to process individual camera stream"""
        print(f"🎥 Starting camera thread for {window_name}: {rtsp_url}")
        
        # Create window for this camera
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.frame_width, self.frame_height)
        
        # Position window based on camera ID
        cameras_per_row = 2  # Adjust based on your screen size
        row = (camera_id - 1) // cameras_per_row
        col = (camera_id - 1) % cameras_per_row
        
        pos_x = self.window_offset_x + (col * self.window_spacing_x)
        pos_y = self.window_offset_y + (row * self.window_spacing_y)
        cv2.moveWindow(window_name, pos_x, pos_y)
        
        try:
            # Show initial connecting message
            connecting_frame = self.create_placeholder_frame("Connecting...", camera_id)
            cv2.imshow(window_name, connecting_frame)
            cv2.waitKey(1)
            
            frame_generator = video_detection(rtsp_url)
            
            for frame in frame_generator:
                if self.stop_flags.get(camera_id, False):
                    break
                
                # Resize frame to fit window
                resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Add camera info overlay
                overlay_height = 60
                overlay = np.zeros((overlay_height, self.frame_width, 3), dtype=np.uint8)
                
                # Semi-transparent background
                cv2.rectangle(overlay, (0, 0), (self.frame_width, overlay_height), (0, 0, 0), -1)
                
                # Camera info text
                info_text = f"Camera {camera_id} - Live Feed"
                cv2.putText(overlay, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(overlay, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Controls info
                controls_text = "Press 'q' to close | 'r' to restart"
                text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(overlay, controls_text, (self.frame_width - text_size[0] - 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Blend overlay with frame
                alpha = 0.7
                resized_frame[0:overlay_height] = cv2.addWeighted(
                    resized_frame[0:overlay_height], alpha, overlay, 1 - alpha, 0
                )
                
                cv2.imshow(window_name, resized_frame)
                
                # Handle key presses for this window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"🛑 Quit requested for {window_name}")
                    self.stop_flags[camera_id] = True
                    break
                elif key == ord('r'):
                    print(f"🔄 Restart requested for {window_name}")
                    # This will break the loop and the thread will restart
                    break
                    
        except Exception as e:
            print(f"❌ Error in {window_name}: {e}")
            # Show error message
            error_frame = self.create_placeholder_frame(f"Error: {str(e)[:50]}...", camera_id)
            cv2.imshow(window_name, error_frame)
            cv2.waitKey(2000)  # Show error for 2 seconds
        
        finally:
            print(f"🏁 Camera thread {camera_id} ended")
            cv2.destroyWindow(window_name)
    
    def start_camera(self, camera_id, rtsp_url):
        """Start a single camera thread"""
        window_name = f"PPE Detection - Camera {camera_id}"
        
        # Stop existing thread if running
        if camera_id in self.stop_flags:
            self.stop_flags[camera_id] = True
            
        if camera_id in self.camera_threads:
            thread = self.camera_threads[camera_id]
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Start new thread
        self.stop_flags[camera_id] = False
        thread = threading.Thread(
            target=self.camera_thread,
            args=(camera_id, rtsp_url, window_name),
            daemon=True
        )
        self.camera_threads[camera_id] = thread
        thread.start()
        print(f"🚀 Started window for Camera {camera_id}")
    
    def start_all_cameras(self):
        """Start all camera threads"""
        print("🎬 Starting all camera windows...")
        for i, rtsp_url in enumerate(self.rtsp_urls):
            camera_id = i + 1
            self.start_camera(camera_id, rtsp_url)
            time.sleep(0.5)  # Small delay between camera starts
    
    def stop_all_cameras(self):
        """Stop all camera threads"""
        print("🛑 Stopping all camera threads...")
        
        # Signal all threads to stop
        for camera_id in self.stop_flags:
            self.stop_flags[camera_id] = True
        
        # Wait for threads to finish
        for camera_id, thread in self.camera_threads.items():
            if thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    print(f"⚠️ Thread for Camera {camera_id} did not stop gracefully")
        
        # Destroy all windows
        cv2.destroyAllWindows()
        
        # Clear dictionaries
        self.camera_threads.clear()
        self.stop_flags.clear()
        print("✅ All camera threads stopped")
    
    def monitor_windows(self):
        """Monitor for global key presses and window events"""
        print("\n🎮 Global Controls:")
        print("   • ESC: Quit all cameras")
        print("   • SPACE: Restart all cameras")
        print("   • 1-9: Restart specific camera")
        print("   • Individual window controls: 'q' to close, 'r' to restart")
        
        try:
            while self.running:
                # Create a small control window for global commands
                control_frame = np.zeros((150, 400, 3), dtype=np.uint8)
                
                # Add instructions
                cv2.putText(control_frame, "PPE Detection Control", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(control_frame, "ESC: Quit All", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(control_frame, "SPACE: Restart All", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(control_frame, "1-9: Restart Camera N", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Show active cameras count
                active_count = sum(1 for t in self.camera_threads.values() if t.is_alive())
                cv2.putText(control_frame, f"Active Cameras: {active_count}/{len(self.rtsp_urls)}", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("PPE Control Panel", control_frame)
                
                key = cv2.waitKey(100) & 0xFF
                
                if key == 27:  # ESC key
                    print("🛑 ESC pressed - Quitting all cameras")
                    break
                elif key == 32:  # SPACE key
                    print("🔄 SPACE pressed - Restarting all cameras")
                    self.stop_all_cameras()
                    time.sleep(1)
                    self.start_all_cameras()
                elif key >= ord('1') and key <= ord('9'):  # Number keys 1-9
                    camera_num = key - ord('0')
                    if camera_num <= len(self.rtsp_urls):
                        print(f"🔄 Restarting Camera {camera_num}")
                        rtsp_url = self.rtsp_urls[camera_num - 1]
                        self.start_camera(camera_num, rtsp_url)
                
                # Check if any camera threads died unexpectedly
                for camera_id, thread in list(self.camera_threads.items()):
                    if not thread.is_alive() and not self.stop_flags.get(camera_id, True):
                        print(f"⚠️ Camera {camera_id} thread died, restarting...")
                        rtsp_url = self.rtsp_urls[camera_id - 1]
                        time.sleep(1)
                        self.start_camera(camera_id, rtsp_url)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupt received")
        finally:
            self.running = False
    
    def run(self):
        """Main run method"""
        print("🚀 Multi-Window PPE Detection Camera Viewer")
        print("=" * 60)
        print(f"📹 Opening {len(self.rtsp_urls)} camera windows")
        
        try:
            # Start all cameras
            self.start_all_cameras()
            
            # Wait a moment for windows to appear
            time.sleep(2)
            
            # Monitor for global controls
            self.monitor_windows()
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.stop_all_cameras()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

if __name__ == "__main__":
    # RTSP URLs - same as in your Flask app
    RTSP_URLS = [
        "rtsp://admin:India123%23@10.45.1.63:5545/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:India123%23@10.45.1.64:5543/cam/realmonitor?channel=1&subtype=0"
    ]
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        viewer = MultiWindowCameraViewer(RTSP_URLS)
        viewer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()