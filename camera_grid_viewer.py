import cv2
import threading
import time
import numpy as np
from YOLO_Video import video_detection
import signal
import sys

class CameraGridViewer:
    def __init__(self, rtsp_urls):
        self.rtsp_urls = rtsp_urls
        self.camera_frames = {}
        self.camera_threads = {}
        self.stop_flags = {}
        self.running = True
        
        # Window settings
        self.window_name = "PPE Detection - Live Camera Grid"
        self.grid_size = self.calculate_grid_size(len(rtsp_urls))
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize frames with black images
        for i in range(len(rtsp_urls)):
            camera_id = i + 1
            self.camera_frames[camera_id] = self.create_placeholder_frame(f"Camera {camera_id} - Connecting...")
    
    def calculate_grid_size(self, num_cameras):
        """Calculate optimal grid layout for cameras"""
        if num_cameras <= 1:
            return (1, 1)
        elif num_cameras <= 2:
            return (1, 2)
        elif num_cameras <= 4:
            return (2, 2)
        elif num_cameras <= 6:
            return (2, 3)
        elif num_cameras <= 9:
            return (3, 3)
        else:
            return (4, 4)  # Maximum 16 cameras
    
    def create_placeholder_frame(self, text):
        """Create a placeholder frame with text"""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)
        thickness = 2
        
        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        return frame
    
    def camera_thread(self, camera_id, rtsp_url):
        """Thread function to process individual camera stream"""
        print(f"🎥 Starting camera thread for Camera {camera_id}: {rtsp_url}")
        
        try:
            frame_generator = video_detection(rtsp_url)
            
            for frame in frame_generator:
                if self.stop_flags.get(camera_id, False):
                    break
                
                # Resize frame to fit grid
                resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Add camera label
                cv2.rectangle(resized_frame, (0, 0), (200, 30), (0, 0, 0), -1)
                cv2.putText(resized_frame, f"Camera {camera_id}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                self.camera_frames[camera_id] = resized_frame
                
        except Exception as e:
            print(f"❌ Error in camera {camera_id}: {e}")
            error_text = f"Camera {camera_id} - Error"
            self.camera_frames[camera_id] = self.create_placeholder_frame(error_text)
        
        print(f"🏁 Camera thread {camera_id} ended")
    
    def start_cameras(self):
        """Start all camera threads"""
        for i, rtsp_url in enumerate(self.rtsp_urls):
            camera_id = i + 1
            self.stop_flags[camera_id] = False
            
            thread = threading.Thread(
                target=self.camera_thread,
                args=(camera_id, rtsp_url),
                daemon=True
            )
            self.camera_threads[camera_id] = thread
            thread.start()
    
    def stop_cameras(self):
        """Stop all camera threads"""
        print("🛑 Stopping all camera threads...")
        
        # Signal all threads to stop
        for camera_id in self.stop_flags:
            self.stop_flags[camera_id] = True
        
        # Wait for threads to finish
        for camera_id, thread in self.camera_threads.items():
            thread.join(timeout=2.0)
        
        self.camera_threads.clear()
        print("✅ All camera threads stopped")
    
    def create_grid_display(self):
        """Create a grid display of all camera feeds"""
        rows, cols = self.grid_size
        
        # Create empty grid
        grid_height = rows * self.frame_height
        grid_width = cols * self.frame_width
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with camera frames
        for i in range(min(len(self.rtsp_urls), rows * cols)):
            camera_id = i + 1
            row = i // cols
            col = i % cols
            
            start_y = row * self.frame_height
            end_y = start_y + self.frame_height
            start_x = col * self.frame_width
            end_x = start_x + self.frame_width
            
            if camera_id in self.camera_frames:
                grid_image[start_y:end_y, start_x:end_x] = self.camera_frames[camera_id]
        
        return grid_image
    
    def run(self):
        """Main display loop"""
        print("🚀 Starting Camera Grid Viewer")
        print(f"📹 Monitoring {len(self.rtsp_urls)} cameras")
        print("Press 'q' to quit, 'r' to restart cameras")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size
        rows, cols = self.grid_size
        window_width = cols * self.frame_width
        window_height = rows * self.frame_height
        cv2.resizeWindow(self.window_name, window_width, window_height)
        
        # Start cameras
        self.start_cameras()
        
        try:
            while self.running:
                # Create and display grid
                grid_display = self.create_grid_display()
                
                # Add instructions
                instructions = "Press 'q' to quit | 'r' to restart cameras"
                cv2.putText(grid_display, instructions, (10, grid_display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(self.window_name, grid_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('r'):
                    print("🔄 Restarting cameras...")
                    self.stop_cameras()
                    time.sleep(1)
                    self.start_cameras()
                
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.stop_cameras()
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
        viewer = CameraGridViewer(RTSP_URLS)
        viewer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()