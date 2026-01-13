#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import threading
from flask import Flask, Response, render_template_string, jsonify
import json
import time
import numpy as np

# --- Flask App Setup ---
app = Flask(__name__)

# Global variables to share data between ROS thread and Flask thread
latest_frame = None
latest_status = {"status": "Waiting for data..."}
latest_tracks = []
frame_lock = threading.Lock()
data_lock = threading.Lock()

def get_standby_frame():
    # Helper to create a "No Signal" frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Waiting for Camera Feed...", (100, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# HTML Template for the Web UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Limo Patrol Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #222; color: white; text-align: center; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; padding: 20px; }
        .video-box { border: 2px solid #555; margin: 10px; }
        .info-box { background-color: #333; padding: 20px; border-radius: 10px; margin: 10px; width: 300px; text-align: left;}
        h2 { color: #4CAF50; }
        .alert { color: red; font-weight: bold; font-size: 1.2em; }
        pre { background: #111; padding: 10px; overflow-x: auto; }
    </style>
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update Brain Status
                    let statusHtml = "<h3>🧠 Brain Status</h3>";
                    if (data.brain && data.brain.suspicious) {
                        statusHtml += `<p class='alert'>⚠️ DETECTED: ${data.brain.reason}</p>`;
                        statusHtml += `<p>Confidence: ${data.brain.confidence}</p>`;
                    } else if (data.brain) {
                        statusHtml += "<p>Status: Scanning...</p>";
                    } else {
                        statusHtml += "<p>No active alerts.</p>";
                    }
                    document.getElementById('brain-status').innerHTML = statusHtml;

                    // Update Tracking Info
                    let trackHtml = "<h3>👁️ Tracking (" + data.tracks.length + " people)</h3>";
                    trackHtml += "<pre>" + JSON.stringify(data.tracks, null, 2) + "</pre>";
                    document.getElementById('track-status').innerHTML = trackHtml;
                });
        }
        setInterval(updateStatus, 1000); // Poll every second
    </script>
</head>
<body>
    <h1>🤖 Limo Patrol Dashboard</h1>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" width="640" height="480" alt="Video Stream">
        </div>
        <div>
            <div id="brain-status" class="info-box"><h3>🧠 Brain Status</h3><p>Waiting...</p></div>
            <div id="track-status" class="info-box"><h3>👁️ Tracking</h3><p>Waiting...</p></div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = get_standby_frame()
            else:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1) # Limit to 10FPS to save bandwidth when idle

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    global latest_status, latest_tracks
    with data_lock:
        return jsonify({
            "brain": latest_status,
            "tracks": latest_tracks
        })

# --- ROS Node ---
class WebUINode(Node):
    def __init__(self):
        super().__init__('web_ui_node')
        self.bridge = CvBridge()
        
        # Subscribe to Images (Visual Debug)
        self.subscription_img = self.create_subscription(
            Image, '/debug_image', self.image_callback, 10)
            
        # Subscribe to Brain Alerts
        self.subscription_brain = self.create_subscription(
            String, '/pursuit_target', self.brain_callback, 10)
            
        # Subscribe to Tracking Data
        self.subscription_tracks = self.create_subscription(
            String, '/tracked_persons', self.tracker_callback, 10)
            
        self.get_logger().info("Web UI Node Started. Connect to http://<robot-ip>:5000")

    def image_callback(self, msg):
        global latest_frame
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with frame_lock:
                latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def brain_callback(self, msg):
        global latest_status
        try:
            data = json.loads(msg.data)
            with data_lock:
                latest_status = data
        except ValueError:
            pass

    def tracker_callback(self, msg):
        global latest_tracks
        try:
            data = json.loads(msg.data)
            with data_lock:
                latest_tracks = data
        except ValueError:
            pass

def run_flask():
    # Run Flask on all interfaces so we can access it from outside the robot
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main(args=None):
    rclpy.init(args=args)
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start ROS Node in main thread
    node = WebUINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
