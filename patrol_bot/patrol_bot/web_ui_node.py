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
from collections import deque

try:
    from limo_msgs.msg import LimoStatus
except ImportError:
    LimoStatus = None

# --- Flask App Setup ---
app = Flask(__name__)

# Global variables to share data between ROS thread and Flask thread
latest_frame = None
latest_status = {"status": "Waiting for data..."}
latest_tracks = []
latest_battery = {
    "voltage": 0.0, 
    "percentage": 0.0, 
    "status": "Unknown",
    "discharge_rate": 0.0,
    "runtime_remaining": "Calculating..."
}
voltage_history = deque(maxlen=600)  # Store (timestamp, voltage)

current_fps = 0.0
last_fps_time = time.time()
frame_count = 0
last_brain_update_time = 0
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
        .safe { color: lightgreen; font-weight: bold; font-size: 1.2em; }
        pre { background: #111; padding: 10px; overflow-x: auto; }
    </style>
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update Brain Status
                    let statusHtml = "<h3>🧠 Brain Status</h3>";
                    if (data.brain) {
                        if (data.brain.suspicious === true) {
                            statusHtml += `<p class='alert'>⚠️ DETECTED: ${data.brain.reason}</p>`;
                            statusHtml += `<p>Confidence: ${data.brain.confidence}</p>`;
                        } else if (data.brain.suspicious === false) {
                            statusHtml += `<p class='safe'>✅ SAFE: ${data.brain.reason}</p>`;
                            statusHtml += `<p>Confidence: ${data.brain.confidence}</p>`;
                        } else if (data.brain.status) {
                            statusHtml += `<p>${data.brain.status}</p>`;
                        } else {
                            statusHtml += "<p>Status: Scanning...</p>";
                        }
                    } else {
                        statusHtml += "<p>No active alerts.</p>";
                    }
                    document.getElementById('brain-status').innerHTML = statusHtml;

                    // Update Metrics
                    let metricsHtml = `<h3>📊 Metrics</h3>`;
                    metricsHtml += `<p>Video FPS: ${data.fps}</p>`;
                    if (data.tracks && data.tracks.length > 0) {
                         let avgConf = data.tracks.reduce((acc, t) => acc + (t.confidence || 0), 0) / data.tracks.length;
                         metricsHtml += `<p>Avg Track Conf: ${avgConf.toFixed(2)}</p>`;
                    }
                    document.getElementById('metrics-status').innerHTML = metricsHtml;

                    // Update Battery - ADDED
                    let batteryHtml = `<h3>🔋 Battery</h3>`;
                    if (data.battery && data.battery.status !== "Unknown") {
                        let color = data.battery.percentage > 20 ? 'lightgreen' : 'red';
                        batteryHtml += `<p style='color:${color}; font-size: 1.5em;'>${data.battery.percentage}% (${data.battery.voltage}V)</p>`;
                        batteryHtml += `<p>Est. Runtime: <b>${data.battery.runtime_remaining}</b></p>`;
                        batteryHtml += `<p style='font-size:0.8em; color:#aaa'>Drop Rate: ${data.battery.discharge_rate} V/min</p>`;
                    } else {
                        batteryHtml += `<p>waiting for LIMO status...</p>`;
                    }
                    document.getElementById('battery-status').innerHTML = batteryHtml;

                    // Update Tracking Info
                    let trackHtml = "<h3>👁️ Tracking (" + (data.tracks ? data.tracks.length : 0) + " people)</h3>";
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
            <div id="metrics-status" class="info-box"><h3>📊 Metrics</h3><p>Waiting...</p></div>
            <div id="battery-status" class="info-box"><h3>🔋 Battery</h3><p>Waiting...</p></div>
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
    global latest_status, latest_tracks, current_fps, last_brain_update_time, latest_battery
    with data_lock:
        status_to_send = latest_status
        # If no update for 5 seconds, say "Offline"
        if time.time() - last_brain_update_time > 5.0 and last_brain_update_time > 0:
            status_to_send = {"status": "Brain Offline / No Data"}

        return jsonify({
            "brain": status_to_send,
            "tracks": latest_tracks,
            "battery": latest_battery,
            "fps": round(current_fps, 1)
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
            
        # Subscribe to Battery Status
        if LimoStatus:
            self.subscription_status = self.create_subscription(
                LimoStatus, '/limo_status', self.status_callback, 10)
            self.get_logger().info("Subscribed to /limo_status")
        else:
            self.get_logger().warn("LimoStatus message not found. Battery monitoring disabled.")
            
        self.get_logger().info("Web UI Node Started. Connect to http://<robot-ip>:5000")

    def image_callback(self, msg):
        global latest_frame, frame_count, last_fps_time, current_fps
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with frame_lock:
                latest_frame = cv_image
            
            # FPS Calculation
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                current_fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def brain_callback(self, msg):
        global latest_status, last_brain_update_time
        try:
            data = json.loads(msg.data)
            with data_lock:
                latest_status = data
                last_brain_update_time = time.time()
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

    def status_callback(self, msg):
        global latest_battery, voltage_history
        try:
            current_time = time.time()
            voltage = msg.battery_voltage
            
            # Simple Voltage -> Percentage (approximate for 3S LiPo)
            # 12.6V = 100%, 10.0V = 0%
            max_v = 12.6
            min_v = 10.0
            percentage = max(0, min(100, (voltage - min_v) / (max_v - min_v) * 100))
            
            # Store history
            with data_lock:
                voltage_history.append((current_time, voltage))
                
                # Calculate discharge rate and runtime
                discharge_rate = 0.0
                runtime_str = "Calculating..."
                
                # Check discharge rate every 30 seconds worth of data
                if len(voltage_history) > 100: 
                    # Calculate slope (voltage drop per minute)
                    # We compare average of first 20 vs last 20 samples to smooth out noise
                    start_avg = np.mean([v for t, v in list(voltage_history)[:20]])
                    end_avg = np.mean([v for t, v in list(voltage_history)[-20:]])
                    time_diff = voltage_history[-1][0] - voltage_history[0][0] # Seconds
                    
                    if time_diff > 10: 
                        drop = start_avg - end_avg
                        if drop > 0.01: # Significant drop
                            discharge_rate = drop / (time_diff / 60.0) # Volts per minute
                            minutes_left = (voltage - min_v) / discharge_rate
                            runtime_str = f"{int(minutes_left)} min"
                        else:
                            runtime_str = "Stable (> 100h)"

                latest_battery = {
                    "voltage": round(voltage, 2),
                    "percentage": int(percentage),
                    "status": "Online",
                    "discharge_rate": round(discharge_rate, 4),
                    "runtime_remaining": runtime_str
                }
        except Exception as e:
            self.get_logger().error(f"Error in battery callback: {e}")

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
