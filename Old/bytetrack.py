import cv2
import time
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
# Use the ENGINE file we just created for max speed
MODEL_PATH = 'yolov8n.engine' 
TARGET_FPS = 20
CAMERA_INDEX = 0 
# ---------------------

print(f"Loading TensorRT Model: {MODEL_PATH}...")
# Task='detect' is vital when loading engine files directly
model = YOLO(MODEL_PATH, task='detect')

def generate_frames():
    # Attempt to use GStreamer for lower CPU usage on Jetson
    # If this fails, revert to cv2.VideoCapture(CAMERA_INDEX)
    gst_str = (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    # Try GStreamer first, fallback to USB
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    while True:
        loop_start = time.time()
        success, frame = cap.read()
        if not success:
            break

        # --- TRACKING & FILTERING ---
        # classes=[0] tells YOLO to ONLY look for people (Class 0)
        # tracker="bytetrack.yaml" is still the fastest tracker
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[0],  # FILTER: Only detect People
            conf=0.4,     # Filter out weak detections
            verbose=False
        )

        # Plot frame
        annotated_frame = results[0].plot()

        # Encode for Flask
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Limit FPS to keep Jetson cool
        process_time = time.time() - loop_start
        wait_time = (1.0 / TARGET_FPS) - process_time
        if wait_time > 0:
            time.sleep(wait_time)

@app.route('/')
def index():
    return render_template_string('''
        <body style="background:black; color:white; text-align:center;">
            <h1>Limo Pro - Person Tracker</h1>
            <img src="{{ url_for('video_feed') }}" style="border: 2px solid white; width:80%;">
        </body>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)