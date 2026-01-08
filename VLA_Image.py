import cv2
import torch
import threading
import time
from flask import Flask, Response, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import rclpy
from geometry_msgs.msg import Twist

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "vikhyatk/moondream2" 
REVISION = "2024-08-26"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
img_width = 640
img_height = 480
FRAME_SKIP = 5 

# ---------------------------
# Robot Control Interface
# ---------------------------
class RobotController:
    def __init__(self):
        # Initialize ROS2 Node
        rclpy.init()
        self.node = rclpy.create_node('vla_driver')
        self.pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.lock = threading.Lock()
        
    def drive(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def stop(self):
        self.drive(0.0, 0.0)

# Initialize Robot
robot = RobotController()

# ---------------------------
# Load VLA Model
# ---------------------------
print(f"Loading VLA ({MODEL_ID}) to {DEVICE}...")
# trust_remote_code=True is required for Moondream
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    revision=REVISION
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print("System Ready.")

# ---------------------------
# Globals
# ---------------------------
output_lock = threading.Lock()
latest_status = "WAITING"
latest_action = "STOP"

app = Flask(__name__)

# ---------------------------
# VLA Logic
# ---------------------------
def process_frame(frame_bgr):
    """
    Asks the VLA what to do based on the image.
    """
    global latest_status, latest_action
    
    # 1. Prepare Image
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    
    # 2. The "VLA Prompt"
    # We force the LLM to act as a controller
    prompt = (
        "You are a robot controller. Look at the image. "
        "If the path is clear, say 'FORWARD'. "
        "If there is a person or obstacle too close, say 'STOP'. "
        "If you need to turn to avoid something, say 'LEFT' or 'RIGHT'. "
        "Only output the single word action."
    )

    try:
        # 3. Inference
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        
        # 4. Parse Answer
        action = answer.strip().upper()
        
        # Safety fallback if model hallucinates
        valid_actions = ["FORWARD", "STOP", "LEFT", "RIGHT"]
        if not any(x in action for x in valid_actions):
            action = "STOP" # Fail safe
            
        with output_lock:
            latest_status = "ACTIVE"
            latest_action = action
            
        # 5. Execute Action directly
        if "FORWARD" in action:
            robot.drive(0.2, 0.0)
        elif "LEFT" in action:
            robot.drive(0.0, 0.5)
        elif "RIGHT" in action:
            robot.drive(0.0, -0.5)
        else:
            robot.stop()

    except Exception as e:
        print(f"Inference Error: {e}")
        robot.stop()

# ---------------------------
# Main Loop
# ---------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, img_width)
    cap.set(4, img_height)
    
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        # Run VLA Inference every N frames
        if frame_count % FRAME_SKIP == 0:
            # Run in background to keep video smooth? 
            # For simplicity in this single script, we run inline but it might lag slightly.
            # On Orin Nano, this inference takes ~0.3-0.5s
            process_frame(frame)

        # UI Overlay
        with output_lock:
            act = latest_action
            
        cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
        color = (0, 255, 0) if act == "FORWARD" else (0, 0, 255)
        cv2.putText(frame, f"ACTION: {act}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        frame_count += 1

@app.route('/')
def index():
    return render_template_string('<img src="/video_feed" width="640">')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        robot.node.destroy_node()
        rclpy.shutdown()