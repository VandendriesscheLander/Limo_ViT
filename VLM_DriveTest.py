import cv2
import torch
import threading
import time
from collections import deque
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import rclpy
from geometry_msgs.msg import Twist

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SKIP = 5 

# Control Config
img_width = 640
img_height = 480
center_x = img_width // 2
STOPPING_SIZE = 400 # If box height > this, stop (too close)

PROMPTS = [
    "a photo of a physical fight",
    "a photo of a person punching someone",
    "a photo of a person kicking someone",
    "a photo of a person stabbing someone",
    "a photo of a person shooting a gun",
    "a photo of a person wrestling with another person",
    "a photo of someone being attacked",
    "a photo of a person holding a weapon",
    "a photo of people shaking hands",
    "a photo of people hugging",
    "a photo of a person shredding paper",
    "a photo of a person reading a document",
    "a photo of a person typing on a laptop",
    "a photo of a person drinking from a cup",
    "a photo of a person waving their hand",
    "a photo of a person standing normally",
    "a photo of a person walking",
    "a photo of a person sitting and working"
]

DANGER_LABELS = PROMPTS[:8] # First 8 are danger

# ---------------------------
# Robot Control Interface
# ---------------------------
class RobotController:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('vision_interceptor')
        self.pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

    def drive(self, linear_vel, angular_vel):
        msg = Twist()
        msg.linear.x = float(linear_vel)
        msg.angular.z = float(angular_vel)
        self.pub.publish(msg)

    def stop(self):
        self.drive(0.0, 0.0)

# Initialize Controller
robot = RobotController()

# ---------------------------
# Load Models
# ---------------------------
print(f"Loading CLIP-ViT to {DEVICE}...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print("Loading YOLOv8n (Gatekeeper)...")
yolo = YOLO("yolov8n.pt")
print("System Ready.")

# ---------------------------
# Globals
# ---------------------------
output_lock = threading.Lock()
latest_status = "PATROL"
latest_action = "Scanning..."
latest_conf = 0.0
person_detected = False
target_box = None # (x1, y1, x2, y2)

app = Flask(__name__)

# ---------------------------
# Inference Logic (CLIP)
# ---------------------------
def process_frame_with_clip(frame_rgb):
    global latest_status, latest_action, latest_conf

    try:
        inputs = processor(text=PROMPTS, images=frame_rgb, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            top_prob, top_idx = probs[0].topk(1)
            top_label = PROMPTS[top_idx.item()]
            conf = top_prob.item()

        is_danger = top_label in DANGER_LABELS
        
        with output_lock:
            if is_danger:
                if conf > 0.35: # Slightly higher threshold for action
                    latest_status = "INTERCEPT"
                    latest_action = f"DANGER: {top_label.replace('a photo of ', '')}"
                else:
                    latest_status = "WARNING"
                    latest_action = f"Unsure: {top_label.replace('a photo of ', '')}"
            else:
                latest_status = "PATROL" # Revert to patrol if safe
                latest_action = f"Safe: {top_label.replace('a photo of ', '')}"
            latest_conf = conf

    except Exception as e:
        print(f"CLIP Error: {e}")

# ---------------------------
# Main Logic
# ---------------------------
def generate_frames():
    global person_detected, target_box
    cap = cv2.VideoCapture(0)
    cap.set(3, img_width)
    cap.set(4, img_height)
    
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success: break

        # 1. YOLO Detection (Find the target)
        results = yolo(frame, verbose=False, classes=[0], conf=0.4)
        
        detections = results[0].boxes
        person_detected = len(detections) > 0
        
        # Pick the largest person (usually the closest/most prominent)
        current_target = None
        if person_detected:
            # Sort by area (width * height) descending
            sorted_boxes = sorted(detections, key=lambda x: x.xywh[0][2] * x.xywh[0][3], reverse=True)
            current_target = sorted_boxes[0] # The biggest person
            
            # Extract box for drawing and logic
            x1, y1, x2, y2 = current_target.xyxy[0].cpu().numpy()
            target_box = (int(x1), int(y1), int(x2), int(y2))
        else:
            target_box = None

        annotated_frame = results[0].plot()

        # 2. Run CLIP (Check for Danger)
        if person_detected and frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            process_frame_with_clip(rgb)

        # 3. CONTROL LOOP: Chase vs Wall Follow
        with output_lock:
            status = latest_status
        
        if status == "INTERCEPT" and target_box is not None:
            # --- CHASE LOGIC ---
            tx1, ty1, tx2, ty2 = target_box
            target_center_x = (tx1 + tx2) / 2
            target_height = ty2 - ty1
            
            # A. Calculate Steering (PID - Proportional)
            error_x = target_center_x - center_x
            # Gain Kp: 0.005 is a guess, tune this! 
            # If error is -300 (far left), turn = -1.5 (turn left)
            angular_z = -0.004 * error_x 
            
            # B. Calculate Speed
            # If target is too close (height is big), stop moving forward
            if target_height < STOPPING_SIZE:
                linear_x = 0.4 # Drive forward
            else:
                linear_x = 0.0 # Stop, we are there (or engage other mechanism)
            
            # Send to robot
            robot.drive(linear_x, angular_z)
            
            # Visual Feedback
            cv2.putText(annotated_frame, "TARGET LOCKED", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.line(annotated_frame, (int(target_center_x), int((ty1+ty2)/2)), (center_x, img_height//2), (0,0,255), 2)
            
        elif status == "PATROL":
             # --- WALL FOLLOW LOGIC ---
             # Since you said you already have wall following code:
             # Call your wall follow function here, or resume that thread.
             # robot.do_wall_following()
             pass
        
        else:
            # Safety stop if unsure or nothing detected
            robot.stop()

        frame_count += 1

        # UI Overlay
        cv2.rectangle(annotated_frame, (0,0), (640, 80), (0,0,0), -1)
        color = (0,0,255) if status == "INTERCEPT" else (0,255,0)
        cv2.putText(annotated_frame, f"MODE: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, f"ACT: {latest_action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('<img src="/video_feed" width="640">')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)