import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import cv2
import torch
import math
import threading
import time
import numpy as np
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, Response, render_template_string


# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SKIP = 5 
IMG_WIDTH = 640
IMG_HEIGHT = 480
STOPPING_SIZE = 400

# CLIP Prompts
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
    "a photo of a person standing normally",
    "a photo of a person walking",
    "a photo of a person sitting and working"
]
DANGER_LABELS = PROMPTS[:8]  # First 8 are considered threats

# ---------------------------
# FLASK & GLOBALS
# ---------------------------
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

class SecurityPatrolBot(Node):

    def __init__(self):
        super().__init__('security_patrol_bot')
        
        # --- ROS SETUP ---
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # --- STATE MANAGEMENT ---
        self.state = "PATROL"  # Options: PATROL, INTERCEPT, RECOVERY
        self.target_box = None
        self.cooldown_timer = 0
        
        # --- WALL FOLLOWER PARAMETERS (From follow_wall.py) ---
        self.target_dist = 0.55
        self.wall_found_thresh = 1.0
        self.hallway_thresh = 1.5
        self.front_obs_dist = 0.55
        self.kp = 1.8 
        self.kd = 8.0 
        self.scan_history = {'front': [], 'right': [], 'f_right': []}
        self.lost_wall_timer = 0
        
        # --- AI MODEL LOADING ---
        self.get_logger().info(f"Loading Models on {DEVICE}...")
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()
        self.yolo = YOLO("yolov8n.pt")
        self.get_logger().info("Models Loaded. System Ready.")

        # --- VISION THREAD ---
        # Run vision in a separate thread so it doesn't block LiDAR callbacks
        self.vision_thread = threading.Thread(target=self.vision_loop)
        self.vision_thread.daemon = True
        self.vision_thread.start()

    # =========================================
    # PART 1: LIDAR & PATROL LOGIC
    # =========================================
    
    def get_smoothed_range(self, msg, angle_deg, key):
        """Helper to smooth LiDAR data"""
        angle_rad = math.radians(angle_deg)
        center_idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
        sector = 10 
        start = max(0, center_idx - sector)
        end = min(len(msg.ranges), center_idx + sector)
        slice_data = [r for r in msg.ranges[start:end] if 0.05 < r < 10.0]
        
        raw_val = min(slice_data) if slice_data else 10.0
        
        self.scan_history[key].append(raw_val)
        if len(self.scan_history[key]) > 3: 
            self.scan_history[key].pop(0)
        return sum(self.scan_history[key]) / len(self.scan_history[key])

    def lidar_callback(self, msg):
        """
        Runs at high frequency via ROS. 
        If State is PATROL: Drives the robot.
        If State is INTERCEPT: Yields control to Vision Loop.
        """
        # Always process sensors for safety (e.g. panic stop), but mainly for Wall Following
        front = self.get_smoothed_range(msg, 0, 'front')
        right = self.get_smoothed_range(msg, -90, 'right')
        f_right = self.get_smoothed_range(msg, -45, 'f_right')
        
        # EMERGENCY STOP override (active in all modes)
        if front < 0.2: 
            self.get_logger().warn("CRITICAL PROXIMITY - STOPPING")
            stop_cmd = Twist()
            self.publisher_.publish(stop_cmd)
            return

        # If we are busy intercepting a target, do not run wall follower logic
        if self.state == "INTERCEPT":
            return 

        # --- PATROL LOGIC (Wall Follower) ---
        cmd = Twist()
        
        # PID Calc
        error_p = self.target_dist - right
        error_d = (right - f_right)
        steer = (self.kp * error_p) + (self.kd * error_d)

        # 1. OBSTACLE AHEAD -> PIVOT
        if front < self.front_obs_dist:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.9 
            self.lost_wall_timer = 0
            
        # 2. WALL FOLLOWING
        elif right < self.wall_found_thresh:
            self.lost_wall_timer = 0
            cmd.linear.x = 0.25
            cmd.angular.z = max(min(steer, 1.5), -1.5)

        # 3. HALLWAY / OPEN SPACE
        elif right > self.hallway_thresh and front > 2.0:
            self.lost_wall_timer = 0
            cmd.linear.x = 0.35
            cmd.angular.z = 0.0

        # 4. CORNER RECOVERY
        else:
            self.lost_wall_timer += 1
            if self.lost_wall_timer < 15: 
                cmd.linear.x = 0.2
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.15
                cmd.angular.z = -0.6

        self.publisher_.publish(cmd)

    # =========================================
    # PART 2: VISION & INTERCEPT LOGIC
    # =========================================

    def vision_loop(self):
        global output_frame, lock
        cap = cv2.VideoCapture(0)
        cap.set(3, IMG_WIDTH)
        cap.set(4, IMG_HEIGHT)
        
        frame_count = 0
        center_x = IMG_WIDTH // 2

        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: continue
            
            # Decrease cooldown if active
            if self.cooldown_timer > 0:
                self.cooldown_timer -= 1

            # 1. DETECT PEOPLE
            results = self.yolo(frame, verbose=False, classes=[0], conf=0.4) # Class 0 = Person
            detections = results[0].boxes
            
            current_threat_box = None
            is_violent = False
            
            # 2. ANALYZE BEHAVIOR (CLIP)
            if len(detections) > 0 and self.cooldown_timer == 0:
                # Find largest person (assumed closest)
                sorted_boxes = sorted(detections, key=lambda x: x.xywh[0][2] * x.xywh[0][3], reverse=True)
                target = sorted_boxes[0]
                x1, y1, x2, y2 = target.xyxy[0].cpu().numpy()
                current_threat_box = (int(x1), int(y1), int(x2), int(y2))

                # Run CLIP periodically to check for violence
                if frame_count % FRAME_SKIP == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # We could crop the person here for better accuracy, but full frame gives context
                    inputs = self.processor(text=PROMPTS, images=rgb, return_tensors="pt", padding=True)
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1)
                        top_prob, top_idx = probs[0].topk(1)
                        label = PROMPTS[top_idx.item()]
                        
                        if label in DANGER_LABELS and top_prob.item() > 0.35:
                            is_violent = True
                            self.get_logger().warn(f"VIOLENCE DETECTED: {label}")
                        else:
                            is_violent = False

            # 3. STATE TRANSITIONS & CONTROL
            
            # TRIGGER INTERCEPT
            if is_violent and self.state == "PATROL":
                self.state = "INTERCEPT"
                self.get_logger().info(">>> SWITCHING TO INTERCEPT MODE <<<")

            # INTERCEPT BEHAVIOR
            if self.state == "INTERCEPT" and current_threat_box is not None:
                tx1, ty1, tx2, ty2 = current_threat_box
                target_center_x = (tx1 + tx2) / 2
                target_h = ty2 - ty1
                
                # Check Arrival
                if target_h > STOPPING_SIZE:
                    self.get_logger().info(">>> ARRIVED AT TARGET. STOPPING. <<<")
                    
                    # Stop Robot
                    stop_msg = Twist()
                    self.publisher_.publish(stop_msg)
                    
                    # Wait a moment (simulate action/recording) then return to patrol
                    time.sleep(3.0) 
                    self.state = "PATROL"
                    self.cooldown_timer = 100 # Ignore detections for ~3-4 seconds to turn away
                    self.get_logger().info(">>> RETURNING TO PATROL <<<")
                
                else:
                    # Drive towards target
                    twist = Twist()
                    # Steer
                    error_x = target_center_x - center_x
                    twist.angular.z = -0.004 * error_x 
                    # Drive
                    twist.linear.x = 0.4
                    self.publisher_.publish(twist)
                    
                # Draw Box
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
                cv2.putText(frame, "INTERCEPTING", (tx1, ty1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # PATROL VISUALS
            elif self.state == "PATROL":
                cv2.putText(frame, "PATROLLING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            frame_count += 1
            
            # Update global frame for Flask
            with lock:
                output_frame = frame.copy()
        
        cap.release()

# ---------------------------
# FLASK WEB SERVER
# ---------------------------

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template_string('''
        <html>
            <body style="background:black; color:white; text-align:center;">
                <h1>Security Bot V1</h1>
                <img src="/video_feed" style="border: 2px solid grey; width: 80%;">
            </body>
        </html>
    ''')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

def main(args=None):
    rclpy.init(args=args)
    node = SecurityPatrolBot()
    
    # Start Flask in a background thread
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()