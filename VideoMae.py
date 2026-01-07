import cv2
import torch
import numpy as np
import threading
import time
from collections import deque
from flask import Flask, Response, render_template_string
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, AutoConfig
from ultralytics import YOLO
import os

# ---------------------------
# Config
# ---------------------------
# Using the official Kinetics-400 model (knows 400 actions)
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
ONNX_MODEL_PATH = "videomae.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
FRAME_SKIP = 6
CONFIDENCE_THRESHOLD = 0.5  # Kinetics has 400 classes, so 50% is actually very high confidence

# Define what constitutes "Danger" for your robot
DANGER_KEYWORDS = [
    "fighting", "punching", "kicking", "shooting", "wrestling", 
    "slapping", "stabbing", "violence", "hitting"
]

# ---------------------------
# Load Models
# ---------------------------
print(f"Loading General Action VideoMAE...")
USE_ONNX = False
mae_model = None
mae_session = None

# Always load the processor for input preparation
print(f"Loading Processor from {MODEL_NAME}...")
mae_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
id2label = config.id2label

if os.path.exists(ONNX_MODEL_PATH):
    print(f"Found ONNX model at {ONNX_MODEL_PATH}. Using ONNX Runtime.")
    try:
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        mae_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        USE_ONNX = True
    except ImportError:
        print("onnxruntime not installed. Falling back to PyTorch.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}. Falling back to PyTorch.")

if not USE_ONNX:
    print(f"Loading PyTorch model to {DEVICE}...")
    mae_model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
    mae_model.to(DEVICE)
    mae_model.eval()

print("Loading YOLOv8n (Person Detector)...")
yolo_model = YOLO("yolov8n.pt") 
print("All models loaded.")

# ---------------------------
# Global State
# ---------------------------
output_lock = threading.Lock()
frame_buffer = deque(maxlen=NUM_FRAMES)
latest_status = "Scanning..."
latest_action = "N/A"
latest_confidence = 0.0
person_detected = False
robot_mode = "PATROL" 

app = Flask(__name__)

# ---------------------------
# Helper: Safety Check
# ---------------------------
def check_safety(action_label):
    """
    Returns (Is_Dangerous, Status_Message)
    """
    label_lower = action_label.lower()
    
    # 1. Check for specific danger keywords
    for keyword in DANGER_KEYWORDS:
        if keyword in label_lower:
            return True, f"DANGER: {action_label.upper()}"
            
    # 2. Otherwise, it is assumed safe/neutral
    return False, f"Safe: {action_label}"

# ---------------------------
# AI Inference Thread
# ---------------------------
def inference_loop():
    global latest_status, latest_action, latest_confidence, robot_mode
    
    while True:
        # 1. GATEKEEPER: If no person, don't run VideoMAE (saves huge battery/heat)
        if not person_detected:
            with output_lock:
                latest_status = "SAFE (Area Clear)"
                latest_action = "No Human"
                latest_confidence = 1.0
                robot_mode = "PATROL"
            time.sleep(0.2)
            continue

        # 2. ANALYZER: If buffer full, run Action Recognition
        if len(frame_buffer) == NUM_FRAMES:
            clip_frames = list(frame_buffer)
            
            try:
                inputs = mae_processor(clip_frames, return_tensors="pt")
                
                if USE_ONNX:
                    # ONNX Runtime expects numpy arrays
                    ort_inputs = {mae_session.get_inputs()[0].name: inputs['pixel_values'].numpy()}
                    ort_outs = mae_session.run(None, ort_inputs)
                    logits = ort_outs[0]
                    # Convert to torch for consistent processing
                    probs = torch.tensor(logits).softmax(dim=-1)
                else:
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = mae_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                    
                # Get the top prediction from 400 classes
                score, idx = torch.max(probs, dim=-1)
                top_score = score.item()
                top_label = id2label[idx.item()]

                # 3. LOGIC: Is the action dangerous?
                is_dangerous, status_msg = check_safety(top_label)
                
                with output_lock:
                    if is_dangerous and top_score > CONFIDENCE_THRESHOLD:
                        latest_status = "UNSAFE (DETECTED)"
                        latest_action = top_label
                        latest_confidence = top_score
                        robot_mode = "INTERCEPT"
                    else:
                        # Even if confidence is low, if it's not a danger word, it's safe
                        latest_status = "SAFE (Monitoring)"
                        latest_action = top_label
                        latest_confidence = top_score
                        robot_mode = "OBSERVE"

            except Exception as e:
                print(f"Error: {e}")
        
        time.sleep(0.2)

# ---------------------------
# Camera & YOLO Loop
# ---------------------------
def generate_frames():
    global person_detected
    
    cap = cv2.VideoCapture(0)
    # Lower resolution for speed
    cap.set(3, 640)
    cap.set(4, 480)
    
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. YOLO Person Check (Runs fast on GPU)
        results = yolo_model(frame, verbose=False, classes=[0], conf=0.4)
        current_person_count = len(results[0].boxes)
        person_detected = current_person_count > 0
        
        # 2. Visuals
        annotated_frame = results[0].plot()

        # 3. Buffer Update (Stride for temporal context)
        if frame_counter % FRAME_SKIP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb_frame)
        
        frame_counter += 1

        # 4. Overlay Info
        with output_lock:
            status = latest_status
            action = latest_action
            conf = latest_confidence
            mode = robot_mode

        # Color Coding
        if mode == "INTERCEPT":
            color = (0, 0, 255) # Red
        elif mode == "OBSERVE":
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 255, 0) # Green

        # UI Overlay
        cv2.rectangle(annotated_frame, (0,0), (640, 100), (0,0,0), -1)
        cv2.putText(annotated_frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_frame, f"ACTION: {action} ({conf:.1%})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"MODE: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html><body style="background:black; color:white; text-align:center;">
        <h1>Limo Pro 2 - Action Recognition</h1>
        <img src="{{ url_for('video_feed') }}" width="640" style="border: 2px solid grey;">
        </body></html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)