import cv2
import torch
import threading
import time
import numpy as np
from collections import deque
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# --- FIX IMPORTS ---
try:
    from transformers import XClipProcessor, XClipModel
except ImportError:
    print("Standard import failed. Trying direct import...")
    # This forces the import and will show exactly which library is missing
    from transformers.models.x_clip.processing_x_clip import XClipProcessor
    from transformers.models.x_clip.modeling_x_clip import XClipModel

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "microsoft/xclip-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8 
FRAME_SKIP = 5

# Define Safe vs Unsafe
TEXT_LABELS = [
    "fighting", 
    "punching", 
    "shooting a gun", 
    "stabbing", 
    "shredding paper",  # Safe class
    "holding a document", 
    "standing", 
    "walking"
]
DANGER_LABELS = ["fighting", "punching", "shooting a gun", "stabbing"]

# ---------------------------
# Load Models
# ---------------------------
print(f"Loading X-CLIP to {DEVICE}...")
processor = XClipProcessor.from_pretrained(MODEL_NAME)
model = XClipModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print("Loading YOLOv8n...")
yolo = YOLO("yolov8n.pt")

# ---------------------------
# Global State
# ---------------------------
output_lock = threading.Lock()
frame_buffer = deque(maxlen=NUM_FRAMES)
latest_status = "PATROL"
latest_action = "Scanning..."
latest_prob = 0.0
person_detected = False

app = Flask(__name__)

# ---------------------------
# Inference Loop
# ---------------------------
def inference_loop():
    global latest_status, latest_action, latest_prob
    
    while True:
        if not person_detected:
            with output_lock:
                latest_status = "PATROL"
                latest_action = "Area Clear"
                latest_prob = 0.0
            time.sleep(0.5)
            continue

        if len(frame_buffer) == NUM_FRAMES:
            # Create a clean list of frames
            # X-CLIP expects a list of numpy arrays (H, W, C)
            clip = list(frame_buffer)
            
            try:
                # X-CLIP Processor handles the resizing and normalization
                inputs = processor(
                    text=TEXT_LABELS, 
                    videos=clip, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits_per_video[0]
                    probs = torch.softmax(logits, dim=-1)

                    top_prob, top_idx = probs.max(dim=-1)
                    top_label = TEXT_LABELS[top_idx.item()]
                    top_prob_val = top_prob.item()

                with output_lock:
                    # 70% Confidence Threshold
                    if top_label in DANGER_LABELS and top_prob_val > 0.7:
                        latest_status = "INTERCEPT"
                        latest_action = f"DANGER: {top_label.upper()}"
                    else:
                        latest_status = "OBSERVE"
                        latest_action = f"Safe: {top_label}"
                    latest_prob = top_prob_val

            except Exception as e:
                print(f"Inference Error: {e}")
                frame_buffer.clear()

        time.sleep(0.25) # Cool down logic

# ---------------------------
# Camera Loop
# ---------------------------
def generate_frames():
    global person_detected
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. YOLO
        results = yolo(frame, verbose=False, classes=[0], conf=0.4)
        person_detected = len(results[0].boxes) > 0
        annotated_frame = results[0].plot()

        # 2. Buffer Management
        if frame_count % FRAME_SKIP == 0:
            # Convert BGR to RGB for transformers
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(list(rgb))

        frame_count += 1

        # 3. Draw Overlay
        with output_lock:
            status = latest_status
            action = latest_action
            prob = latest_prob

        color = (0,0,255) if status == "INTERCEPT" else (0,255,0)
        cv2.rectangle(annotated_frame, (0,0), (640, 80), (0,0,0), -1)
        cv2.putText(annotated_frame, f"{status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, f"{action} ({prob:.1%})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('<img src="/video_feed" width="640">')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)