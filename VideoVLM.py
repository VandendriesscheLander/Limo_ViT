import cv2
import torch
import threading
import time
import logging
import numpy as np
from collections import deque
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from transformers import XCLIPProcessor, XCLIPModel

# ---------------------------
# Config
# ---------------------------
# Suppress the "padding" warning
logging.getLogger("transformers").setLevel(logging.ERROR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/xclip-base-patch32"

# VLM Params
SEQUENCE_LENGTH = 8      
SAMPLE_INTERVAL = 3      # Capture every 3rd frame (covers ~1 second of action)
CONFIDENCE_THRESHOLD = 0.55

# PROMPT ENGINEERING: Crucial change -> "a video of..."
PROMPTS = [
    # --- DANGER ---
    "a video of someone punching another person",
    "a video of someone kicking another person",
    "a video of people fighting",
    "a video of someone shooting a gun",
    "a video of violent wrestling",
    
    # --- SAFE ---
    "a video of people shaking hands",
    "a video of someone waving hello",
    "a video of someone walking normally",
    "a video of someone sitting down",
    "a video of someone standing still",
    "a video of someone typing on a keyboard"
]

DANGER_LABELS = [
    "a video of someone punching another person",
    "a video of someone kicking another person",
    "a video of people fighting",
    "a video of someone shooting a gun",
    "a video of violent wrestling"
]

# ---------------------------
# Global State
# ---------------------------
class GlobalState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=SEQUENCE_LENGTH) 
        self.status = "INITIALIZING"
        self.action = "Loading..."
        self.conf = 0.0
        self.person_detected = False
        self.running = True

state = GlobalState()

# ---------------------------
# 1. Camera Thread
# ---------------------------
def camera_worker():
    """Reads camera without blocking Flask or AI"""
    print("Starting Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    frame_counter = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap.open(0)
            continue

        with state.lock:
            state.frame = frame.copy()
            
            # Smart Buffer Filling
            if state.person_detected:
                if frame_counter % SAMPLE_INTERVAL == 0:
                    # Convert to RGB immediately for the AI
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    state.buffer.append(rgb)
            else:
                # Clear buffer if person leaves
                if len(state.buffer) > 0 and frame_counter % 10 == 0:
                    state.buffer.popleft()

        frame_counter += 1
        time.sleep(0.01)
    cap.release()

# ---------------------------
# 2. AI Worker (Corrected)
# ---------------------------
def ai_worker():
    print(f"Loading X-CLIP to {DEVICE}...")
    
    yolo = YOLO("yolov8n.pt")
    
    processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
    model = XCLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    print("AI Ready.")
    state.status = "PATROL"

    while state.running:
        # 1. Grab Data
        current_frame = None
        buffer_snapshot = []
        
        with state.lock:
            if state.frame is not None:
                current_frame = state.frame.copy()
            # Snapshot the buffer if it is full
            if len(state.buffer) == SEQUENCE_LENGTH:
                buffer_snapshot = list(state.buffer)

        if current_frame is None:
            time.sleep(0.1)
            continue

        # 2. YOLO Gatekeeper (Fast)
        results = yolo(current_frame, verbose=False, classes=[0], conf=0.4)
        has_person = len(results[0].boxes) > 0
        
        with state.lock:
            state.person_detected = has_person
            if not has_person:
                state.status = "PATROL"
                state.action = "Scanning..."
                state.conf = 0.0

        # 3. X-CLIP Inference (Slow/Accurate)
        # ONLY run if we have a person AND a full buffer
        if has_person and len(buffer_snapshot) == SEQUENCE_LENGTH:
            try:
                # CRITICAL FIX: Wrap buffer in another list!
                # Structure: [ [frame1, frame2, ... frame8] ]
                # This creates a Batch Size of 1, Video Length of 8.
                inputs = processor(
                    text=PROMPTS, 
                    videos=[buffer_snapshot],  # <--- THE FIX
                    return_tensors="pt", 
                    padding=True
                )
                
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_video.softmax(dim=1)
                
                top_prob, top_idx = probs[0].topk(1)
                conf = top_prob.item()
                label = PROMPTS[top_idx.item()]
                clean_label = label.replace("a video of ", "")

                # Debug Print to verify it's working
                print(f"Seen: {clean_label} ({conf:.2f})")

                status = "OBSERVE"
                if label in DANGER_LABELS:
                    status = "INTERCEPT" if conf > CONFIDENCE_THRESHOLD else "WARNING"
                
                with state.lock:
                    state.status = status
                    state.action = clean_label
                    state.conf = conf

            except Exception as e:
                print(f"X-CLIP Error: {e}")
                # If X-CLIP fails, fallback to simple output
                with state.lock:
                    state.action = "Analysis Failed"
        
        # Limit inference speed to keep heat/load down (2 FPS for AI is enough)
        time.sleep(0.25) 

# ---------------------------
# 3. Flask Server
# ---------------------------
app = Flask(__name__)

def generate_mjpeg():
    while state.running:
        with state.lock:
            if state.frame is None: 
                continue
            display_frame = state.frame.copy()
            status = state.status
            action = state.action
            conf = state.conf
            buf_len = len(state.buffer)

        # UI
        color = (0, 255, 0)
        if status == "INTERCEPT": color = (0, 0, 255)
        elif status == "WARNING": color = (0, 165, 255)

        cv2.rectangle(display_frame, (0,0), (640, 85), (0,0,0), -1)
        cv2.putText(display_frame, f"STATUS: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"{action} ({conf:.1%})", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Buffer Progress Bar
        bar_w = int((buf_len / SEQUENCE_LENGTH) * 640)
        cv2.line(display_frame, (0, 478), (bar_w, 478), (255, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.04) # 25 FPS web stream

@app.route('/')
def index():
    return render_template_string('<img src="/video_feed" width="640" style="border: 2px solid #444;">')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start threads
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=ai_worker, daemon=True).start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)