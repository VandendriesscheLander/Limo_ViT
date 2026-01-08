import cv2
import torch
import threading
import time
from collections import deque
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Config
# ---------------------------
# We use standard CLIP (ViT-B/32). It is a pure ViT and runs great on Orin Nano.
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SKIP = 5 

# Prompt Engineering: Expanded to better distinguish between violence and normal tasks
PROMPTS = [
    # --- DANGER ---
    "a photo of a physical fight",
    "a photo of a person punching someone",
    "a photo of a person kicking someone",
    "a photo of a person stabbing someone",
    "a photo of a person shooting a gun",
    "a photo of a person wrestling with another person",
    "a photo of someone being attacked",
    "a photo of a person holding a weapon",
    
    # --- SAFE ---
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

DANGER_LABELS = [
    "a photo of a physical fight",
    "a photo of a person punching someone",
    "a photo of a person kicking someone",
    "a photo of a person stabbing someone",
    "a photo of a person shooting a gun",
    "a photo of a person wrestling with another person",
    "a photo of someone being attacked",
    "a photo of a person holding a weapon"
]

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

app = Flask(__name__)

# ---------------------------
# Inference Thread
# ---------------------------
def inference_loop():
    global latest_status, latest_action, latest_conf
    
    # Pre-tokenize text once (Saves massive compute)
    inputs_text = processor(text=PROMPTS, return_tensors="pt", padding=True).to(DEVICE)

    while True:
        # 1. Gatekeeper: Only run ViT if human is present
        if not person_detected:
            with output_lock:
                latest_status = "PATROL"
                latest_action = "Area Clear"
                latest_conf = 0.0
            time.sleep(0.5)
            continue

        # 2. Grab latest frame from global buffer (conceptually)
        # For simplicity in this threaded model, we will grab from a shared variable in production,
        # but here we rely on the logic that we only run when YOLO says so.
        # *In a real implementation, you'd pass the frame object.* # (See integration below)
        
        time.sleep(0.2) # Wait for next check

def process_frame_with_clip(frame_rgb):
    """Runs single-frame ViT classification"""
    global latest_status, latest_action, latest_conf

    try:
        # Preprocess Image
        inputs = processor(text=PROMPTS, images=frame_rgb, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is image-text similarity score
            probs = logits_per_image.softmax(dim=1) # softmax to get probabilities

            # Get top prediction
            top_prob, top_idx = probs[0].topk(1)
            top_label = PROMPTS[top_idx.item()]
            conf = top_prob.item()

        # Logic
        is_danger = top_label in DANGER_LABELS
        
        with output_lock:
            if is_danger:
                if conf > 0.3:
                    latest_status = "INTERCEPT"
                    latest_action = f"DANGER: {top_label.replace('a photo of ', '')}"
                else:
                    latest_status = "WARNING"
                    latest_action = f"Possible: {top_label.replace('a photo of ', '')}"
            else:
                latest_status = "OBSERVE"
                latest_action = f"Safe: {top_label.replace('a photo of ', '')}"
            latest_conf = conf

    except Exception as e:
        print(f"CLIP Error: {e}")

# ---------------------------
# Main Video Loop
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

        # 1. YOLO Check
        results = yolo(frame, verbose=False, classes=[0], conf=0.4)
        person_detected = len(results[0].boxes) > 0
        annotated_frame = results[0].plot()

        # 2. Run CLIP-ViT every N frames (if person exists)
        if person_detected and frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run inference directly here to simplify thread handling for this specific script
            process_frame_with_clip(rgb)

        frame_count += 1

        # 3. Draw UI
        with output_lock:
            status = latest_status
            action = latest_action
            conf = latest_conf

        color = (0,0,255) if status == "INTERCEPT" else (0,255,0)
        
        cv2.rectangle(annotated_frame, (0,0), (640, 80), (0,0,0), -1)
        cv2.putText(annotated_frame, f"{status}: {action} ({conf:.1%})", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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