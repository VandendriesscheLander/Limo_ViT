import torch
import threading
import time
import cv2
import copy
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from uform.gen_model import VLMProcessor, VLMForCausalLM
from PIL import Image

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "unum-cloud/uform-gen2-dpo"
DEVICE = "cpu" # Force CPU since you are likely memory limited
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Loading UForm-Gen2 (0.5B) to {DEVICE}...")
# UForm is extremely lightweight
model = VLMForCausalLM.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
processor = VLMProcessor.from_pretrained(MODEL_ID)

print("Loading YOLOv8n (Gatekeeper)...")
yolo = YOLO("yolov8n.pt") 

# ---------------------------
# Globals & Locks
# ---------------------------
# We separate the "Current Frame" from the "Inference Frame"
frame_lock = threading.Lock()
latest_frame = None  # The raw frame from camera
output_frame = None  # The annotated frame for the web browser

state_lock = threading.Lock()
current_status = "PATROL"
current_caption = "System Initializing..."
last_inference_time = 0

app = Flask(__name__)

# ---------------------------
# AI Worker Thread
# ---------------------------
def ai_worker():
    """
    Runs in the background. It wakes up, grabs the latest frame, 
    thinks about it, updates the global status, and sleeps.
    It NEVER blocks the video feed.
    """
    global current_status, current_caption, last_inference_time
    
    # Prompt to check for violence/action
    prompt = "Describe this image in detail."
    danger_keywords = ["fight", "punch", "kick", "knife", "gun", "weapon", "blood", "attack", "hitting"]

    while True:
        # 1. Grab a snapshot of the latest frame
        img_snapshot = None
        with frame_lock:
            if latest_frame is not None:
                img_snapshot = latest_frame.copy()
        
        # If no frame yet (system starting), wait
        if img_snapshot is None:
            time.sleep(0.1)
            continue

        # 2. Gatekeeper: Only run VLM if YOLO sees a person
        # This saves massive CPU by not processing empty hallways
        results = yolo(img_snapshot, verbose=False, classes=[0], conf=0.4)
        person_detected = len(results[0].boxes) > 0

        if not person_detected:
            with state_lock:
                current_status = "PATROL"
                current_caption = "Area Clear (No Person)"
            time.sleep(0.5) # Scan less frequently when empty
            continue

        # 3. Run VLM Inference (The heavy part)
        try:
            start_t = time.time()
            
            # Convert OpenCV (BGR) to PIL (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(img_snapshot, cv2.COLOR_BGR2RGB))
            
            # Prepare inputs
            inputs = processor(texts=[prompt], images=[pil_image], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Generate Description
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=40 # Keep brief for speed
                )
            
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            inference_dur = time.time() - start_t

            # 4. Logic Analysis
            is_danger = any(k in text.lower() for k in danger_keywords)
            
            with state_lock:
                if is_danger:
                    current_status = "INTERCEPT"
                    current_caption = f"DANGER: {text}"
                else:
                    current_status = "OBSERVE"
                    current_caption = f"Safe: {text}"
                
                last_inference_time = inference_dur

        except Exception as e:
            print(f"AI Error: {e}")
            with state_lock:
                current_caption = "AI Error"

        # Adaptive sleep: If CPU is hot, sleep longer.
        # This prevents the AI thread from eating 100% CPU.
        time.sleep(0.1) 

# ---------------------------
# Main Video Loop (The Fast Loop)
# ---------------------------
def generate_frames():
    global latest_frame, output_frame
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Start the AI thread once
    t = threading.Thread(target=ai_worker, daemon=True)
    t.start()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Update the shared frame for the AI thread to see
        with frame_lock:
            latest_frame = frame

        # Read the latest status from the AI thread
        with state_lock:
            status = current_status
            caption = current_caption
            latency = last_inference_time

        # --- DRAW UI ---
        # We draw on the video thread so it's always 30fps
        # even if the caption is 5 seconds old.
        color = (0, 255, 0) # Green
        if status == "INTERCEPT":
            color = (0, 0, 255) # Red
        elif status == "PATROL":
            color = (255, 200, 0) # Yellow

        # Top Bar
        cv2.rectangle(frame, (0, 0), (640, 90), (0,0,0), -1)
        
        # Status
        cv2.putText(frame, f"STATUS: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Caption (Auto-wrap visually by slicing)
        display_cap = caption[:50] + "..." if len(caption) > 50 else caption
        cv2.putText(frame, f"AI: {display_cap}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats
        cv2.putText(frame, f"AI Latency: {latency:.2f}s", (480, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('<img src="/video_feed" width="100%">')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)