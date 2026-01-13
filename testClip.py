import cv2
import torch
import threading
import time
from flask import Flask, Response, render_template_string
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Refined Prompts (Add more if you want to test specific scenarios)
PROMPTS = [
    # --- DANGER ---
    "a photo of a physical fight",
    "a photo of a person punching someone",
    "a photo of a person kicking",
    "a photo of a person holding a knife",
    "a photo of a person holding a gun",
    "a photo of a person wrestling",
    "a photo of violence",
    
    # --- SAFE / CONFUSERS ---
    # Adding specific "confusers" helps the model distinguish better
    "a photo of people shaking hands",
    "a photo of people hugging",
    "a photo of a person standing still",
    "a photo of a person walking",
    "a photo of a person sitting",
    "a photo of a blurry image",
    "a photo of an empty room",
    "a photo of a robot",
    "a photo of a person holding a cellphone" 
]

# ---------------------------
# Load Model
# ---------------------------
print(f"Loading CLIP on {DEVICE}...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Pre-tokenize text prompts once
print("Pre-tokenizing prompts...")
inputs_text = processor(text=PROMPTS, return_tensors="pt", padding=True).to(DEVICE)

# ---------------------------
# Global State
# ---------------------------
output_lock = threading.Lock()
latest_frame = None
top_predictions = []

app = Flask(__name__)

# ---------------------------
# Inference Thread
# ---------------------------
def inference_loop():
    global top_predictions, latest_frame
    
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        try:
            # 1. Grab latest frame (Thread safe copy)
            with output_lock:
                frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)

            # 2. Process Image
            inputs = processor(images=frame_rgb, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Add text inputs
            inputs['input_ids'] = inputs_text['input_ids']
            inputs['attention_mask'] = inputs_text['attention_mask']

            # 3. Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                # Get Top 3
                top_probs, top_idxs = probs[0].topk(3)
                
                new_preds = []
                for i in range(3):
                    idx = top_idxs[i].item()
                    score = top_probs[i].item()
                    label = PROMPTS[idx]
                    new_preds.append((label, score))
                
                top_predictions = new_preds

        except Exception as e:
            print(f"Inference Error: {e}")
        
        # Run at max 2-3 Hz to save heat
        time.sleep(0.3)

# ---------------------------
# Camera Thread
# ---------------------------
def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    
    # Lower resolution for speed
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if success:
            with output_lock:
                latest_frame = frame
        else:
            time.sleep(0.1)

# ---------------------------
# Web Server
# ---------------------------
def generate_mjpeg():
    while True:
        with output_lock:
            if latest_frame is None:
                continue
            
            # Draw overlay on the JPG sent to browser (not the raw frame)
            display_frame = latest_frame.copy()
            
            # Draw Top 3 predictions
            y_start = 30
            for i, (label, score) in enumerate(top_predictions):
                color = (0, 0, 255) if "photo of a person holding a" in label or "fight" in label else (0, 255, 0)
                text = f"{i+1}. {label}: {score:.1%}"
                
                # Black background for readability
                cv2.rectangle(display_frame, (5, y_start - 25), (450, y_start + 5), (0,0,0), -1)
                cv2.putText(display_frame, text, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_start += 40

            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head>
            <title>CLIP Debugger</title>
            <style>
                body { background-color: #222; color: white; font-family: monospace; text-align: center; }
                img { border: 2px solid #555; margin-top: 20px; max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>CLIP-ViT Real-Time Debugger</h1>
            <img src="/video_feed" />
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start threads
    t1 = threading.Thread(target=camera_loop)
    t2 = threading.Thread(target=inference_loop)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

    print("Starting Web Server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)