import torch
import threading
import time
import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

# ---------------------------
# Config
# ---------------------------
# Florence-2-base is ~230M params (Tiny compared to Moondream's 1.6B)
MODEL_ID = "microsoft/Florence-2-base" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 if on GPU to save even more VRAM, otherwise float32 for CPU
TORCH_DTYPE = torch.float16 if DEVICE == "cuda"

# Keywords to trigger "INTERCEPT" status from the generated caption
DANGER_KEYWORDS = [
    "fight", "punch", "kick", "hit", "stab", "shoot", "gun", 
    "weapon", "knife", "attack", "violence", "blood", "wrestling", "strangling"
]

# ---------------------------
# Load Models
# ---------------------------
print(f"Loading Florence-2-base to {DEVICE}...")
# trust_remote_code=True is required for Florence-2
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=TORCH_DTYPE, 
    trust_remote_code=True
    ).to(DEVICE)
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
person_detected = False
current_caption = ""

app = Flask(__name__)

# ---------------------------
# VLM Processing Function
# ---------------------------
def process_frame_with_florence(frame_rgb):
    """
    Generates a caption describing the scene and checks for danger keywords.
    """
    global latest_status, latest_action, current_caption

    try:
        # 1. Prepare Prompt: <DETAILED_CAPTION> gives richer context than <CAPTION>
        # You can also use <OD> to get bounding boxes if you want "Action" coordinates later
        prompt = "<DETAILED_CAPTION>"
        
        # 2. Preprocess
        inputs = processor(text=prompt, images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(DEVICE, TORCH_DTYPE) if v.dtype.is_floating_point else v.to(DEVICE) for k, v in inputs.items()}

        # 3. Generate (Inference)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,  # Keep it short for speed
                num_beams=3         # Lower beams = faster, less accurate
            )

        # 4. Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process to get pure text
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(frame_rgb.shape[1], frame_rgb.shape[0])
        )
        
        description = parsed_answer[prompt].lower()
        current_caption = description # Save for UI

        # 5. Logic / Decision Making
        is_danger = any(keyword in description for keyword in DANGER_KEYWORDS)

        with output_lock:
            if is_danger:
                latest_status = "INTERCEPT"
                # Find which keyword triggered it for the status
                trigger = next((k for k in DANGER_KEYWORDS if k in description), "unknown")
                latest_action = f"DANGER DETECTED: {trigger.upper()}"
            else:
                latest_status = "OBSERVE"
                # Truncate long descriptions for display
                display_desc = (description[:30] + '..') if len(description) > 30 else description
                latest_action = f"Safe: {display_desc}"

    except Exception as e:
        print(f"Florence Error: {e}")

# ---------------------------
# Main Video Loop
# ---------------------------
def generate_frames():
    global person_detected
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    frame_count = 0
    FRAME_SKIP = 10 # Florence is slightly heavier than CLIP, so maybe skip more frames

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. YOLO Check (Gatekeeper)
        # Only detecting 'person' (class 0)
        results = yolo(frame, verbose=False, classes=[0], conf=0.4)
        person_detected = len(results[0].boxes) > 0
        annotated_frame = results[0].plot()

        # 2. Run Florence-2 VLM logic
        if person_detected and frame_count % FRAME_SKIP == 0:
            # Convert to RGB for Transformers
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run in a separate thread so video doesn't stutter? 
            # For simplicity here, we run blocking, but in production use a thread.
            # Using a simple daemon thread to not block the render loop:
            threading.Thread(target=process_frame_with_florence, args=(rgb,), daemon=True).start()

        frame_count += 1

        # 3. Draw UI
        with output_lock:
            status = latest_status
            action = latest_action
            caption = current_caption

        # Dynamic Color: Red for Danger, Green for Safe, Yellow for Patrol
        if status == "INTERCEPT":
            color = (0, 0, 255) # Red
        elif status == "OBSERVE":
            color = (0, 255, 0) # Green
        else:
            color = (0, 255, 255) # Yellow

        # UI Overlay
        cv2.rectangle(annotated_frame, (0, 0), (640, 100), (0, 0, 0), -1)
        
        # Status Line
        cv2.putText(annotated_frame, f"STATUS: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Action/Reason Line
        cv2.putText(annotated_frame, f"ACT: {action}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Full Caption Line (Small)
        cv2.putText(annotated_frame, f"VLM: {caption[:60]}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <body style="background:black; color:white; text-align:center;">
                <h1>VLA Security Node (Florence-2)</h1>
                <img src="/video_feed" style="border: 2px solid grey;">
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Threaded is vital for flask + video stream
    app.run(host='0.0.0.0', port=5000, threaded=True)