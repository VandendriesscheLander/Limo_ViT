#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import torch
import json
import time
import os
from transformers import AutoProcessor, AutoModel 

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # --- CONFIGURATION ---
        self.model_repo = "google/siglip-base-patch16-224"
        self.local_model_path = "./src/patrol_bot/models/siglip" # Local save path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.confidence_threshold = 0.45 
        self.inference_interval = 0.5
        self.margin_factor = 0.3  # Add 30% padding around the person for context

        self.get_logger().info(f"Setting up on {self.device}")
        
        # --- MODEL LOADING (OFFLINE CAPABLE) ---
        try:
            # 1. Check if we have the model locally
            if os.path.exists(self.local_model_path):
                self.get_logger().info(f"Loading from local storage: {self.local_model_path}")
                self.processor = AutoProcessor.from_pretrained(self.local_model_path)
                self.model = AutoModel.from_pretrained(self.local_model_path)
            else:
                # 2. If not, download and save it
                self.get_logger().info(f"Model not found locally. Downloading {self.model_repo}...")
                self.processor = AutoProcessor.from_pretrained(self.model_repo)
                self.model = AutoModel.from_pretrained(self.model_repo)
                
                self.get_logger().info(f"Saving model to {self.local_model_path} for future offline use...")
                self.model.save_pretrained(self.local_model_path)
                self.processor.save_pretrained(self.local_model_path)

            # 3. Move to GPU and Optimize
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.get_logger().error(f"CRITICAL LOAD ERROR: {e}")
            return

        # --- PROMPTS ---
        self.danger_prompts = [
            "violence", "fighting", "punching", "kicking", "attack",
            "aggressive pose", "people wrestling", "a struggle", 
            "play fighting", "dramatic acting", "raised fists", "threat",
            "a person crouching", "a person hidden behind an object"
        ]

        self.safe_prompts = [
            "a hallway", "a floor", "a wall", "background",
            "walking", "standing", "talking", "hugging", 
            "shaking hands", "dancing", "playing", "friendly"
        ]

        self.all_prompts = self.danger_prompts + self.safe_prompts
        
        # --- ROBUST TOKENIZATION ---
        self.get_logger().info("Tokenizing prompts...")
        self.text_inputs = self.processor(
            text=self.all_prompts, 
            padding="max_length", 
            return_tensors="pt"
        ).to(self.device)

        # --- ROS SETUP ---
        # Reduced queue size to 1 to prevent image buffering if GPU lags
        self.subscription = self.create_subscription(Image, '/debug_image', self.image_callback, 1)
        
        # NEW: Subscribe to the tracker node to get bounding boxes
        self.tracker_sub = self.create_subscription(String, '/tracked_persons', self.tracker_callback, 10)
        
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        self.brain_ready_pub = self.create_publisher(Bool, '/brain_ready', 10)
        
        self.bridge = CvBridge()
        self.last_run_time = 0
        self.latest_tracked_persons = []
        
        self.get_logger().info("Brain Node Ready (Per-Person Analysis Mode).")

    def tracker_callback(self, msg):
        """Update the latest known tracking data."""
        try:
            self.latest_tracked_persons = json.loads(msg.data)
        except ValueError:
            pass

    def image_callback(self, msg):
        ready_msg = Bool()
        ready_msg.data = True
        self.brain_ready_pub.publish(ready_msg)

        now = time.time()
        if now - self.last_run_time < self.inference_interval:
            return
        self.last_run_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = rgb_img.shape

            # 1. Prepare Crops (Per Person)
            crops = []
            crop_metadata = [] # Stores dicts with ID and other info per crop

            # If we have tracked persons, crop them with margin
            if self.latest_tracked_persons:
                for person in self.latest_tracked_persons:
                    cx, cy = person['center']
                    w, h = person['size']
                    
                    # Calculate padding (to see the context/scenario)
                    pad_w = w * self.margin_factor
                    pad_h = h * self.margin_factor
                    
                    # Calculate coordinates with boundary checks
                    x1 = int(max(0, cx - w/2 - pad_w))
                    y1 = int(max(0, cy - h/2 - pad_h))
                    x2 = int(min(img_w, cx + w/2 + pad_w))
                    y2 = int(min(img_h, cy + h/2 + pad_h))
                    
                    if x2 > x1 and y2 > y1:
                        crop = rgb_img[y1:y2, x1:x2]
                        crops.append(crop)
                        crop_metadata.append(person)

            # Fallback: If no one is tracked, use the whole image (target_id=None)
            if not crops:
                crops.append(rgb_img)
                crop_metadata.append({"id": None})

            # 2. Batch Inference
            # The processor can accept a list of images (crops)
            image_inputs = self.processor(images=crops, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model(
                    pixel_values=image_inputs['pixel_values'],
                    input_ids=self.text_inputs['input_ids'],
                    attention_mask=self.text_inputs.get('attention_mask')
                )
                
                # logits_per_image shape: [batch_size, num_prompts]
                logits_per_image = outputs.logits_per_image
                probs_batch = torch.sigmoid(logits_per_image) 

            # 3. Analyze Results per Crop
            suspicious_found = []
            safe_found = []

            for i, probs in enumerate(probs_batch):
                danger_score = probs[:len(self.danger_prompts)].sum().item()
                safe_score = probs[len(self.danger_prompts):].sum().item()
                
                # Retrieve the ID associated with this crop
                person_id = crop_metadata[i].get("id")

                if danger_score > safe_score and danger_score > self.confidence_threshold:
                    
                    danger_probs = probs[:len(self.danger_prompts)]
                    top_idx = danger_probs.argmax().item()
                    specific_reason = self.danger_prompts[top_idx]

                    suspicious_found.append({
                        "suspicious": True,
                        "reason": specific_reason,
                        "confidence": round(danger_score, 2),
                        "target_id": person_id 
                    })
                else:
                    safe_probs = probs[len(self.danger_prompts):]
                    top_safe_idx = safe_probs.argmax().item()
                    top_safe = self.safe_prompts[top_safe_idx]

                    safe_found.append({
                        "suspicious": False,
                        "reason": top_safe,
                        "confidence": round(safe_score, 2),
                        "target_id": person_id
                    })

            # 4. Decision Logic: Prioritize Suspicious Events
            final_result = None

            if suspicious_found:
                # If any person is suspicious, pick the one with highest confidence
                final_result = max(suspicious_found, key=lambda x: x['confidence'])
                self.get_logger().warn(f"DETECTED: {final_result['reason']} (ID: {final_result['target_id']})")
            elif safe_found:
                # If all are safe, pick the highest confidence safe result (or just the first one)
                final_result = max(safe_found, key=lambda x: x['confidence'])
                self.get_logger().info(f"Safe: {final_result['reason']} (ID: {final_result['target_id']})")

            # 5. Publish
            if final_result:
                msg_out = String()
                msg_out.data = json.dumps(final_result)
                self.publisher.publish(msg_out)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main(args=None):
    rclpy.init(args=args)
    node = ClipBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()