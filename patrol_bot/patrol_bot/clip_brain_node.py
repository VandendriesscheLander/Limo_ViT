#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import json
import time
import os
import numpy as np
from transformers import AutoProcessor, AutoModel 

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # --- CONFIGURATION ---
        self.model_repo = "google/siglip-base-patch16-224"
        self.local_model_path = "./src/patrol_bot/models/siglip" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.confidence_threshold = 0.45 
        self.inference_interval = 0.5 
        
        # Margin around the person to include context (percentage of box size)
        self.crop_margin = 0.2 

        self.get_logger().info(f"Setting up on {self.device}")
        
        # --- MODEL LOADING (OFFLINE CAPABLE) ---
        try:
            if os.path.exists(self.local_model_path):
                self.get_logger().info(f"Loading from local storage: {self.local_model_path}")
                self.processor = AutoProcessor.from_pretrained(self.local_model_path)
                self.model = AutoModel.from_pretrained(self.local_model_path)
            else:
                self.get_logger().info(f"Model not found locally. Downloading {self.model_repo}...")
                self.processor = AutoProcessor.from_pretrained(self.model_repo)
                self.model = AutoModel.from_pretrained(self.model_repo)
                
                self.get_logger().info(f"Saving model to {self.local_model_path}...")
                self.model.save_pretrained(self.local_model_path)
                self.processor.save_pretrained(self.local_model_path)

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
        
        # --- PRE-TOKENIZATION ---
        self.get_logger().info("Tokenizing prompts...")
        self.text_inputs = self.processor(
            text=self.all_prompts, 
            padding="max_length", 
            return_tensors="pt"
        ).to(self.device)

        # --- ROS SETUP ---
        self.subscription = self.create_subscription(Image, '/debug_image', self.image_callback, 1)
        # Subscribe to the tracker to get bounding boxes
        self.tracker_sub = self.create_subscription(String, '/tracked_persons', self.tracker_callback, 10)
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        
        self.bridge = CvBridge()
        self.last_run_time = 0
        
        # State
        self.latest_tracks = []
        self.last_track_time = 0
        
        self.get_logger().info("Brain Node Ready (Batched RoI Mode).")

    def tracker_callback(self, msg):
        try:
            self.latest_tracks = json.loads(msg.data)
            self.last_track_time = time.time()
        except ValueError:
            pass

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_run_time < self.inference_interval:
            return
        self.last_run_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = rgb_img.shape

            # 1. Prepare Batch
            # If tracks are stale (>1.0s old), ignore them and use full image fallback
            use_tracks = (len(self.latest_tracks) > 0) and (now - self.last_track_time < 1.0)
            
            images_to_process = []
            metadata_map = [] # To map batch index back to track ID

            if use_tracks:
                for person in self.latest_tracks:
                    # Extract Box
                    cx, cy = person['center']
                    w, h = person['size']
                    
                    # Add Context Margin
                    margin_w = w * self.crop_margin
                    margin_h = h * self.crop_margin
                    
                    x1 = int(max(0, cx - w/2 - margin_w))
                    y1 = int(max(0, cy - h/2 - margin_h))
                    x2 = int(min(img_w, cx + w/2 + margin_w))
                    y2 = int(min(img_h, cy + h/2 + margin_h))
                    
                    # Validate crop
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    crop = rgb_img[y1:y2, x1:x2]
                    images_to_process.append(crop)
                    metadata_map.append(person)
            else:
                # Fallback: Process full image
                images_to_process.append(rgb_img)
                metadata_map.append({"id": None})

            if not images_to_process:
                return

            # 2. Run Batched Inference
            # The processor handles lists of numpy arrays automatically
            image_inputs = self.processor(images=images_to_process, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model(
                    pixel_values=image_inputs['pixel_values'],
                    input_ids=self.text_inputs['input_ids'],
                    attention_mask=self.text_inputs.get('attention_mask')
                )
                
                # logits_per_image shape: [batch_size, num_prompts]
                logits_per_image = outputs.logits_per_image
                # Sigmoid to get independent probabilities
                probs_batch = torch.sigmoid(logits_per_image)

            # 3. Analyze Results
            best_suspicious_result = None
            safest_result = None
            
            for i, probs in enumerate(probs_batch):
                track_info = metadata_map[i]
                
                danger_score = probs[:len(self.danger_prompts)].sum().item()
                safe_score = probs[len(self.danger_prompts):].sum().item()
                
                # Determine status for this specific person/crop
                is_suspicious = False
                reason = ""
                confidence = 0.0
                
                if danger_score > safe_score and danger_score > self.confidence_threshold:
                    is_suspicious = True
                    danger_probs = probs[:len(self.danger_prompts)]
                    top_idx = danger_probs.argmax().item()
                    reason = self.danger_prompts[top_idx]
                    confidence = danger_score
                else:
                    safe_probs = probs[len(self.danger_prompts):]
                    top_idx = safe_probs.argmax().item()
                    reason = self.safe_prompts[top_idx]
                    confidence = safe_score

                result_obj = {
                    "suspicious": is_suspicious,
                    "reason": reason,
                    "confidence": round(confidence, 2),
                    "target_id": track_info.get("id") # Pass the ID through
                }
                
                # Logic: We want to prioritize reporting a suspicious target.
                # If multiple are suspicious, pick the one with highest danger score.
                if is_suspicious:
                    if best_suspicious_result is None or confidence > best_suspicious_result['confidence']:
                        best_suspicious_result = result_obj
                else:
                    # Keep track of the "best" safe result just in case we need to report something
                    if safest_result is None or confidence > safest_result['confidence']:
                        safest_result = result_obj

            # 4. Final Decision & Publish
            final_output = None
            
            if best_suspicious_result:
                final_output = best_suspicious_result
                self.get_logger().warn(f"DETECTED ID {final_output['target_id']}: {final_output['reason']} ({final_output['confidence']:.2f})")
            elif safest_result:
                final_output = safest_result
                # Log less frequently for safe
                self.get_logger().info(f"ID {final_output['target_id']}: {final_output['reason']} (Safe)")
            
            if final_output:
                msg_out = String()
                msg_out.data = json.dumps(final_output)
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