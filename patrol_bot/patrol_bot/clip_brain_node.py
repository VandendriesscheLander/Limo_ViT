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
from transformers import AutoProcessor, AutoModel 

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # --- CONFIGURATION ---
        self.model_repo = "google/siglip-base-patch16-224"
        self.local_model_path = "./model_data/siglip_local" # Local save path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.confidence_threshold = 0.45 
        self.inference_interval = 0.5 

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
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        self.bridge = CvBridge()
        self.last_run_time = 0
        
        self.get_logger().info("Brain Node Ready (Offline + FP16 Mode).")

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_run_time < self.inference_interval:
            return
        self.last_run_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 1. Process Image
            image_inputs = self.processor(images=rgb_img, return_tensors="pt").to(self.device)

            # 2. Run Inference
            # inference_mode() is slightly faster and safer for memory than no_grad()
            with torch.inference_mode():
                outputs = self.model(
                    pixel_values=image_inputs['pixel_values'],
                    input_ids=self.text_inputs['input_ids'],
                    attention_mask=self.text_inputs.get('attention_mask')
                )
                
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)[0] 

            # 3. Summing Scores
            danger_score = probs[:len(self.danger_prompts)].sum().item()
            safe_score = probs[len(self.danger_prompts):].sum().item()

            # 4. Decision Logic
            if danger_score > safe_score and danger_score > self.confidence_threshold:
                
                danger_probs = probs[:len(self.danger_prompts)]
                top_idx = danger_probs.argmax().item()
                specific_reason = self.danger_prompts[top_idx]

                result = {
                    "suspicious": True,
                    "reason": specific_reason,
                    "confidence": round(danger_score, 2),
                    "target_id": None 
                }
                
                self.get_logger().warn(f"DETECTED: {specific_reason} (Score: {danger_score:.2f})")
                
                msg_out = String()
                msg_out.data = json.dumps(result)
                self.publisher.publish(msg_out)

            else:
                safe_probs = probs[len(self.danger_prompts):]
                top_safe_idx = safe_probs.argmax().item()
                top_safe = self.safe_prompts[top_safe_idx]
                self.get_logger().info(f"Safe: {top_safe} (Safe: {safe_score:.2f} vs Danger: {danger_score:.2f})")

                result = {
                    "suspicious": False,
                    "reason": top_safe,
                    "confidence": round(safe_score, 2),
                    "target_id": None 
                }
                
                msg_out = String()
                msg_out.data = json.dumps(result)
                self.publisher.publish(msg_out)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")
            # Optional: trigger garbage collection if we hit a weird memory snag
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