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
from transformers import AutoProcessor, AutoModel 

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # --- CONFIGURATION ---
        self.model_name = "google/siglip-base-patch16-224"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = 0.45 
        self.inference_interval = 0.5 

        self.get_logger().info(f"Loading Model: {self.model_name} on {self.device}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
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
        # We process text ONCE and move it to device immediately.
        # We explicitly ask for padding to ensure consistent tensor shapes.
        self.text_inputs = self.processor(
            text=self.all_prompts, 
            padding="max_length", 
            return_tensors="pt"
        ).to(self.device)

        # --- ROS SETUP ---
        self.subscription = self.create_subscription(Image, '/debug_image', self.image_callback, 10)
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        self.bridge = CvBridge()
        self.last_run_time = 0
        
        self.get_logger().info("Brain Node Ready (SigLIP Mode - Demo).")

    def image_callback(self, msg):
        if time.time() - self.last_run_time < self.inference_interval:
            return
        self.last_run_time = time.time()

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 1. Process Image Only
            # This returns a dict containing 'pixel_values'
            image_inputs = self.processor(images=rgb_img, return_tensors="pt").to(self.device)

            # 2. Run Inference with EXPLICIT Arguments
            # Instead of merging dicts, we pass named arguments.
            # We use .get() for the mask so it doesn't crash if missing.
            with torch.no_grad():
                outputs = self.model(
                    pixel_values=image_inputs['pixel_values'],
                    input_ids=self.text_inputs['input_ids'],
                    attention_mask=self.text_inputs.get('attention_mask') # Safe access
                )
                
                # SigLIP: Use Sigmoid
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

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ClipBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()