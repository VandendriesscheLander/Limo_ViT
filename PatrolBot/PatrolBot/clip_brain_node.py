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

# Hugging Face Transformers
from transformers import CLIPProcessor, CLIPModel

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # --- CONFIGURATION ---
        # We use standard CLIP (ViT-B/32) as requested.
        self.model_name = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = 0.3
        self.inference_interval = 0.5 # Run max 2 times per second (prevent overheating)

        # --- LOAD MODELS ---
        self.get_logger().info(f"Loading CLIP: {self.model_name} on {self.device}...")
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

        # --- PROMPTS ---
        self.prompts = [
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

        self.danger_labels = [
            "a photo of a physical fight",
            "a photo of a person punching someone",
            "a photo of a person kicking someone",
            "a photo of a person stabbing someone",
            "a photo of a person shooting a gun",
            "a photo of a person wrestling with another person",
            "a photo of someone being attacked",
            "a photo of a person holding a weapon"
        ]

        # Pre-tokenize text (Optimization)
        self.get_logger().info("Pre-tokenizing prompts...")
        self.inputs_text = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)

        # --- ROS SETUP ---
        # We subscribe to the tracker's debug output so we see what the tracker sees
        self.subscription = self.create_subscription(
            Image,
            '/debug_image', 
            self.image_callback,
            10)
            
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        self.bridge = CvBridge()
        self.last_run_time = 0
        
        self.get_logger().info("CLIP Brain Node Ready.")

    def image_callback(self, msg):
        # Rate Limiting
        if time.time() - self.last_run_time < self.inference_interval:
            return
        self.last_run_time = time.time()

        try:
            # 1. Convert ROS Image -> OpenCV -> RGB
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 2. Prepare Inputs
            # Note: We pass the pre-tokenized text separately to save compute
            inputs = self.processor(images=rgb_img, return_tensors="pt", padding=True)
            inputs['input_ids'] = self.inputs_text['input_ids']
            inputs['attention_mask'] = self.inputs_text['attention_mask']
            
            # Move image inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 3. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image # image-text similarity
                probs = logits_per_image.softmax(dim=1) # probabilities

                # Get top prediction
                top_prob, top_idx = probs[0].topk(1)
                top_label = self.prompts[top_idx.item()]
                conf = top_prob.item()

            # 4. Logic
            is_danger = top_label in self.danger_labels
            
            if is_danger and conf > self.confidence_threshold:
                result = {
                    "suspicious": True,
                    "reason": top_label,
                    "confidence": round(conf, 2),
                    # We don't know the exact ID here, but the controller handles "None"
                    # by targeting the closest person.
                    "target_id": None 
                }
                self.get_logger().warn(f"DETECTED: {top_label} ({conf:.2f})")
                
                # Publish to Control Node
                msg_out = String()
                msg_out.data = json.dumps(result)
                self.publisher.publish(msg_out)
            else:
                # Optional: Logging safe state occasionally
                # self.get_logger().info(f"Safe: {top_label} ({conf:.2f})")
                pass

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ClipBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()