import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import torch
import clip
from PIL import Image as PILImage

class ClipBrainNode(Node):
    def __init__(self):
        super().__init__('clip_brain_node')
        
        # 1. Load CLIP (Standard OpenAI version)
        # "ViT-B/32" is a good balance of speed vs accuracy for Jetson
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 2. Define the "Labels" to check against
        # CLIP works by comparing the image to these text descriptions.
        self.suspicious_prompts = [
            "a person holding a weapon", 
            "a person fighting", 
            "a person punching someone",
            "a person breaking into a door"
        ]
        self.safe_prompts = [
            "a person walking normally", 
            "a person standing still", 
            "a person sitting",
            "an empty room"
        ]
        
        # Pre-encode text so we don't re-compute it every frame (Saves huge compute)
        all_text = self.suspicious_prompts + self.safe_prompts
        text_inputs = clip.tokenize(all_text).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.subscription = self.create_subscription(
            Image, '/debug_image', self.image_callback, 10)
        self.publisher = self.create_publisher(String, '/pursuit_target', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS -> PIL
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            # Prepare image for CLIP
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            # Compare Image vs Text
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)
                
                top_index = indices[0].item()
                confidence = values[0].item()

            # Logic: Is the top match one of our "suspicious" prompts?
            is_suspicious = top_index < len(self.suspicious_prompts)
            
            if is_suspicious and confidence > 0.6: # Threshold to prevent false positives
                result = {
                    "suspicious": True,
                    "reason": self.suspicious_prompts[top_index],
                    "confidence": f"{confidence:.2f}"
                    # Note: You need to grab the ID from the tracker topic separately
                    # or rely on the tracker to be sending the ID synced with this image.
                }
                self.get_logger().warn(f"DETECTED: {result['reason']}")
                msg = String()
                msg.data = json.dumps(result)
                self.publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"CLIP Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ClipBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()