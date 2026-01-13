#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import os
from ultralytics import YOLO

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # 1. Initialize Direct Camera Access (Bypassing ROS Topics)
        # Try index 0 (default), then 1 if that fails.
        self.get_logger().info("Attempting to open camera at /dev/video0...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().warn("Failed to open /dev/video0. Trying /dev/video1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.get_logger().error("CRITICAL: Could not open any camera device!")
        
        # Optimize for speed (640x480 is plenty for YOLO Nano)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 2. Initialize YOLOv8
        self.get_logger().info("Loading YOLOv8 model...")
        model_path = '/home/agilex/limo_ros2_ws/src/PatrolBot/yolov8n.engine'
        
        self.get_logger().info(f"Using model: {model_path}")
        self.model = YOLO(model_path, task='detect') 

        # 3. Publishers
        # This publishes the JSON data for the Brain/Control nodes
        self.tracking_pub = self.create_publisher(String, '/tracked_persons', 10)
        # This publishes the Visual Video Feed (with boxes) for the Dashboard/Brain
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        self.bridge = CvBridge()

        # 4. Timer Loop (The "Heartbeat")
        # Instead of waiting for a message, we grab a frame every 33ms (~30 FPS)
        self.timer = self.create_timer(0.033, self.process_frame)
        
        self.get_logger().info("Tracker Node (Direct Camera Mode) Started.")

    def process_frame(self):
        # 1. Read from Hardware
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No frame from camera", throttle_duration_sec=2.0)
            return

        try:
            # 2. Run YOLOv8 Tracking
            # persist=True maintains IDs (1, 2, 3...) across frames
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            tracked_persons = []
            
            # 3. Draw the Boxes & IDs
            res_plotted = results[0].plot()

            # 4. Extract Data for the Brain
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    person_data = {
                        "id": track_id,
                        "center": [float(x), float(y)],
                        "size": [float(w), float(h)]
                    }
                    tracked_persons.append(person_data)

                    # Optional: Draw a massive ID on the screen so the CLIP Brain can see it easily
                    cv2.putText(res_plotted, f"ID: {track_id}", (int(x), int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 5. Publish Data (JSON)
            msg_json = String()
            msg_json.data = json.dumps(tracked_persons)
            self.tracking_pub.publish(msg_json)

            # 6. Publish Video (Image)
            # This is what your Web Dashboard and CLIP Brain will see
            debug_msg = self.bridge.cv2_to_imgmsg(res_plotted, "bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in tracker loop: {str(e)}')

    def destroy_node(self):
        # Clean up camera when shutting down
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()