#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import os
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # 1. Initialize YOLOv8 with ByteTrack
        # 'yolov8n.pt' will download automatically if not present.
        # We use the 'nano' version for max FPS on Jetson.
        package_share_directory = get_package_share_directory('PatrolBot')
        model_path = os.path.join(package_share_directory, 'yolov8n.engine')
        self.model = YOLO(model_path, task='detect')  # Use engine file for TensorRT speed

        # 2. ROS 2 Communication
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Updated for Limo's Orbbec Astra
            self.image_callback,
            10)
        
        # Publisher for the VLM (JSON data)
        self.tracking_pub = self.create_publisher(String, '/tracked_persons', 10)
        
        # Publisher for Debugging (Visual image with IDs drawn)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info("YOLOv8 Tracker Node Started on GPU")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 3. Run YOLOv8 Tracking (ByteTrack is built-in!)
            # persist=True is crucial: it tells YOLO to keep IDs across frames
            # classes=[0] ensures we ONLY track people (Class 0 in COCO)
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            tracked_persons = []
            
            # 4. Extract Data & Draw IDs
            # Ultralytics handles drawing boxes/IDs automatically on 'res_plotted'
            res_plotted = results[0].plot()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()  # x, y, width, height
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    
                    # Prepare data for VLM
                    person_data = {
                        "id": track_id,
                        "center": [float(x), float(y)],
                        "size": [float(w), float(h)]
                    }
                    tracked_persons.append(person_data)

                    # (Optional) Add extra visual "pop" for the VLM if needed
                    # The .plot() method already draws the ID, but you can draw a giant green number 
                    # here if the VLM struggles to read the small default text.
                    cv2.putText(res_plotted, f"ID: {track_id}", (int(x), int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 5. Publish Data
            # Publish JSON for the VLM/Control node
            msg_json = String()
            msg_json.data = json.dumps(tracked_persons)
            self.tracking_pub.publish(msg_json)

            # Publish Visual Image for VLM Node to look at
            debug_msg = self.bridge.cv2_to_imgmsg(res_plotted, "bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in tracker loop: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()