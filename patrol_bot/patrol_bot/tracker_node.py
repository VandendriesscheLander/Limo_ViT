#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import time
from ultralytics import YOLO

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # State variables for auto-recovery
        self.cap = None
        self.fail_count = 0
        self.FAIL_THRESHOLD = 30  # Reset after ~1 second of failures (30 * 33ms)

        # 1. Initialize Camera (moved to a function)
        self.setup_camera()

        # 2. Initialize YOLO
        self.get_logger().info("Loading YOLO model...")
        # Ensure this path is correct for your robot
        model_path = '/home/agilex/limo_ros2_ws/src/patrol_bot/yolo26n.engine'
        
        try:
            self.model = YOLO(model_path, task='detect')
            self.get_logger().info(f"Using model: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            # Fallback to standard nano if engine fails, or exit
            # self.model = YOLO('yolov8n.pt') 

        # 3. Publishers
        self.tracking_pub = self.create_publisher(String, '/tracked_persons', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        self.bridge = CvBridge()

        # 4. Timer Loop
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info("Tracker Node (Self-Healing Mode) Started.")

    def setup_camera(self):
        """
        Handles the creation and configuration of the VideoCapture object.
        Can be called during init or to reset a frozen camera.
        """
        if self.cap is not None:
            # Release the old frozen handle
            self.get_logger().info("Releasing old camera handle...")
            self.cap.release()
            # strictly necessary to let the OS/Driver cleanup the USB resource
            time.sleep(1.0) 

        self.get_logger().info("Attempting to open camera (V4L2)...")
        
        # Explicitly asking for V4L2 backend often helps with select() timeouts
        # Try /dev/video0
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().warn("Failed to open /dev/video0. Trying /dev/video1...")
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Clear buffer by grabbing a dummy frame
            self.cap.grab()
            self.get_logger().info("Camera successfully (re)initialized.")
            self.fail_count = 0
        else:
            self.get_logger().error("CRITICAL: Could not open any camera device during setup.")

    def process_frame(self):
        # 1. Read from Hardware
        if self.cap is None or not self.cap.isOpened():
            self.fail_count += 1
            if self.fail_count % 30 == 0: # Log every second, not every 33ms
                self.get_logger().warn("Camera disconnected. Retrying...")
                self.setup_camera()
            return

        ret, frame = self.cap.read()

        # --- RECOVERY LOGIC START ---
        if not ret:
            self.fail_count += 1
            self.get_logger().warn(f"Dropped frame {self.fail_count}/{self.FAIL_THRESHOLD}", throttle_duration_sec=1.0)
            
            # If we miss 30 frames in a row, kill and restart the camera
            if self.fail_count >= self.FAIL_THRESHOLD:
                self.get_logger().error("Camera frozen (select timeout). Triggering hard reset...")
                self.setup_camera()
            return
        
        # If we got a frame, reset the counter
        self.fail_count = 0
        # --- RECOVERY LOGIC END ---

        try:
            # 2. Run YOLO Tracking
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            tracked_persons = []
            
            # 3. Draw Boxes & Prepare Data
            # Note: plot() creates a new copy of the image
            res_plotted = results[0].plot()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                if results[0].boxes.conf is not None:
                    confs = results[0].boxes.conf.cpu().numpy()
                else:
                    confs = [0.0] * len(boxes)

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x, y, w, h = box
                    person_data = {
                        "id": track_id,
                        "center": [float(x), float(y)],
                        "size": [float(w), float(h)],
                        "confidence": round(float(conf), 2)
                    }
                    tracked_persons.append(person_data)

                    # Add text to the plotted image (optional, plot() usually does this)
                    # keeping your custom overlay ensures visibility
                    cv2.putText(res_plotted, f"ID:{track_id}", (int(x-w/2), int(y-h/2)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. Publish JSON
            msg_json = String()
            msg_json.data = json.dumps(tracked_persons)
            self.tracking_pub.publish(msg_json)

            # 5. Publish Video
            debug_msg = self.bridge.cv2_to_imgmsg(res_plotted, "bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in tracker loop: {str(e)}')

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()