#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json
import time

class PursuitController(Node):
    def __init__(self):
        super().__init__('pursuit_controller')

        # --- TUNING PARAMETERS ---
        self.image_width = 640
        self.center_x = self.image_width / 2

        # 1. STEERING (Lower gain + Deadband is the secret to stability)
        self.k_turn = 0.004      # Lowered from 0.006 to reduce overshoot
        self.turn_deadband = 40  # If target is within 40px of center, DON'T turn.
        self.max_turn_speed = 1.0

        # 2. DRIVING
        self.target_box_width = 200
        self.k_drive = 0.005     # Moderate acceleration
        self.max_drive_speed = 2.0
        self.dist_deadband = 10  # If size is within 10px of target, stop driving

        # --- STATE ---
        self.current_target_id = None
        self.last_seen_time = 0
        self.latest_detections = []
        self.is_pursuing = False

        # --- ROS SETUP ---
        self.target_sub = self.create_subscription(String, '/pursuit_target', self.target_callback, 10)
        self.tracker_sub = self.create_subscription(String, '/tracked_persons', self.tracker_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Run at 20Hz (0.05s). 
        # CAUTION: If your camera is slow (e.g., 5fps), this loop will run too fast 
        # and reuse old data. 20Hz is usually safe for standard webcams.
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pursuit Controller: Oscillation Fix Applied.")

    def target_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("suspicious"):
                # If we get a specific ID, lock it.
                if data.get("target_id") is not None:
                    self.current_target_id = data.get("target_id")
                    self.is_pursuing = True
                # If no ID provided but we aren't pursuing, grab closest
                elif not self.is_pursuing:
                    self.current_target_id = self.find_closest_to_center_id()
                    if self.current_target_id: self.is_pursuing = True
        except ValueError:
            pass

    def tracker_callback(self, msg):
        try:
            self.latest_detections = json.loads(msg.data)
        except ValueError:
            pass

    def control_loop(self):
        twist = Twist()

        # 1. Safety Check: If not pursuing or target lost recently, stop.
        if not self.is_pursuing or self.current_target_id is None:
            self.cmd_vel_pub.publish(twist) # Send 0,0
            return

        # 2. Find Target
        target = None
        for person in self.latest_detections:
            if person['id'] == self.current_target_id:
                target = person
                break

        if target:
            self.last_seen_time = time.time()
            
            # --- ANGULAR CONTROL (With Deadband) ---
            center_x = target['center'][0]
            error_x = self.center_x - center_x # Positive = Target is to the LEFT

            # THE FIX: If error is small, send 0.0. This stops the wiggling.
            if abs(error_x) < self.turn_deadband:
                twist.angular.z = 0.0
            else:
                twist.angular.z = float(error_x * self.k_turn)
            
            # Clamp Angular
            twist.angular.z = max(min(twist.angular.z, self.max_turn_speed), -self.max_turn_speed)

            # --- LINEAR CONTROL (No Reverse) ---
            width = target['size'][0]
            error_width = self.target_box_width - width # Positive = Target is far away (small)

            if abs(error_width) < self.dist_deadband:
                twist.linear.x = 0.0
            elif error_width > 0:
                # Target is too small (far away) -> Drive Forward
                twist.linear.x = float(error_width * self.k_drive)
                
                # Turn-Aware Throttling (Simplified)
                # If we are turning fast, slow down linear speed to 50%
                if abs(twist.angular.z) > 0.5:
                    twist.linear.x *= 0.5
            else:
                # Target is too big (too close) -> STOP. Do not reverse.
                twist.linear.x = 0.0

            # Clamp Linear
            twist.linear.x = max(min(twist.linear.x, self.max_drive_speed), 0.0)

        else:
            # Target currently not visible
            if time.time() - self.last_seen_time > 2.0:
                self.is_pursuing = False
                self.current_target_id = None
                self.get_logger().info("Target lost. Stopping.")
            
            # While lost (but < 2s), send 0 commands so we don't ghost ride
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def find_closest_to_center_id(self):
        if not self.latest_detections: return None
        closest = None
        min_dist = 9999
        for p in self.latest_detections:
            d = abs(p['center'][0] - self.center_x)
            if d < min_dist:
                min_dist = d
                closest = p['id']
        return closest

def main(args=None):
    rclpy.init(args=args)
    node = PursuitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()