#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import json
import time

class PursuitController(Node):
    def __init__(self):
        super().__init__('pursuit_controller')

        # --- TUNING PARAMETERS ---
        self.image_width = 640
        self.center_x = self.image_width / 2

        # 1. STEERING
        self.k_turn = 0.004
        self.turn_deadband = 80
        self.max_turn_speed = 1.0

        # 2. DRIVING
        self.target_box_width = 200
        self.k_drive = 0.007
        self.max_drive_speed = 3.0
        self.dist_deadband = 10
        
        self.lost_target_timeout = 5.0
        self.arrived_timeout = 6.0

        # --- STATE ---
        self.arrived_start_time = None
        self.current_target_id = None
        self.last_seen_time = 0
        self.latest_detections = []
        self.is_pursuing = False
        
        # Store latest patrol command (from SLAM node)
        self.patrol_twist = Twist()

        # --- ROS SETUP ---
        self.target_sub = self.create_subscription(String, '/pursuit_target', self.target_callback, 10)
        self.tracker_sub = self.create_subscription(String, '/tracked_persons', self.tracker_callback, 10)
        
        # SUBSCRIBE TO PATROL NODE
        self.patrol_sub = self.create_subscription(Twist, '/cmd_vel_patrol', self.patrol_callback, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(Bool, '/pursuit_status', 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pursuit Controller + Patrol Integration Ready.")

    def target_callback(self, msg):
        if self.is_pursuing:
            return

        try:
            data = json.loads(msg.data)
            if data.get("suspicious"):
                if data.get("target_id") is not None:
                    self.current_target_id = data.get("target_id")
                    self.is_pursuing = True
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
            
    def patrol_callback(self, msg):
        # Always update the latest available patrol command
        self.patrol_twist = msg

    def control_loop(self):
        # Publish pursuit status for other nodes (e.g. to helper nodes to sleep)
        status_msg = Bool()
        status_msg.data = self.is_pursuing
        self.status_pub.publish(status_msg)

        twist = Twist()

        # --- MODE SWITCHING ---
        # If we have LOST the target for > 2.0 seconds, we are in PATROL mode.
        if not self.is_pursuing or self.current_target_id is None:
            # Pass through the Patrol/SLAM command
            self.cmd_vel_pub.publish(self.patrol_twist)
            return

        # --- PURSUIT MODE ---
        # 2. Find Target
        target = None
        for person in self.latest_detections:
            if person['id'] == self.current_target_id:
                target = person
                break

        if target:
            self.last_seen_time = time.time()
            
            # Angular
            center_x = target['center'][0]
            error_x = self.center_x - center_x
            if abs(error_x) < self.turn_deadband:
                twist.angular.z = 0.0
            else:
                twist.angular.z = float(error_x * self.k_turn)
            twist.angular.z = max(min(twist.angular.z, self.max_turn_speed), -self.max_turn_speed)

            # Linear
            width = target['size'][0]
            error_width = self.target_box_width - width
            if abs(error_width) < self.dist_deadband:
                twist.linear.x = 0.0
            elif error_width > 0:
                twist.linear.x = float(error_width * self.k_drive)
                if abs(twist.angular.z) > 0.5:
                    twist.linear.x *= 0.5
            else:
                twist.linear.x = 0.0
            twist.linear.x = max(min(twist.linear.x, self.max_drive_speed), 0.0)
            
            # Check for arrival (stopped/near)
            if twist.linear.x == 0.0:
                if self.arrived_start_time is None:
                    self.arrived_start_time = time.time()
                elif time.time() - self.arrived_start_time > self.arrived_timeout:
                    self.get_logger().info("Arrived at target. Resuming patrol.")
                    self.is_pursuing = False
                    self.current_target_id = None
                    self.arrived_start_time = None
                    # No return needed, just won't drive next loop
            else:
                self.arrived_start_time = None

            # Publish Pursuit Command
            self.cmd_vel_pub.publish(twist)

        else:
            self.arrived_start_time = None
            # Target currently not visible
            if time.time() - self.last_seen_time > self.lost_target_timeout:
                self.is_pursuing = False
                self.current_target_id = None
                self.get_logger().info("Target lost. Switching to Patrol Mode.")
                # We do not publish here, next loop will catch 'if not self.is_pursuing'
            else:
                # While momentarily lost (buffer zone), stay still or ghost ride
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