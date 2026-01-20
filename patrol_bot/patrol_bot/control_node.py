#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import json
import time
import math

class PursuitController(Node):
    def __init__(self):
        super().__init__('pursuit_controller')

        # --- TUNING ---
        self.image_width = 640
        self.center_x = self.image_width / 2

        # Steering
        self.k_turn = 0.005
        self.max_turn_speed = 1.2
        self.turn_deadband = 40  # pixels

        # Driving
        self.target_box_width = 180 
        self.k_drive = 0.006
        self.max_drive_speed = 0.6 # Slower for safety during testing
        self.dist_deadband = 20

        # Timeouts
        self.lost_target_timeout = 1.5  # Reduced from 5.0s to 1.5s to prevent "freezing"
        self.patrol_timeout = 0.5       # If no patrol cmd received, stop

        # --- STATE ---
        self.mode = "PATROL"  # Options: PATROL, INVESTIGATE, PURSUIT
        self.target_id = None
        self.last_seen_time = 0
        self.latest_detections = []
        
        # Patrol State
        self.patrol_twist = Twist()
        self.last_patrol_cmd_time = 0

        # --- ROS SETUP ---
        self.target_sub = self.create_subscription(String, '/pursuit_target', self.brain_callback, 10)
        self.tracker_sub = self.create_subscription(String, '/tracked_persons', self.tracker_callback, 10)
        self.patrol_sub = self.create_subscription(Twist, '/cmd_vel_patrol', self.patrol_callback, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        
        # Control Loop (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Control Node Ready (Multi-Mode).")

    def patrol_callback(self, msg):
        self.patrol_twist = msg
        self.last_patrol_cmd_time = time.time()

    def tracker_callback(self, msg):
        try:
            self.latest_detections = json.loads(msg.data)
        except ValueError:
            pass

    def brain_callback(self, msg):
        try:
            data = json.loads(msg.data)
            tid = data.get("target_id")
            conf = data.get("confidence", 0.0)
            
            # 1. PURSUIT MODE: High confidence danger
            if data.get("suspicious") and conf > 0.55:
                if self.mode != "PURSUIT":
                    self.get_logger().warn(f"SWITCHING TO PURSUIT: Target {tid}")
                self.mode = "PURSUIT"
                self.target_id = tid
            
            # 2. INVESTIGATE MODE: Low confidence danger OR Safe but curious
            # (We only investigate if we aren't already chasing someone else)
            elif self.mode != "PURSUIT" and tid is not None:
                # If we were patrolling, maybe just glance at them
                if self.mode == "PATROL":
                    self.mode = "INVESTIGATE"
                    self.target_id = tid
                    
        except ValueError:
            pass

    def control_loop(self):
        twist = Twist()
        now = time.time()
        
        # --- 1. FIND TARGET (if needed) ---
        target = None
        if self.target_id is not None:
            for person in self.latest_detections:
                if person['id'] == self.target_id:
                    target = person
                    self.last_seen_time = now
                    break

        # --- 2. MODE SWITCHING LOGIC ---
        
        # If we haven't seen the target in a while, downgrade mode
        if now - self.last_seen_time > self.lost_target_timeout:
            if self.mode in ["PURSUIT", "INVESTIGATE"]:
                self.get_logger().info("Target lost. Resuming Patrol.")
                self.mode = "PATROL"
                self.target_id = None

        # --- 3. EXECUTION ---
        
        if self.mode == "PATROL":
            # Failsafe: If SLAM node died, don't keep driving the last command
            if now - self.last_patrol_cmd_time < 1.0:
                self.cmd_vel_pub.publish(self.patrol_twist)
            else:
                # Stop if no patrol commands
                self.cmd_vel_pub.publish(Twist())

        elif self.mode == "INVESTIGATE":
            # Logic: Keep moving forward (patrol speed), but TURN towards target
            # This creates a "swooping" motion rather than a stop
            
            # Base forward speed from patrol, or default crawl
            twist.linear.x = 0.15 
            
            if target:
                # Calculate turn to center the target
                center_x = target['center'][0]
                error_x = self.center_x - center_x
                twist.angular.z = float(error_x * self.k_turn)
            else:
                # If target momentarily lost during investigate, just coast
                twist.angular.z = 0.0

            # Publish
            twist.angular.z = max(min(twist.angular.z, 1.0), -1.0)
            self.cmd_vel_pub.publish(twist)

        elif self.mode == "PURSUIT":
            if target:
                # Aggressive Turning
                center_x = target['center'][0]
                error_x = self.center_x - center_x
                twist.angular.z = float(error_x * self.k_turn)
                
                # Aggressive Driving (Stop at distance)
                width = target['size'][0]
                error_width = self.target_box_width - width
                
                if error_width > 0:
                    twist.linear.x = float(error_width * self.k_drive)
                else:
                    twist.linear.x = 0.0 # Stop if too close

                # Clamp
                twist.linear.x = max(min(twist.linear.x, self.max_drive_speed), 0.0)
                twist.angular.z = max(min(twist.angular.z, self.max_turn_speed), -self.max_turn_speed)
                
                self.cmd_vel_pub.publish(twist)
            else:
                # Target lost momentarily?
                # Ghost ride: Decelerate slowly instead of hard stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)

        # Status Update
        status = String()
        status.data = f"Mode: {self.mode} | ID: {self.target_id}"
        self.status_pub.publish(status)

def main(args=None):
    rclpy.init(args=args)
    node = PursuitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()