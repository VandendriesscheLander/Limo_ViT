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

        # --- CONFIGURATION (TUNE THESE FOR YOUR ROBOT) ---
        self.image_width = 640  # Standard camera width
        self.center_x = self.image_width / 2
        
        # 1. Turning Gains (How fast to turn to center the person)
        self.k_turn = 0.005     # 1 pixel error = 0.005 rad/s angular velocity
        self.max_turn_speed = 0.8 # Max rad/s
        
        # 2. Forward Gains (How fast to drive to get closer)
        self.target_box_width = 200 # Desired width of person in pixels (how close to get)
        self.k_drive = 0.002    # Speed multiplier for distance error
        self.max_drive_speed = 0.4 # Max m/s (Keep low for safety initially!)
        self.stop_distance_buffer = 10 # Deadband to prevent jittering when close enough

        # --- STATE VARIABLES ---
        self.current_target_id = None
        self.last_seen_time = 0
        self.latest_detections = []
        self.is_pursuing = False

        # --- ROS 2 INTERFACE ---
        # 1. Listen for the "Brain" to tell us WHO to chase
        self.target_sub = self.create_subscription(
            String, '/pursuit_target', self.target_callback, 10)

        # 2. Listen for the "Eyes" to tell us WHERE everyone is
        self.tracker_sub = self.create_subscription(
            String, '/tracked_persons', self.tracker_callback, 10)

        # 3. Command the Wheels
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 4. Control Loop Timer (Runs at 20Hz for smooth driving)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Pursuit Controller Started. Waiting for target...")

    def target_callback(self, msg):
        """
        Received from VLM/CLIP. Example: {"suspicious": true, "target_id": 1}
        """
        try:
            data = json.loads(msg.data)
            if data.get("suspicious"):
                # Only switch targets if we aren't already locked on, or if logic dictates
                new_id = data.get("target_id")
                # Note: You might need logic here to grab ID from 'tracked_persons' 
                # if CLIP didn't provide it explicitly.
                # For this code, we assume the Brain passes an ID, or we default to the closest person.
                
                # If VLM just says "SUSPICIOUS" without ID (common with CLIP), 
                # we might pursue the person currently closest to center.
                if new_id is None:
                    self.current_target_id = self.find_closest_to_center_id()
                else:
                    self.current_target_id = new_id
                
                if self.current_target_id is not None:
                    self.is_pursuing = True
                    self.get_logger().warn(f"PURSUIT MODE ENGAGED: Target ID {self.current_target_id}")
            else:
                # Optional: Stop pursuing if brain says "False" (Clear)
                # self.is_pursuing = False
                pass 

        except ValueError:
            pass

    def tracker_callback(self, msg):
        """
        Received from Tracker Node. List of all visible people.
        """
        try:
            self.latest_detections = json.loads(msg.data)
        except ValueError:
            self.latest_detections = []

    def control_loop(self):
        """
        Decides how to move the robot every 50ms.
        """
        twist = Twist()

        # Safety: If we haven't received detections in a while, STOP.
        # (Implementation detail: you could add a timestamp check here)

        if not self.is_pursuing or self.current_target_id is None:
            # Idle Mode: Do nothing (or implement Patrol logic here)
            self.cmd_vel_pub.publish(twist)
            return

        # 1. Find our target in the current list of people
        target_person = None
        for person in self.latest_detections:
            if person['id'] == self.current_target_id:
                target_person = person
                break
        
        # 2. If target found, calculate movement
        if target_person:
            self.last_seen_time = time.time()
            
            # --- STEERING (Angular Z) ---
            # Error = Target X - Center X
            # If target is to the left (x < 320), error is negative -> turn left (positive z)
            # Note: ROS coordinates usually are: Left turn = Positive Z
            # But pixels: 0 is left. So: (Center - Target) -> Positive = Left Turn
            
            x_pos = target_person['center'][0]
            error_x = self.center_x - x_pos
            
            # Apply Gain
            angular_z = error_x * self.k_turn
            
            # Clamp Speed
            twist.angular.z = max(min(angular_z, self.max_turn_speed), -self.max_turn_speed)

            # --- DRIVING (Linear X) ---
            # Use box width as a proxy for distance (Wider = Closer)
            current_width = target_person['size'][0]
            width_error = self.target_box_width - current_width
            
            # If width_error > 0 (Too small), drive forward.
            # If width_error < 0 (Too big), stop or reverse.
            
            if abs(width_error) > self.stop_distance_buffer:
                linear_x = width_error * self.k_drive
                twist.linear.x = max(min(linear_x, self.max_drive_speed), -self.max_drive_speed)
            else:
                twist.linear.x = 0.0

            self.get_logger().info(f"Chasing ID {self.current_target_id}: Turn={twist.angular.z:.2f}, Fwd={twist.linear.x:.2f}")

        else:
            # 3. Target Lost Logic
            # If we lost them recently (< 2 seconds), maybe spin to find them?
            if time.time() - self.last_seen_time < 2.0:
                # Lost target recently, stop moving but keep "Pursuing" state active
                twist.linear.x = 0.0
                twist.angular.z = 0.0 
            else:
                # Lost for too long, give up
                self.get_logger().info(f"Target {self.current_target_id} lost. Stopping.")
                self.is_pursuing = False
                self.current_target_id = None
        
        # 4. Send Command
        self.cmd_vel_pub.publish(twist)

    def find_closest_to_center_id(self):
        """Helper to find which ID is currently center-frame (for CLIP targeting)"""
        if not self.latest_detections:
            return None
        
        closest_id = None
        min_dist = 9999
        
        for person in self.latest_detections:
            dist = abs(person['center'][0] - self.center_x)
            if dist < min_dist:
                min_dist = dist
                closest_id = person['id']
        
        return closest_id

def main(args=None):
    rclpy.init(args=args)
    node = PursuitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()