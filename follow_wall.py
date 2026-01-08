import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower')
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile)
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # --- PARAMETERS ---
        self.target_dist = 0.55       # Target distance from wall
        self.wall_found_thresh = 1.0  # If wall is closer than this, we are "following"
        self.hallway_thresh = 1.5     # If walls are further than this, we are in "open space"
        self.front_obs_dist = 0.55    # Obstacle Panic Distance
        
        # PID Gains (Lowered KP slightly to reduce jitter)
        self.kp = 1.8 
        self.kd = 8.0 # High KD helps dampen the wiggle
        
        # Smoothing buffers (Stores last 5 readings to average them)
        self.scan_history = {
            'front': [],
            'right': [],
            'f_right': []
        }
        
        # To handle "Cutting the corner"
        self.lost_wall_timer = 0  # Counter to delay turning when wall ends

    def get_smoothed_range(self, msg, angle_deg, key):
        """
        Gets a range and smooths it using a moving average of the last 3 frames.
        This removes 'jitter' from noisy lidar data.
        """
        # 1. Get raw reading
        angle_rad = math.radians(angle_deg)
        center_idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
        
        # Use a small sector (e.g. 10 indices) to get robust raw data
        sector = 10 
        start = max(0, center_idx - sector)
        end = min(len(msg.ranges), center_idx + sector)
        slice_data = [r for r in msg.ranges[start:end] if 0.05 < r < 10.0]
        
        if not slice_data:
            raw_val = 10.0
        else:
            raw_val = min(slice_data)

        # 2. Add to history buffer
        self.scan_history[key].append(raw_val)
        if len(self.scan_history[key]) > 3: # Keep last 3 frames
            self.scan_history[key].pop(0)
            
        # 3. Return Average
        return sum(self.scan_history[key]) / len(self.scan_history[key])

    def listener_callback(self, msg):
        cmd = Twist()
        
        # --- GET SMOOTHED SENSOR DATA ---
        front = self.get_smoothed_range(msg, 0, 'front')
        right = self.get_smoothed_range(msg, -90, 'right')
        f_right = self.get_smoothed_range(msg, -45, 'f_right') # Front-Right Diagonal
        
        # --- PID CALCULATION ---
        error_p = self.target_dist - right
        
        # Derivative: (Current Right - Front Right)
        # If f_right is smaller, we are angling IN -> Positive correction needed
        error_d = (right - f_right)

        # Output
        steer = (self.kp * error_p) + (self.kd * error_d)
        
        print(f"F: {front:.2f} | R: {right:.2f} | FR: {f_right:.2f}")

        # --- LOGIC FLOW ---

        # 1. OBSTACLE AHEAD -> PIVOT TURN LEFT
        if front < self.front_obs_dist:
            print("!!! OBSTACLE - PIVOT !!!")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.9 
            self.lost_wall_timer = 0 # Reset timer

        # 2. WALL FOLLOWING (Standard)
        elif right < self.wall_found_thresh:
            print(f"Following | Steer: {steer:.2f}")
            self.lost_wall_timer = 0 # Reset timer
            
            cmd.linear.x = 0.25 # Consistent speed
            cmd.angular.z = max(min(steer, 1.5), -1.5)

        # 3. OPEN HALLWAY (No walls nearby)
        # If right is far AND front is clear, drive STRAIGHT.
        elif right > self.hallway_thresh and front > 2.0:
            print(">>> HALLWAY - DRIVING STRAIGHT <<<")
            self.lost_wall_timer = 0
            
            cmd.linear.x = 0.35 # Go faster in straight line
            cmd.angular.z = 0.0 # Don't turn!

        # 4. CORNER RECOVERY (Wall ended abruptly)
        else:
            # We lost the wall, but we might be at a corner.
            # Don't turn immediately, or we clip the wall.
            self.lost_wall_timer += 1
            
            if self.lost_wall_timer < 15: 
                # For the first ~15 cycles (approx 0.5 - 1.0 sec), drive STRAIGHT
                print("Corner detected - Extending...")
                cmd.linear.x = 0.2
                cmd.angular.z = 0.0
            else:
                # After delay, start searching (curving right)
                print("Searching/Turning Right...")
                cmd.linear.x = 0.15
                cmd.angular.z = -0.6

        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()