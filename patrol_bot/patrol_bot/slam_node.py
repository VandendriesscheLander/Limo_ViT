#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class SmartPatrolNode(Node):
    def __init__(self):
        super().__init__('smart_patrol_node')

        # --- CRITICAL CHANGE: Publish to intermediate topic ---
        self.scan_topic = '/scan'
        self.cmd_topic = '/cmd_vel_patrol' 
        
        # Brain Status
        self.brain_ready = False
        self.create_subscription(Bool, '/brain_ready', self.brain_callback, 10) 
        
        # Navigation Parameters
        self.stop_distance = 0.5    
        self.wall_buffer = 0.6      
        self.drive_speed = 0.3      
        self.turn_speed = 0.7       

        # Hysteresis (Prevents jittery decision switching)
        self.last_turn_dir = 0.0 # 0: None, 1: Left, -1: Right

        # QoS Setup
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_policy
        )
        self.publisher = self.create_publisher(Twist, self.cmd_topic, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.latest_scan = None
        self.get_logger().info("Smart Patrol Ready (Feeding /cmd_vel_patrol).")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def brain_callback(self, msg):
        self.brain_ready = msg.data
        if self.brain_ready:
           self.get_logger().info("Brain Ready. Starting Patrol.", once=True)

    def get_sector_distance(self, ranges, start_idx, end_idx):
        valid_ranges = []
        for r in ranges[start_idx:end_idx]:
            if r > 0.05 and r < 10.0:
                valid_ranges.append(r)
            elif r == float('inf'):
                valid_ranges.append(10.0)
        
        if not valid_ranges: return 0.0
        return min(valid_ranges)

    def control_loop(self):
        if not self.brain_ready:
            return

        if self.latest_scan is None: return

        twist = Twist()
        ranges = self.latest_scan.ranges
        count = len(ranges)
        
        # Slicing (Assume 0 is Back/Right, Middle is Front)
        mid_idx = count // 2
        one_third = count // 6 
        
        dist_right = self.get_sector_distance(ranges, mid_idx - (one_third * 2), mid_idx - one_third)
        dist_front = self.get_sector_distance(ranges, mid_idx - (one_third // 2), mid_idx + (one_third // 2))
        dist_left = self.get_sector_distance(ranges, mid_idx + one_third, mid_idx + (one_third * 2))

        # --- LOGIC WITH HYSTERESIS ---
        
        # 1. Blocked Front
        if dist_front < self.stop_distance:
            twist.linear.x = 0.0
            
            # If we were already turning, keep turning that way to avoid "shaking"
            if self.last_turn_dir != 0:
                twist.angular.z = self.turn_speed * self.last_turn_dir
            else:
                # Pick a new direction
                if dist_left > dist_right:
                    self.last_turn_dir = 1.0 # Left
                else:
                    self.last_turn_dir = -1.0 # Right
                twist.angular.z = self.turn_speed * self.last_turn_dir
        
        # 2. Path Clear
        else:
            self.last_turn_dir = 0.0 # Reset turn memory
            
            # Wall Centering (Nudge)
            if dist_left < self.wall_buffer:
                twist.linear.x = self.drive_speed * 0.8
                twist.angular.z = -0.3 
            elif dist_right < self.wall_buffer:
                twist.linear.x = self.drive_speed * 0.8
                twist.angular.z = 0.3
            else:
                twist.linear.x = self.drive_speed
                twist.angular.z = 0.0 

        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SmartPatrolNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()