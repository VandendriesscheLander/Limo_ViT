#!/bin/bash
# Patrol Bot Launcher for LIMO Pro
# This script launches all required ROS2 nodes for the patrol bot system

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Source the workspace
source /home/agilex/limo_ros2_ws/install/setup.bash

# Export any required environment variables
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

echo "=========================================="
echo "   Starting Patrol Bot System"
echo "=========================================="
echo ""

# Launch limo_bringup in background
echo "[1/2] Starting LIMO base drivers..."
ros2 launch limo_bringup limo_start.launch.py &
LIMO_PID=$!

# Give the base time to initialize
sleep 3

# Launch patrol_bot
echo "[2/2] Starting Patrol Bot nodes..."
ros2 launch patrol_bot patrol.launch.py &
PATROL_PID=$!

echo ""
echo "=========================================="
echo "   Patrol Bot System Running"
echo "   Press Ctrl+C to stop all nodes"
echo "=========================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Patrol Bot System..."
    kill $PATROL_PID 2>/dev/null
    kill $LIMO_PID 2>/dev/null
    # Kill any remaining ROS2 nodes from this session
    pkill -P $$ 2>/dev/null
    echo "Shutdown complete."
    exit 0
}

# Trap Ctrl+C and terminal close
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
