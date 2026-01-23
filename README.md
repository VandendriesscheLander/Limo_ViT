# Patrol Bot

A ROS 2 package designed for the Limo Pro platform that implements autonomous patrol capabilities with a web-based user interface.

## Prerequisites

- **Hardware**: Limo Pro platform.
- **Software**: ROS 2 environment (assumed setup in `limo_ros2_ws`).

## Installation

1. **Navigate to the Workspace**:
   On the Limo Pro, navigate to your ROS 2 workspace `src` directory:
   ```bash
   cd ~/limo_ros2_ws/src
   ```

2. **Clone the Repository**:
   Clone this repository into the `src` folder (if not already present).

3. **Set up Python Environment**:
   Navigate to the `patrol_bot` directory inside the repository and set up a virtual environment:
   ```bash
   cd patrol_bot
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

5. **Build the Project**:
   Build the package using `colcon`:
   ```bash
   cd ~/limo_ros2_ws
   colcon build --packages-select patrol_bot --symlink-install
   ```

## Usage

**⚠️ Safety Warning**: Ensure the robot is placed on the ground before launching the system. The patrol behavior will begin immediately once the main control node is active.

To launch the system, you will need two separate terminals.

**Terminal 1: Base Driver**
Launch the Limo base drivers:
```bash
ros2 launch limo_bringup limo_start.launch.py
```

**Terminal 2: Patrol System**
Launch the patrol bot logic and web UI:
```bash
source install/setup.bash
ros2 launch patrol_bot patrol.launch.py
```

This will start the patrol behavior and the web-based user interface.
