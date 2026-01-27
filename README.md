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

6. **Create Desktop Launcher** (Optional but Recommended):
   Set up the desktop shortcut for easy launching:
   ```bash
   # Make the launch script executable
   chmod +x ~/limo_ros2_ws/src/patrol_bot/scripts/launch_patrol.sh
   
   # Copy desktop file to Desktop and make it executable
   cp ~/limo_ros2_ws/src/patrol_bot/scripts/PatrolBot.desktop ~/Desktop/
   chmod +x ~/Desktop/PatrolBot.desktop
   
   # Mark as trusted (enables double-click launching)
   gio set ~/Desktop/PatrolBot.desktop metadata::trusted true

    # This is required to be able to run it as an executable
   desktop-file-install --dir=/home/agilex/.local/share/applications /home/agilex/Desktop/PatrolBot.desktop && cp /home/agilex/.local/share/applications/PatrolBot.desktop /home/agilex/Desktop/ && chmod +x /home/agilex/Desktop/PatrolBot.desktop
   ```
   
   A "Patrol Bot" icon will appear on your desktop for one-click launching.

## Usage

**⚠️ Safety Warning**: Ensure the robot is placed on the ground before launching the system. The patrol behavior will begin immediately once the main control node is active.

### Easy Launch (Recommended)

After installation, you'll find a **"Patrol Bot"** icon on the desktop. Simply double-click it to start the entire system. This will automatically:
- Launch the Limo base drivers
- Start all patrol bot nodes
- Open the web-based user interface

Press `Ctrl+C` in the terminal window to stop all nodes.

### Manual Launch (Advanced)

If you prefer manual control or need to debug, you can launch the system using two separate terminals:

**Terminal 1: Base Driver**
```bash
ros2 launch limo_bringup limo_start.launch.py
```

**Terminal 2: Patrol System**
```bash
source install/setup.bash
ros2 launch patrol_bot patrol.launch.py
```

Alternatively, use the launch script directly:
```bash
~/limo_ros2_ws/src/patrol_bot/scripts/launch_patrol.sh
```
