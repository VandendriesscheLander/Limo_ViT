from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Environment variable for specific nodes to fix static TLS block error
    # Instead of global os.environ, we pass it to specific nodes
    ld_preload = {'LD_PRELOAD': '/usr/lib/aarch64-linux-gnu/libgomp.so.1'}

    return LaunchDescription([
        # 1. Tracker Node
        Node(
            package='PatrolBot',
            executable='tracker_node',
            name='tracker',
            output='screen',
            additional_env=ld_preload
        ),
        # 2. VLM Node
        Node(
            package='PatrolBot',
            executable='clip_brain_node',
            name='vlm_reasoning',
            output='screen',
            additional_env=ld_preload
        ),
        # 3. Pursuit Control Node
        Node(
            package='PatrolBot',
            executable='control_node',
            name='pursuit_controller',
            output='screen',
            additional_env=ld_preload
        ),
        # 4. Web UI Node (Flask Dashboard)
        Node(
            package='PatrolBot',
            executable='web_ui_node',
            name='web_dashboard',
            output='screen'
        )
    ])