from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Tracker Node
        Node(
            package='PatrolBot',
            executable='tracker_node',
            name='tracker',
            output='screen'
        ),
        # 2. VLM Node
        Node(
            package='PatrolBot',
            executable='clip_brain_node',
            name='vlm_reasoning',
            output='screen'
        ),
        # 3. Pursuit Control Node
        Node(
            package='PatrolBot',
            executable='control_node',
            name='pursuit_controller',
            output='screen'
        ),
        # 4. Web UI Node (Flask Dashboard)
        Node(
            package='PatrolBot',
            executable='web_ui_node',
            name='web_dashboard',
            output='screen'
        )
    ])