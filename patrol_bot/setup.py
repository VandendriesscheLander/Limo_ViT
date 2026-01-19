from setuptools import setup

package_name = 'patrol_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/patrol.launch.py']),
        ('share/' + package_name, ['yolov8n.engine']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Patrol Robot Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = patrol_bot.tracker_node:main',
            'clip_brain_node = patrol_bot.clip_brain_node:main',
            'control_node = patrol_bot.control_node:main',
            'web_ui_node = patrol_bot.web_ui_node:main',
            'slam_node = patrol_bot.slam_node:main'
        ],
    },
)