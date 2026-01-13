from setuptools import setup

package_name = 'PatrolBotgit'

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
            # format: 'node_name = package_name.script_name:main'
            'tracker_node = PatrolBot.tracker_node:main',
            'clip_brain_node = PatrolBot.clip_brain_node:main',
            'control_node = PatrolBot.control_node:main',
            'web_ui_node = PatrolBot.web_ui_node:main'
        ],
    },
)