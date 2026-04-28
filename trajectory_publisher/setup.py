import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'trajectory_publisher'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Agustín Capovilla',
    maintainer_email='acapovilla@sinc.unl.edu.ar',
    description='A ROS2 package that publishes paths based on various parameterizations to test path-following controllers.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cubic_spline_publisher = trajectory_publisher.cubic_spline_trajectory_publisher:main',
            'square_publisher = trajectory_publisher.square_trajectory_publisher:main',
            'trajectory_recorder = trajectory_publisher.trajectory_recorder:main'
        ],
    },
)
