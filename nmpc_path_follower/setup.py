from setuptools import find_packages, setup

package_name = 'nmpc_path_follower'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Guido Sanchez',
    maintainer_email='gsanchez@sinc.unl.edu.ar',
    description='A ROS2 package that uses NMPC to control a UGV along a given path.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'run_mpc_minimal = nmpc_path_follower.MPC_minimal:main',
            'quaternions_mpc_controller = nmpc_path_follower.quaternions_mpc:main',
        ],
    },
)