from setuptools import find_packages, setup

package_name = 'buddy_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ashandth',
    maintainer_email='ashandth0309@gmail.com',
    description='Core runtime package for the BUDDY intelligent robot dog',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'core_smoke_test = buddy_core.core_smoke_test:main',
        'config_smoke_test = buddy_core.config_smoke_test:main',
        'logging_smoke_test = buddy_core.logging_smoke_test:main',
        'obstacle_sensor = buddy_core.sensors.obstacle_node:main',
        'safety_supervisor = buddy_core.safety.safety_node:main',
        'motor_executor = buddy_core.motors.motor_node:main',
    ],
},
)
