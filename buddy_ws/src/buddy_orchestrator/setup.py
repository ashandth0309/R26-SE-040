from glob import glob

from setuptools import find_packages, setup

package_name = 'buddy_orchestrator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='S. S. Y. Wickramasinghe',
    maintainer_email='yohanwick17@gmail.com',
    description=(
        'Central coordinator that connects the BUDDY voice, vision and '
        'navigation components without owning any hardware'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'buddy_coordinator = '
            'buddy_orchestrator.nodes.coordinator_node:main',
        ],
    },
)
