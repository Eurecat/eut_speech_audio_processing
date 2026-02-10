from setuptools import find_packages, setup

package_name = "audio_stream_manager"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/audio_stream_manager.launch.py"]),
        ("share/" + package_name + "/config", ["config/audio_params.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "sounddevice"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="joan.omedes@eurecat.org",
    description="Audio stream management for ROS2",
    license="Apache-2.0",
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "audio_capturing_select_device = audio_stream_manager.audio_capturing_select_device:main",
            "audio_capturing_automatic_device = audio_stream_manager.audio_capturing_automatic_device:main",
            "audio_capturing_automatic_device_node = audio_stream_manager.audio_capturing_automatic_device:main",
            "save_mp3 = audio_stream_manager.save_mp3:main",
        ],
    },
)
