from setuptools import find_packages, setup
import os

package_name = "audio_stream_manager"

launch_files = [f"launch/{f}" for f in os.listdir("launch") if f.endswith(".launch.py")]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, launch_files),
        ("share/" + package_name + "/config", ["config/audio_params.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "sounddevice"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="joan.omedes@eurecat.org",
    description="Audio stream management for ROS2",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "audio_capturing = audio_stream_manager.audio_capturing:main",
            "audio_to_mp3 = audio_stream_manager.audio_to_mp3:main",
            "android_audio_bridge = audio_stream_manager.android_audio_bridge:main",
        ],
    },
)
