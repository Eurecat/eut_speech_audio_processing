from setuptools import find_packages, setup
import os

package_name = "speech_recognition"

launch_files = [f"launch/{f}" for f in os.listdir("launch") if f.endswith(".launch.py")]
config_files = [f"config/{f}" for f in os.listdir("config") if f.endswith(".yaml")]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, launch_files),
        ("share/" + package_name + "/config", config_files),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "torch",
        "diart",
        "openwakeword",
        "faster-whisper",
    ],
    zip_safe=True,
    maintainer="root",
    maintainer_email="joan.omedes@eurecat.org",
    description="Speech recognition and processing for ROS2",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vad_node = speech_recognition.vad:main",
            "diarization_node = speech_recognition.diarization:main",
            "asr_node = speech_recognition.asr:main",
            "wake_word_node = speech_recognition.wake_word:main",
        ],
    },
)
